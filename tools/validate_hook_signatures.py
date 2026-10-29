#!/usr/bin/env python3
"""Statically validate lazy plugin REPLACE hooks against an SGLang checkout."""

from __future__ import annotations

import argparse
import ast
import json
import importlib
import sys
from pathlib import Path

if __package__:
    from .plugin_layout import resolve_plugin_package
else:
    from plugin_layout import resolve_plugin_package
from typing import Any


def module_file(root: Path, module_name: str) -> Path | None:
    relative = Path(*module_name.split("."))
    file_path = root / relative.with_suffix(".py")
    if file_path.is_file():
        return file_path
    init_path = root / relative / "__init__.py"
    return init_path if init_path.is_file() else None


def split_target(root: Path, dotted_path: str) -> tuple[Path, list[str]]:
    parts = dotted_path.split(".")
    for index in range(len(parts), 0, -1):
        path = module_file(root, ".".join(parts[:index]))
        if path is not None:
            return path, parts[index:]
    raise FileNotFoundError(f"No module file found for {dotted_path}")


def find_node(path: Path, attrs: list[str]) -> ast.AST:
    node: ast.AST = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for attr in attrs:
        body = getattr(node, "body", ())
        match = next(
            (
                item
                for item in body
                if isinstance(
                    item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                )
                and item.name == attr
            ),
            None,
        )
        if match is None:
            raise AttributeError(f"{path}: missing AST attribute {'.'.join(attrs)}")
        node = match
    return node


def signature(node: ast.AST) -> dict[str, Any]:
    if isinstance(node, ast.ClassDef):
        return {"kind": "class"}
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise TypeError(f"Expected function or class node, got {type(node).__name__}")
    args = node.args
    return {
        "kind": "function",
        "posonly": [arg.arg for arg in args.posonlyargs],
        "positional": [arg.arg for arg in args.args],
        "vararg": args.vararg.arg if args.vararg else None,
        "kwonly": [arg.arg for arg in args.kwonlyargs],
        "kwarg": args.kwarg.arg if args.kwarg else None,
        "defaults": len(args.defaults),
        "kw_defaults": [value is not None for value in args.kw_defaults],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-python", type=Path, required=True)
    parser.add_argument("--plugin-root", type=Path, required=True)
    parser.add_argument("--plugin-package", choices=("hcu_sglang", "sglang_dcu_plugin"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    package = resolve_plugin_package(args.plugin_root, args.plugin_package)
    sys.path[:0] = [str(args.plugin_root.resolve()), str(args.official_python.resolve())]
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    register = importlib.import_module(package).register

    HookRegistry.reset()
    register()

    rows = []
    for target, hooks in sorted(HookRegistry._hooks.items()):
        for hook_type, hook, _source in hooks:
            source_module = getattr(hook, "__module__", "")
            source_name = getattr(hook, "__name__", "")
            if hook_type is not HookType.REPLACE or not source_module.startswith(
                f"{package}.models"
            ):
                continue

            row = {
                "target": target,
                "source": f"{source_module}.{source_name}",
                "status": "unknown",
            }
            try:
                target_file, target_attrs = split_target(args.official_python, target)
                source_file, source_attrs = split_target(
                    args.plugin_root, f"{source_module}.{source_name}"
                )
                target_sig = signature(find_node(target_file, target_attrs))
                source_sig = signature(find_node(source_file, source_attrs))
                row.update(
                    {
                        "target_file": str(target_file),
                        "source_file": str(source_file),
                        "target_signature": target_sig,
                        "source_signature": source_sig,
                        "status": "match" if target_sig == source_sig else "mismatch",
                    }
                )
            except Exception as exc:
                row.update(
                    {"status": "missing", "error": f"{type(exc).__name__}: {exc}"}
                )
            rows.append(row)

    summary = {
        status: sum(row["status"] == status for row in rows)
        for status in ("match", "mismatch", "missing")
    }
    payload = {"summary": summary, "hooks": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    for row in rows:
        if row["status"] != "match":
            print(row["status"], row["target"], "<-", row["source"])
            if "error" in row:
                print(" ", row["error"])
    return 1 if summary["mismatch"] or summary["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
