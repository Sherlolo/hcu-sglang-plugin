#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path


def git_files(repo: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-co", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(set(result.stdout.splitlines()))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_map(repo: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for relative in git_files(repo):
        path = repo / relative
        if path.is_file():
            result[relative] = {"size": path.stat().st_size, "sha256": sha256(path)}
    return result


def python_summary(repo: Path, files: dict[str, dict[str, object]]) -> dict[str, object]:
    modules: dict[str, object] = {}
    parse_errors: dict[str, str] = {}
    for relative in files:
        if not relative.endswith(".py"):
            continue
        path = repo / relative
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception as exc:
            parse_errors[relative] = repr(exc)
            continue

        functions: list[str] = []
        classes: list[str] = []
        imports: list[str] = []
        strings: list[str] = []
        hook_targets: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
            elif isinstance(node, ast.Call) and node.args:
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                first = node.args[0]
                if name in {"register", "plugin_hook"} and isinstance(first, ast.Constant) and isinstance(first.value, str):
                    hook_targets.append(first.value)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "sglang." in node.value:
                    strings.append(node.value)

        modules[relative] = {
            "functions": sorted(set(functions)),
            "classes": sorted(set(classes)),
            "imports": sorted(set(imports)),
            "sglang_target_strings": sorted(set(strings)),
            "literal_hook_targets": sorted(set(hook_targets)),
        }
    return {"modules": modules, "parse_errors": parse_errors}


def compare(left: dict[str, dict[str, object]], right: dict[str, dict[str, object]]):
    left_names = set(left)
    right_names = set(right)
    common = left_names & right_names
    return {
        "only_left": sorted(left_names - right_names),
        "only_right": sorted(right_names - left_names),
        "modified": sorted(
            name for name in common if left[name]["sha256"] != right[name]["sha256"]
        ),
        "identical": sorted(
            name for name in common if left[name]["sha256"] == right[name]["sha256"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--compare", action="append", default=[], metavar="LEFT=RIGHT")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    repos: dict[str, Path] = {}
    for item in args.repo:
        name, raw_path = item.split("=", 1)
        repos[name] = Path(raw_path).resolve()

    report: dict[str, object] = {"repositories": {}, "comparisons": {}}
    maps: dict[str, dict[str, dict[str, object]]] = {}
    for name, repo in repos.items():
        files = file_map(repo)
        maps[name] = files
        report["repositories"][name] = {
            "path": str(repo),
            "file_count": len(files),
            "files": files,
            "python": python_summary(repo, files),
        }

    for item in args.compare:
        left, right = item.split("=", 1)
        report["comparisons"][f"{left}_vs_{right}"] = compare(
            maps[left], maps[right]
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
