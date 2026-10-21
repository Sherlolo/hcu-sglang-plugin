#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any


NON_RUNTIME_DISPOSITIONS = {
    "ci": "test/ci/docs only",
    "docs": "test/ci/docs only",
    "test": "test/ci/docs only",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def plugin_targets(plugin_repo: Path) -> list[str]:
    targets: set[str] = set()
    for path in (plugin_repo / "sglang_dcu_plugin").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "register":
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                if first.value.startswith("sglang."):
                    targets.add(first.value)
            elif isinstance(first, ast.JoinedStr):
                literal = "".join(
                    value.value
                    for value in first.values
                    if isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                )
                if literal.startswith("sglang."):
                    targets.add(literal)
    return sorted(targets)


def target_source_path(target: str, official_repo: Path) -> str | None:
    python_root = official_repo / "python"
    parts = target.split(".")
    for length in range(len(parts), 1, -1):
        module_path = Path(*parts[:length])
        source = python_root / module_path.with_suffix(".py")
        if source.is_file():
            return f"python/{module_path.with_suffix('.py')}"
        package = python_root / module_path / "__init__.py"
        if package.is_file():
            return f"python/{module_path}/__init__.py"
    return None


def source_equal(path: str, fork_repo: Path, official_repo: Path) -> bool:
    fork_path = fork_repo / path
    official_path = official_repo / path
    return (
        fork_path.is_file()
        and official_path.is_file()
        and fork_path.read_bytes() == official_path.read_bytes()
    )


def symbols(item: dict[str, Any]) -> list[str]:
    delta = item.get("symbol_delta") or {}
    result: list[str] = []
    for kind in ("added", "removed", "modified"):
        result.extend(delta.get(kind) or [])
    return sorted(set(result))


def default_entry(
    item: dict[str, Any],
    manifest: dict[str, Any],
    source_targets: dict[str, list[str]],
    fork_repo: Path,
    official_repo: Path,
    runtime_targets: bool = False,
) -> dict[str, Any]:
    path = item["path"]
    classification = item["classification"]
    entry = {
        "id": "",
        "fork_commit": "WORKTREE" if item.get("working_tree") else manifest["baselines"]["fork_head"],
        "fork_path": path,
        "symbols": symbols(item),
        "behavior": "Pending semantic description",
        "official_status": "partial" if item.get("also_changed_upstream") else "missing",
        "plugin_status": "missing",
        "disposition": "pending audit",
        "plugin_target": [],
        "hook_type": "N/A",
        "dependencies": item.get("hardware_signals") or [],
        "static_evidence": [],
        "runtime_evidence": [],
        "status": "pending",
        "risk": "Semantic equivalence has not yet been demonstrated.",
        "classification": classification,
    }

    if classification in NON_RUNTIME_DISPOSITIONS:
        entry.update(
            behavior=f"{classification.upper()} artifact; no production runtime symbol is installed by the plugin.",
            official_status="N/A",
            plugin_status="N/A",
            disposition=NON_RUNTIME_DISPOSITIONS[classification],
            static_evidence=[f"Manifest classification: {classification}"],
            status="excluded",
            risk="Production runtime equivalence is unaffected; validation coverage may still be migrated separately.",
        )
    elif source_equal(path, fork_repo, official_repo):
        entry.update(
            behavior="Fork and pinned official source files are byte-identical.",
            official_status="equivalent",
            plugin_status="N/A",
            disposition="already upstream",
            static_evidence=["Byte-for-byte fork/official source equality"],
            status="excluded",
            risk="Recheck equality whenever either pinned source revision moves.",
        )
    elif path in source_targets:
        targets = source_targets[path]
        entry.update(
            behavior="Hook candidate maps to the corresponding source module; semantic coverage is pending.",
            plugin_status="registered" if runtime_targets else "candidate",
            disposition="pending audit",
            plugin_target=targets,
            hook_type="registered Hook(s)" if runtime_targets else "AST candidate",
            static_evidence=["Runtime target export supplied" if runtime_targets else "AST-only candidate discovery", "Target module file exists; attribute resolution not checked"],
            status="pending",
            risk="Registration, target attribute resolution and semantic coverage require separate evidence.",
        )
    return entry


def apply_override(entry: dict[str, Any], override: dict[str, Any]) -> None:
    for key, value in override.items():
        if key == "append_static_evidence":
            entry["static_evidence"].extend(value)
        elif key == "append_runtime_evidence":
            entry["runtime_evidence"].extend(value)
        else:
            entry[key] = value


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SGLang DAS/HCU Plugin Migration Disposition Ledger",
        "",
        "## Summary",
        "",
        f"- Total items: {report['summary']['total']}",
        f"- Registered Hook targets: {report['summary']['registered_hook_targets']}",
        f"- AST candidate Hook targets: {report['summary']['candidate_hook_targets']}",
        f"- Pending items: {report['summary']['status'].get('pending', 0)}",
        "",
        "### Status",
        "",
    ]
    lines.extend(
        f"- `{name}`: {count}"
        for name, count in sorted(report["summary"]["status"].items())
    )
    lines.extend(["", "### Disposition", ""])
    lines.extend(
        f"- `{name}`: {count}"
        for name, count in sorted(report["summary"]["disposition"].items())
    )
    lines.extend(["", "## Items", ""])
    for item in report["items"]:
        lines.extend(
            [
                f"### {item['id']} — `{item['fork_path']}`",
                "",
                f"- Symbols: {', '.join(item['symbols']) or '-'}",
                f"- Behavior: {item['behavior']}",
                f"- Official status: `{item['official_status']}`",
                f"- Plugin status: `{item['plugin_status']}`",
                f"- Disposition: `{item['disposition']}`",
                f"- Plugin target: {', '.join(item['plugin_target']) or '-'}",
                f"- Hook type: `{item['hook_type']}`",
                f"- Dependencies: {', '.join(item['dependencies']) or '-'}",
                f"- Static evidence: {'; '.join(item['static_evidence']) or '-'}",
                f"- Runtime evidence: {'; '.join(item['runtime_evidence']) or '-'}",
                f"- Status: `{item['status']}`",
                f"- Risk: {item['risk']}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--fork-repo", required=True)
    parser.add_argument("--plugin-repo", required=True)
    parser.add_argument("--hook-targets-file")
    parser.add_argument("--overrides", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args()

    manifest = load_json(Path(args.manifest))
    official_repo = Path(args.official_repo).resolve()
    fork_repo = Path(args.fork_repo).resolve()
    plugin_repo = Path(args.plugin_repo).resolve()
    overrides = load_json(Path(args.overrides))
    prefix_rules = overrides.pop("__prefix_rules__", [])

    if args.hook_targets_file:
        targets = sorted(
            {
                line.strip()
                for line in Path(args.hook_targets_file).read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip().startswith("sglang.")
            }
        )
    else:
        targets = plugin_targets(plugin_repo)
    source_targets: dict[str, list[str]] = {}
    for target in targets:
        source_path = target_source_path(target, official_repo)
        if source_path is not None:
            source_targets.setdefault(source_path, []).append(target)

    entries = []
    for index, item in enumerate(manifest["items"], start=1):
        entry = default_entry(
            item, manifest, source_targets, fork_repo, official_repo, bool(args.hook_targets_file)
        )
        entry["id"] = f"MIG-{index:04d}"
        override = overrides.get(item["path"])
        if override is None:
            for rule in prefix_rules:
                if item["path"].startswith(rule["prefix"]):
                    override = rule["override"]
                    break
        if override is not None:
            apply_override(entry, override)
        entries.append(entry)

    status = Counter(entry["status"] for entry in entries)
    disposition = Counter(entry["disposition"] for entry in entries)
    report = {
        "baselines": manifest["baselines"],
        "summary": {
            "total": len(entries),
            "hook_target_source": "runtime-export" if args.hook_targets_file else "ast-candidates",
            "registered_hook_targets": len(targets) if args.hook_targets_file else None,
            "candidate_hook_targets": None if args.hook_targets_file else len(targets),
            "status": dict(sorted(status.items())),
            "disposition": dict(sorted(disposition.items())),
        },
        "items": entries,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    Path(args.output_markdown).write_text(markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
