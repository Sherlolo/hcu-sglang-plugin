#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import subprocess
from pathlib import Path


def run(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def name_status(repo: Path, start: str, end: str) -> dict[str, str]:
    raw = subprocess.run(
        ["git", "-C", str(repo), "diff", "--name-status", "-z", f"{start}..{end}"],
        check=True,
        capture_output=True,
    ).stdout
    fields = raw.decode("utf-8", errors="surrogateescape").split("\0")
    result: dict[str, str] = {}
    index = 0
    while index < len(fields) and fields[index]:
        status = fields[index]
        index += 1
        if status.startswith(("R", "C")):
            old_path, new_path = fields[index : index + 2]
            index += 2
            result[new_path] = f"{status}:{old_path}"
        else:
            result[fields[index]] = status
            index += 1
    return result


def dirty_paths(repo: Path) -> list[str]:
    raw = run(repo, "status", "--porcelain=v1", "-z")
    paths: list[str] = []
    for field in raw.split("\0"):
        if not field:
            continue
        path = field[3:]
        paths.append(path.split(" -> ", 1)[-1])
    return sorted(set(paths))


def repository_paths(repo: Path) -> set[str]:
    raw = run(repo, "ls-files", "-co", "--exclude-standard")
    return set(raw.splitlines())


def worktree_blob(repo: Path, path: str) -> str | None:
    candidate = repo / path
    if not candidate.is_file():
        return None
    return candidate.read_text(encoding="utf-8", errors="replace")


def blob(repo: Path, revision: str, path: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{path}"],
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.decode("utf-8", errors="replace")


def node_hash(node: ast.AST) -> str:
    value = ast.dump(node, include_attributes=False)
    return hashlib.sha256(value.encode()).hexdigest()


def symbols(source: str | None) -> dict[str, str]:
    if source is None:
        return {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"<file>": hashlib.sha256(source.encode()).hexdigest()}

    result: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = node_hash(node)
        elif isinstance(node, ast.ClassDef):
            result[node.name] = node_hash(node)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result[f"{node.name}.{child.name}"] = node_hash(child)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    result[target.id] = node_hash(node)
    return result


def symbol_delta(before: str | None, after: str | None) -> dict[str, list[str]]:
    left = symbols(before)
    right = symbols(after)
    common = set(left) & set(right)
    return {
        "added": sorted(set(right) - set(left)),
        "removed": sorted(set(left) - set(right)),
        "modified": sorted(name for name in common if left[name] != right[name]),
    }


def classify(path: str, content: str) -> tuple[str, list[str]]:
    lower_path = path.lower()
    lower = content.lower()
    signals = [
        word
        for word in ("dcu", "hcu", "hygon", "lightop", "aiter", "hip", "rocm", "gfx")
        if word in lower
    ]
    if lower_path.startswith(("test/", "tests/")) or "/test" in lower_path:
        return "test", signals
    if lower_path.startswith(".github/") or "/ci/" in lower_path:
        return "ci", signals
    if lower_path.startswith("docs/") or lower_path.endswith((".md", ".rst")):
        return "docs", signals
    if any(token in lower_path for token in ("sgl-kernel", "csrc/", "cmake", "setup.py")) or lower_path.endswith((".c", ".cc", ".cpp", ".cu", ".hip", ".h", ".hpp")):
        return "native_or_packaging", signals
    if lower_path.startswith("python/sglang/srt/") and lower_path.endswith(".py"):
        return "runtime_hook_candidate", signals
    if lower_path.endswith(".py"):
        return "python_candidate", signals
    if lower_path.endswith((".toml", ".yaml", ".yml", ".sh")):
        return "packaging_or_invocation", signals
    return "manual_review", signals


def plugin_env_names(plugin_repo: Path) -> set[str]:
    environ_path = plugin_repo / "sglang_dcu_plugin" / "environ.py"
    if not environ_path.is_file():
        return set()
    tree = ast.parse(environ_path.read_text(encoding="utf-8"))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("SGLANG_"):
                result.add(node.value)
    return result


def added_hardware_signals(
    before: str | None,
    after: str | None,
    tracked_env_names: set[str],
) -> list[str]:
    if after is None:
        return []
    diff = difflib.unified_diff(
        (before or "").splitlines(),
        after.splitlines(),
        lineterm="",
    )
    added_lines: list[str] = []
    for line in diff:
        if not line.startswith("+") or line.startswith("+++"):
            continue
        lower = line[1:].lower()
        if any(
            marker in lower
            for marker in (
                "copyright",
                "licensed under",
                "spdx-license",
                "modified by hygon",
            )
        ):
            continue
        added_lines.append(lower)
    signals = {
        word
        for word in ("dcu", "hcu", "hygon", "lightop", "aiter", "hip", "rocm", "gfx")
        if any(word in line for line in added_lines)
    }
    added_text = "\n".join(added_lines)
    if any(name.lower() in added_text for name in tracked_env_names):
        signals.add("plugin_env")
    return sorted(signals)


def plugin_targets(plugin_repo: Path) -> list[str]:
    targets: set[str] = set()
    for path in (plugin_repo / "sglang_dcu_plugin").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith("sglang."):
                    targets.add(node.value)
    return sorted(targets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--fork-repo", required=True)
    parser.add_argument("--plugin-repo", required=True)
    parser.add_argument("--official-ref", default="refs/remotes/github-official/main")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    official_repo = Path(args.official_repo).resolve()
    fork_repo = Path(args.fork_repo).resolve()
    plugin_repo = Path(args.plugin_repo).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fork_head = run(fork_repo, "rev-parse", "HEAD").strip()
    official_head = run(fork_repo, "rev-parse", args.official_ref).strip()
    if official_head != run(official_repo, "rev-parse", "HEAD").strip():
        raise ValueError("official-ref must match the official worktree HEAD")
    merge_base = run(fork_repo, "merge-base", fork_head, official_head).strip()
    fork_changes = name_status(fork_repo, merge_base, fork_head)
    official_changes = name_status(fork_repo, merge_base, official_head)
    dirty = dirty_paths(fork_repo)
    targets = plugin_targets(plugin_repo)
    tracked_env_names = plugin_env_names(plugin_repo)

    items = []
    primary_paths = set(fork_changes) | set(dirty)
    for path in sorted(primary_paths):
        before = blob(fork_repo, merge_base, path)
        worktree_path = fork_repo / path
        after = (worktree_path.read_text(encoding="utf-8", errors="replace")
                 if worktree_path.is_file() else
                 (None if path in dirty else blob(fork_repo, fork_head, path)))
        classification, signals = classify(path, after or "")
        items.append(
            {
                "path": path,
                "fork_status": fork_changes.get(path),
                "working_tree": path in dirty,
                "also_changed_upstream": path in official_changes,
                "classification": classification,
                "hardware_signals": signals,
                "symbol_delta": symbol_delta(before, after) if path.endswith(".py") else None,
                "source": "fork_delta",
                "direct_worktree_diff": False,
            }
        )

    direct_hardware_drift_paths: list[str] = []
    common_paths = repository_paths(fork_repo) & repository_paths(official_repo)
    for path in sorted(common_paths - primary_paths):
        fork_content = worktree_blob(fork_repo, path)
        official_content = worktree_blob(official_repo, path)
        if fork_content == official_content:
            continue
        classification, _ = classify(path, fork_content or "")
        signals = added_hardware_signals(
            official_content,
            fork_content,
            tracked_env_names,
        )
        if not signals:
            continue
        direct_hardware_drift_paths.append(path)
        items.append(
            {
                "path": path,
                "fork_status": None,
                "working_tree": False,
                "also_changed_upstream": path in official_changes,
                "classification": classification,
                "hardware_signals": signals,
                "symbol_delta": (
                    symbol_delta(official_content, fork_content)
                    if path.endswith(".py")
                    else None
                ),
                "source": "direct_hardware_drift",
                "direct_worktree_diff": True,
            }
        )

    report = {
        "baselines": {
            "official_repo": str(official_repo),
            "fork_repo": str(fork_repo),
            "plugin_repo": str(plugin_repo),
            "official_head": official_head,
            "fork_head": fork_head,
            "merge_base": merge_base,
            "plugin_head": run(plugin_repo, "rev-parse", "HEAD").strip(),
        },
        "counts": {
            "fork_changed_paths": len(fork_changes),
            "official_changed_paths": len(official_changes),
            "overlap_paths": len(set(fork_changes) & set(official_changes)),
            "dirty_fork_paths": len(dirty),
            "plugin_target_strings": len(targets),
            "direct_hardware_drift_paths": len(direct_hardware_drift_paths),
        },
        "dirty_fork_paths": dirty,
        "direct_hardware_drift_paths": direct_hardware_drift_paths,
        "plugin_target_strings": targets,
        "items": items,
    }
    (output_dir / "migration_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    by_class: dict[str, int] = {}
    for item in items:
        key = item["classification"]
        by_class[key] = by_class.get(key, 0) + 1
    lines = [
        "# SGLang DAS to Plugin Migration Manifest",
        "",
        "## Baselines",
        "",
        f"- Official HEAD: `{official_head}`",
        f"- Fork HEAD: `{fork_head}`",
        f"- Merge base: `{merge_base}`",
        f"- Plugin HEAD: `{report['baselines']['plugin_head']}`",
        "",
        "## Counts",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in report["counts"].items())
    lines.extend(["", "## Classification", ""])
    lines.extend(f"- {key}: {value}" for key, value in sorted(by_class.items()))
    lines.extend(["", "## Fork Delta Items", ""])
    for item in items:
        changed: list[str] = []
        if item["symbol_delta"]:
            for kind in ("added", "removed", "modified"):
                changed.extend(f"{kind}:{name}" for name in item["symbol_delta"][kind])
        lines.extend(
            [
                f"### `{item['path']}`",
                "",
                f"- Classification: `{item['classification']}`",
                f"- Fork status: `{item['fork_status']}`",
                f"- Working tree: `{item['working_tree']}`",
                f"- Also changed upstream: `{item['also_changed_upstream']}`",
                f"- Hardware signals: {', '.join(item['hardware_signals']) or '-'}",
                f"- Source: `{item['source']}`",
                f"- Direct worktree diff: `{item['direct_worktree_diff']}`",
                f"- Symbol delta: {', '.join(changed) or '-'}",
                "- Disposition: `PENDING`",
                "- Evidence: `PENDING`",
                "",
            ]
        )
    (output_dir / "migration_manifest.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
