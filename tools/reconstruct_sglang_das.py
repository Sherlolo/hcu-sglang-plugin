#!/usr/bin/env python3
"""Build an embedded SGLang DAS source tree from upstream SGLang and the DCU plugin."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


MANIFEST_NAME = ".sglang-dcu-embedded.json"
MANIFEST_KIND = "sglang-dcu-embedded-source"
GENERATOR_VERSION = 3

PLUGIN_PACKAGE = Path("sglang_dcu_plugin")
UPSTREAM_PLUGIN_LOADER = Path("python/sglang/srt/plugins/__init__.py")
UPSTREAM_PACKAGE_INIT = Path("python/sglang/__init__.py")
UPSTREAM_PYPROJECT = Path("python/pyproject.toml")

LOADER_BEGIN = "    # BEGIN SGLANG DCU EMBEDDED PLUGIN"
LOADER_END = "    # END SGLANG DCU EMBEDDED PLUGIN"
LOADER_ANCHOR = "    for name, (func, dist_name) in plugins.items():\n"
LOADER_BLOCK = """    from sglang_dcu_plugin import register as register_embedded_dcu

    plugins.setdefault(
        "dcu", (register_embedded_dcu, "sglang-dcu-embedded")
    )

"""

BOOTSTRAP_BEGIN = "# BEGIN SGLANG DCU EMBEDDED BOOTSTRAP"
BOOTSTRAP_END = "# END SGLANG DCU EMBEDDED BOOTSTRAP"
BOOTSTRAP_ANCHOR = "_redirect_third_party_caches()\n"
BOOTSTRAP_BLOCK = """

from sglang_dcu_plugin.compat.aiter import (
    bootstrap_aiter_compat as _bootstrap_dcu_aiter_compat,
)

_bootstrap_dcu_aiter_compat()
"""

PACKAGE_DATA_TABLE = '[tool.setuptools.package-data]\n'
PACKAGE_DATA_LINE = '"sglang_dcu_plugin" = ["jit_overlay/**/*"]\n'


class ReconstructionError(RuntimeError):
    """Raised when reconstruction cannot be completed without ambiguity."""


@dataclass(frozen=True)
class ReconstructionResult:
    output: str
    upstream_commit: str
    plugin_commit: str
    branch: str
    vendored_files: int
    manifest: str


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    capture: bool = True,
) -> str:
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            # Source validation is read-only, including Git's optional stat cache.
            # Required locks for clone/switch still work in the generated checkout.
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except FileNotFoundError as exc:
        raise ReconstructionError(
            f"required command is unavailable: {command[0]}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise ReconstructionError(
            f"command failed ({' '.join(command)}): {detail or exc.returncode}"
        ) from exc
    return (result.stdout or "").strip()


def _git(repo: Path, *args: str) -> str:
    return _run(("git", "-C", str(repo), *args))


def _validate_repo(repo: Path, label: str) -> str:
    if not repo.is_dir():
        raise ReconstructionError(f"{label} repository does not exist: {repo}")
    if _git(repo, "rev-parse", "--is-inside-work-tree") != "true":
        raise ReconstructionError(f"{label} is not a Git worktree: {repo}")
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        preview = "\n".join(status.splitlines()[:20])
        raise ReconstructionError(
            f"{label} repository must be clean for a reproducible result:\n{preview}"
        )
    return _git(repo, "rev-parse", "HEAD")


def _require_file(root: Path, relative: Path, label: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise ReconstructionError(f"missing {label}: {path}")
    return path


def _tracked_plugin_files(plugin: Path) -> list[Path]:
    output = _git(plugin, "ls-files", "-z", "--", str(PLUGIN_PACKAGE))
    relative_paths = [Path(value) for value in output.split("\0") if value]
    if not relative_paths:
        raise ReconstructionError(
            f"plugin has no tracked files under {plugin / PLUGIN_PACKAGE}"
        )
    required = {
        PLUGIN_PACKAGE / "__init__.py",
        PLUGIN_PACKAGE / "compat/aiter.py",
    }
    missing = required.difference(relative_paths)
    if missing:
        raise ReconstructionError(
            "plugin is missing required tracked files: "
            + ", ".join(str(path) for path in sorted(missing))
        )
    return relative_paths


def _copy_tracked_files(source: Path, destination: Path, files: Iterable[Path]) -> int:
    count = 0
    for relative in files:
        src = source / relative
        if not src.is_file():
            raise ReconstructionError(
                f"tracked plugin file is not a regular file: {src}"
            )
        dst = destination / "python" / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        count += 1
    return count


def _inject_once(
    path: Path,
    anchor: str,
    block: str,
    begin: str,
    end: str,
    *,
    after: bool = False,
    symbol: str,
) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imported_names = {alias.asname or alias.name
                      for node in ast.walk(tree)
                      if isinstance(node, (ast.Import, ast.ImportFrom))
                      for alias in node.names}
    if begin in source or end in source or symbol in imported_names:
        raise ReconstructionError(
            f"embedded integration already exists in pristine source: {path}"
        )
    if source.count(anchor) != 1:
        raise ReconstructionError(
            "expected exactly one injection anchor in "
            f"{path}, found {source.count(anchor)}"
        )
    replacement = anchor + block if after else block + anchor
    updated = source.replace(anchor, replacement, 1)
    ast.parse(updated, filename=str(path))
    path.write_text(updated, encoding="utf-8")


def _add_package_data(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    if PACKAGE_DATA_LINE.strip() in source:
        raise ReconstructionError(f"DCU package-data entry already exists: {path}")
    if source.count(PACKAGE_DATA_TABLE) != 1:
        raise ReconstructionError(
            f"expected exactly one {PACKAGE_DATA_TABLE.strip()} table in {path}"
        )
    updated = source.replace(
        PACKAGE_DATA_TABLE,
        PACKAGE_DATA_TABLE + PACKAGE_DATA_LINE,
        1,
    )
    path.write_text(updated, encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_provenance(
    output: Path,
    *,
    upstream: Path,
    plugin: Path,
    upstream_commit: str,
    plugin_commit: str,
    branch: str,
    vendored_files: list[Path],
) -> Path:
    license_source = _require_file(plugin, Path("LICENSE"), "plugin license")
    license_target = output / "THIRD_PARTY_LICENSES/sglang-dcu-plugin.LICENSE"
    license_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(license_source, license_target)

    requirements_source = plugin / "requirements-dcu.txt"
    if requirements_source.is_file():
        shutil.copy2(requirements_source, output / "python/requirements-dcu.txt")

    readme = output / "DCU_EMBEDDED.md"
    readme.write_text(
        """# Embedded SGLang DCU Adaptation

This repository was reconstructed from upstream SGLang and
`sglang-plugin-das`. The plugin runtime is vendored under
`python/sglang_dcu_plugin` and registered by SGLang's own plugin loader.

No separately installed `sglang-dcu-plugin` distribution and no
`SGLANG_PLUGINS=dcu` setting are required. DCU vendor dependencies remain
required; see `python/requirements-dcu.txt`.

The exact source revisions and vendored file hashes are recorded in
`.sglang-dcu-embedded.json`.
""",
        encoding="utf-8",
    )

    tracked_outputs = [output / "python" / path for path in vendored_files]
    tracked_outputs.extend(
        [
            output / UPSTREAM_PLUGIN_LOADER,
            output / UPSTREAM_PACKAGE_INIT,
            output / UPSTREAM_PYPROJECT,
            license_target,
            readme,
        ]
    )
    if requirements_source.is_file():
        tracked_outputs.append(output / "python/requirements-dcu.txt")

    hashes = {
        str(path.relative_to(output)): _sha256(path)
        for path in sorted(tracked_outputs)
    }
    manifest = {
        "kind": MANIFEST_KIND,
        "generator_version": GENERATOR_VERSION,
        "mode": "source-vendored-hooks",
        "upstream": {"path": str(upstream), "commit": upstream_commit},
        "plugin": {"path": str(plugin), "commit": plugin_commit},
        "branch": branch,
        "vendored_file_count": len(vendored_files),
        "files": hashes,
    }
    manifest_path = output / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _is_generated_output(path: Path) -> bool:
    manifest = path / MANIFEST_NAME
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return data.get("kind") == MANIFEST_KIND


def reconstruct_sglang_das(
    upstream: str | os.PathLike[str],
    plugin: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    revision: str = "HEAD",
    branch: str = "embedded-dcu",
    force: bool = False,
) -> ReconstructionResult:
    """Create a standalone SGLang source repository with the DCU plugin embedded.

    Both source worktrees must be clean. The destination is built in a sibling
    temporary directory and renamed only after validation succeeds. ``force``
    can replace only a destination carrying this generator's manifest.
    """

    upstream_path = Path(upstream).expanduser().resolve()
    plugin_path = Path(plugin).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    for source in (upstream_path, plugin_path):
        if output_path == source or source in output_path.parents or output_path in source.parents:
            raise ReconstructionError("output must not overlap either source repository")

    upstream_head = _validate_repo(upstream_path, "upstream")
    plugin_commit = _validate_repo(plugin_path, "plugin")
    _require_file(upstream_path, UPSTREAM_PLUGIN_LOADER, "SGLang plugin loader")
    _require_file(upstream_path, UPSTREAM_PACKAGE_INIT, "SGLang package initializer")
    _require_file(upstream_path, UPSTREAM_PYPROJECT, "SGLang pyproject")
    vendored_files = _tracked_plugin_files(plugin_path)

    try:
        resolved_revision = _git(upstream_path, "rev-parse", f"{revision}^{{commit}}")
    except ReconstructionError as exc:
        raise ReconstructionError(f"invalid upstream revision {revision!r}") from exc

    if resolved_revision != upstream_head and revision == "HEAD":
        raise ReconstructionError("upstream HEAD changed while reconstruction was starting")

    if output_path.exists():
        if not force:
            raise ReconstructionError(f"output already exists: {output_path}")
        if not _is_generated_output(output_path):
            raise ReconstructionError(
                f"refusing to replace output without a valid {MANIFEST_NAME}: {output_path}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent)
    )
    shutil.rmtree(temp_path)
    try:
        _run(
            (
                "git",
                "clone",
                "--no-hardlinks",
                "--no-checkout",
                str(upstream_path),
                str(temp_path),
            )
        )
        _git(temp_path, "switch", "--create", branch, resolved_revision)

        copied = _copy_tracked_files(plugin_path, temp_path, vendored_files)
        _inject_once(
            temp_path / UPSTREAM_PLUGIN_LOADER,
            LOADER_ANCHOR,
            LOADER_BLOCK,
            LOADER_BEGIN,
            LOADER_END,
            symbol="register_embedded_dcu",
        )
        _inject_once(
            temp_path / UPSTREAM_PACKAGE_INIT,
            BOOTSTRAP_ANCHOR,
            BOOTSTRAP_BLOCK,
            BOOTSTRAP_BEGIN,
            BOOTSTRAP_END,
            symbol="_bootstrap_dcu_aiter_compat",
            after=True,
        )
        _add_package_data(temp_path / UPSTREAM_PYPROJECT)
        manifest_path = _write_provenance(
            temp_path,
            upstream=upstream_path,
            plugin=plugin_path,
            upstream_commit=resolved_revision,
            plugin_commit=plugin_commit,
            branch=branch,
            vendored_files=vendored_files,
        )

        compile_targets = [
            temp_path / UPSTREAM_PLUGIN_LOADER,
            temp_path / UPSTREAM_PACKAGE_INIT,
            *(
                temp_path / "python" / path
                for path in vendored_files
                if path.suffix == ".py"
            ),
        ]
        for target in compile_targets:
            ast.parse(target.read_text(encoding="utf-8"), filename=str(target))

        if output_path.exists():
            shutil.rmtree(output_path)
        temp_path.rename(output_path)
        return ReconstructionResult(
            output=str(output_path),
            upstream_commit=resolved_revision,
            plugin_commit=plugin_commit,
            branch=branch,
            vendored_files=copied,
            manifest=str(output_path / manifest_path.name),
        )
    except Exception:
        shutil.rmtree(temp_path, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct an embedded SGLang DAS source repository."
    )
    parser.add_argument("--upstream", required=True, help="clean upstream SGLang repo")
    parser.add_argument("--plugin", required=True, help="clean sglang-plugin-das repo")
    parser.add_argument("--output", required=True, help="new reconstructed repository")
    parser.add_argument("--revision", default="HEAD", help="upstream commit-ish")
    parser.add_argument("--branch", default="embedded-dcu", help="branch in output repo")
    parser.add_argument(
        "--force",
        action="store_true",
        help=f"replace output only when it has a valid {MANIFEST_NAME}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = reconstruct_sglang_das(
            args.upstream,
            args.plugin,
            args.output,
            revision=args.revision,
            branch=args.branch,
            force=args.force,
        )
    except ReconstructionError as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
