#!/usr/bin/env python3
"""Verify files listed in a source-embedded reconstruction manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MANIFEST_NAME = ".sglang-dcu-embedded.json"
MANIFEST_KIND = "sglang-dcu-embedded-source"


def verify(output: Path) -> dict:
    root = output.resolve()
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("kind") != MANIFEST_KIND:
        raise ValueError("unsupported manifest kind")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("manifest must contain a nonempty files mapping")
    errors = []
    for relative, expected in sorted(files.items()):
        path = root / relative
        if Path(relative).is_absolute() or root not in path.resolve().parents:
            errors.append({"path": relative, "error": "path outside output"})
            continue
        if not path.is_file():
            errors.append({"path": relative, "error": "missing file"})
            continue
        if not isinstance(expected, str) or len(expected) != 64:
            errors.append({"path": relative, "error": "invalid SHA256"})
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            errors.append({"path": relative, "error": "SHA256 mismatch", "actual": digest})
    return {"output": str(root), "checked": len(files), "errors": errors,
            "status": "failed" if errors else "passed"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="generated repository")
    args = parser.parse_args()
    try:
        result = verify(args.output)
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 2
    print(json.dumps(result, indent=2))
    return int(bool(result["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
