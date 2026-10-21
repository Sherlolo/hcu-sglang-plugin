#!/usr/bin/env python3
"""Offline structural gate: no third-party dependencies or runtime imports."""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    errors = []
    files = [p for p in ROOT.rglob("*") if p.is_file()
             and not any(part in {".git", "artifacts", "__pycache__", ".venv"}
                         for part in p.relative_to(ROOT).parts)]
    python_files = [p for p in files if p.suffix == ".py"]
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    skills = sorted((ROOT / "skills").glob("*/SKILL.md"))
    for path in skills:
        text = path.read_text(encoding="utf-8")
        match = re.match(r"\A---\n(.*?)\n---\n", text, re.S)
        if not match:
            errors.append(f"{path}: missing frontmatter")
            continue
        fields = dict(re.findall(r"^(name|description):\s*(.+)$", match[1], re.M))
        name = fields.get("name", "")
        if name != path.parent.name or not re.fullmatch(r"[a-z0-9-]{1,63}", name):
            errors.append(f"{path}: invalid or mismatched name")
        if not fields.get("description"):
            errors.append(f"{path}: missing description")
        meta = path.parent / "agents/openai.yaml"
        if not meta.is_file() or "$" + name not in meta.read_text(encoding="utf-8"):
            errors.append(f"{path}: UI metadata/default prompt missing")
    for path in files:
        if path.suffix != ".md":
            continue
        for target in re.findall(r"\[[^\]]*\]\(([^)\s]+)\)", path.read_text(encoding="utf-8")):
            if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
                continue
            relative = unquote(target.split("#", 1)[0])
            if relative and not (path.parent / relative).exists():
                errors.append(f"{path.relative_to(ROOT)}: missing link {target}")
    for forbidden in ("sglang_dcu_plugin", "sgl_kernel", "sitecustomize.py", "usercustomize.py", "pyproject.toml"):
        if (ROOT / forbidden).exists():
            errors.append(f"runtime/build artifact outside current scope: {forbidden}")
    report = {"status": "failed" if errors else "passed", "python_files": len(python_files),
              "skills": len(skills), "errors": errors}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return int(bool(errors))


if __name__ == "__main__":
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10+ required")
    raise SystemExit(main())
