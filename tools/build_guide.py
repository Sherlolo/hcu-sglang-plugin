#!/usr/bin/env python3
"""Build a standalone, offline guide from the repository's shared text files."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlencode, urlsplit

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("docs", "skills", "tools", "templates", "tests", "ci", ".workflow", "licenses")
EXTENSIONS = {".md", ".py", ".json", ".yaml", ".yml", ".LICENSE"}


def source_files(root: Path) -> list[Path]:
    files = [root / name for name in ("README.md", "AGENTS.md", "LICENSE", ".gitignore")
             if (root / name).is_file() and not (root / name).is_symlink()]
    for folder in FOLDERS:
        for path in (root / folder).rglob("*"):
            if (path.is_file() and not path.is_symlink() and path.suffix in EXTENSIONS
                    and not any(part in {"__pycache__", ".venv", ".pytest_cache"}
                                for part in path.relative_to(root).parts)):
                files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def route(path: str, section: str = "") -> str:
    return "#" + urlencode({"read": path, **({"section": section} if section else {})})


def slug(text: str) -> str:
    text = re.sub(r"!?\[([^]]+)\]\([^)]*\)", r"\1", text)
    return re.sub(r"[^\w\- ]", "", text.lower()).replace(" ", "-")


def resolve_link(target: str, source: Path, root: Path, known: set[str]) -> str | None:
    target = html.unescape(target)
    parsed = urlsplit(target)
    if parsed.scheme in {"http", "https", "mailto"}:
        return target
    if parsed.scheme or parsed.netloc or target.startswith("/"):
        return None
    if not parsed.path:
        return route(source.relative_to(root).as_posix(), unquote(parsed.fragment))
    destination = (source.parent / unquote(parsed.path)).resolve()
    try:
        name = destination.relative_to(root.resolve()).as_posix()
    except ValueError:
        return None
    if name == "architecture.html":
        return "#start"
    if name in known:
        return route(name, unquote(parsed.fragment))
    return None


def inline(text: str, source: Path, root: Path, known: set[str]) -> str:
    saved: list[str] = []

    def save(value: str) -> str:
        saved.append(value)
        return "\x00{}\x00".format(len(saved) - 1)

    text = re.sub(r"(`+)(.+?)\1", lambda m: save("<code>" + html.escape(m[2]) + "</code>"), text)
    text = html.escape(text)

    def link(match: re.Match[str]) -> str:
        target = resolve_link(match[2], source, root, known)
        label = match[1]
        if target is None:
            return label
        return save('<a href="{}">{}</a>'.format(html.escape(target, quote=True), label))

    text = re.sub(r"!?\[([^\]]+)\]\(([^\s)]+)\)", link, text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    # A link label may contain a saved inline-code token.
    for _ in range(3):
        text = re.sub(r"\x00(\d+)\x00", lambda m: saved[int(m[1])], text)
    return text


def table_cells(line: str) -> list[str]:
    cells: list[str] = []
    current: list[str] = []
    code = False
    escaped = False
    for char in line.strip().strip("|"):
        if char == "`" and not escaped:
            code = not code
        if char == "|" and not code and not escaped:
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        escaped = char == "\\" and not escaped
    cells.append("".join(current).strip())
    return cells


def markdown(text: str, source: Path, root: Path, known: set[str]) -> tuple[str, list[dict]]:
    text = re.sub(r"\A---\n.*?\n---\n", "", text, count=1, flags=re.S)
    lines = text.splitlines()
    headings: list[dict] = []
    seen: set[str] = set()
    output: list[str] = []
    idx = 0
    render = lambda value: inline(value, source, root, known)
    list_pattern = re.compile(r"^(\s*)([-+*]|\d+[.)])\s+(.+)$")
    while idx < len(lines):
        line = lines[idx]
        if not line.strip():
            idx += 1
            continue
        fence = re.match(r"^\s*(`{3,}|~{3,})(.*)$", line)
        if fence:
            marker, language = fence[1], fence[2].strip()
            idx += 1
            content = []
            while idx < len(lines) and not re.fullmatch(r"\s*" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}\s*", lines[idx]):
                content.append(lines[idx])
                idx += 1
            escaped = html.escape("\n".join(content))
            code = '<pre><code>{}</code></pre>'.format(escaped)
            if language == "mermaid":
                code = '<details class="diagram-source"><summary>流程图源码（Mermaid）</summary>' + code + '</details>'
            output.append(code)
            idx += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            name = slug(heading[2])
            candidate, number = name, 0
            while candidate in seen:
                number += 1
                candidate = name + "-" + str(number)
            seen.add(candidate)
            level = len(heading[1])
            headings.append({"id": candidate, "text": re.sub(r"[`*]", "", heading[2]), "level": level})
            output.append('<h{0} id="{1}" tabindex="-1">{2}</h{0}>'.format(level, html.escape("article-" + candidate, quote=True), render(heading[2])))
            idx += 1
            continue
        if idx + 1 < len(lines) and "|" in line and re.fullmatch(r"[\s|:\-]+", lines[idx + 1]) and "---" in lines[idx + 1]:
            headers = table_cells(line)
            output.append('<div class="table-scroll" tabindex="0"><table><thead><tr>' + ''.join('<th scope="col">' + render(c) + '</th>' for c in headers) + '</tr></thead><tbody>')
            idx += 2
            while idx < len(lines) and "|" in lines[idx] and lines[idx].strip():
                output.append('<tr>' + ''.join('<td>' + render(c) + '</td>' for c in table_cells(lines[idx])) + '</tr>')
                idx += 1
            output.append('</tbody></table></div>')
            continue
        if line.startswith(">"):
            quoted = []
            while idx < len(lines) and lines[idx].startswith(">"):
                quoted.append(lines[idx].lstrip("> "))
                idx += 1
            output.append('<blockquote>' + render(" ".join(quoted)) + '</blockquote>')
            continue
        item = list_pattern.match(line)
        if item:
            stack: list[tuple[int, str]] = []
            while idx < len(lines):
                match = list_pattern.match(lines[idx])
                if not match:
                    break
                depth = len(match[1].expandtabs(4))
                tag = "ol" if match[2][0].isdigit() else "ul"
                while stack and (depth < stack[-1][0] or (depth == stack[-1][0] and tag != stack[-1][1])):
                    output.append('</li></' + stack.pop()[1] + '>')
                if not stack or depth > stack[-1][0]:
                    output.append('<' + tag + '>')
                    stack.append((depth, tag))
                else:
                    output.append('</li>')
                output.append('<li>' + render(match[3]))
                idx += 1
            while stack:
                output.append('</li></' + stack.pop()[1] + '>')
            continue
        if re.fullmatch(r"\s*([-*_])\1{2,}\s*", line):
            output.append('<hr>')
            idx += 1
            continue
        paragraph = [line]
        idx += 1
        while idx < len(lines) and lines[idx].strip():
            if re.match(r"^(#{1,6}\s|\s*```|\s*~~~|>)", lines[idx]) or list_pattern.match(lines[idx]):
                break
            if idx + 1 < len(lines) and "|" in lines[idx] and re.fullmatch(r"[\s|:\-]+", lines[idx + 1]):
                break
            paragraph.append(lines[idx])
            idx += 1
        output.append('<p>' + render(" ".join(paragraph)) + '</p>')
    return "\n".join(output), headings


def category(name: str) -> str:
    if name.endswith("/SKILL.md"):
        return "Skill"
    if name.startswith("skills/") and "/references/" in name:
        return "参考方法"
    if name.startswith("docs/") or name == "README.md":
        return "文档"
    if name.startswith("tools/"):
        return "工具"
    return "工程资源"


def build(root: Path) -> str:
    root = root.resolve()
    files = source_files(root)
    contents = {p.relative_to(root).as_posix(): p.read_bytes().decode("utf-8") for p in files}
    folders = {parent.as_posix() for name in contents for parent in Path(name).parents if parent.as_posix() != "."}
    known = set(contents) | folders
    catalog = []
    for name, text in contents.items():
        match = re.search(r"^#\s+(.+)$", text, re.M) if name.endswith(".md") else None
        title = match[1] if match else Path(name).name
        if name.endswith(".md"):
            body, headings = markdown(text, root / name, root, known)
        else:
            body = '<h1>{}</h1><pre><code>{}</code></pre>'.format(html.escape(title), html.escape(text))
            headings = []
        catalog.append({"path": name, "title": title, "kind": category(name), "text": text, "html": body, "headings": headings})
    for name in sorted(folders):
        children = sorted(n for n in known if n != name and Path(n).parent.as_posix() == name)
        body = '<h1>{}</h1><ul>{}</ul>'.format(html.escape(name + "/"), ''.join('<li><a href="{}">{}</a></li>'.format(html.escape(route(child), quote=True), html.escape(Path(child).name + ("/" if child in folders else ""))) for child in children))
        catalog.append({"path": name, "title": name + "/", "kind": "目录", "text": "", "html": body, "headings": []})
    template = (root / "templates/architecture.html").read_text(encoding="utf-8")
    digest = hashlib.sha256((template + json.dumps(contents, ensure_ascii=False, sort_keys=True)).encode()).hexdigest()[:12]
    payload = json.dumps(catalog, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace("&", "\\u0026")
    replacements = {"__CATALOG_JSON__": payload, "__SNAPSHOT__": digest,
                    "__SKILL_COUNT__": str(sum(name.endswith("/SKILL.md") for name in contents)),
                    "__TOOL_COUNT__": str(sum(name.startswith("tools/") and name.endswith(".py") for name in contents))}
    return re.sub("|".join(re.escape(token) for token in replacements),
                  lambda match: replacements[match[0]], template)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="method repository root")
    parser.add_argument("--output", type=Path, help="default: <root>/architecture.html")
    parser.add_argument("--check", action="store_true", help="check freshness without writing")
    args = parser.parse_args()
    output = args.output or args.root / "architecture.html"
    rendered = build(args.root)
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != rendered:
            print("Guide is stale: run python3 tools/build_guide.py")
            return 1
        print("Guide is up to date")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print("Built offline guide: {}".format(output.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
