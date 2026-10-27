from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from tools.build_guide import build, markdown, resolve_link, source_files

ROOT = Path(__file__).resolve().parents[1]


class CatalogParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.capture = False
        self.data = []

    def handle_starttag(self, tag, attrs):
        if tag == "script" and dict(attrs).get("id") == "guide-catalog":
            self.capture = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.capture = False

    def handle_data(self, data):
        if self.capture:
            self.data.append(data)


class GuideTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "docs").mkdir()
        (self.root / "templates").mkdir()
        (self.root / "templates/architecture.html").write_text(
            '<!doctype html><script id="guide-catalog" type="application/json">'
            '__CATALOG_JSON__</script><p>__TOOL_COUNT__</p>', encoding="utf-8")
        (self.root / "README.md").write_text(
            '# 入口\n\n[操作](docs/guide.md#重复-1)\n\n[目录](docs)\n', encoding="utf-8")
        (self.root / "docs/guide.md").write_text(
            '# 操作\n\n## 重复\n\n正文\n\n## 重复\n\n[返回](../README.md)\n', encoding="utf-8")

    def catalog(self):
        parser = CatalogParser()
        parser.feed(build(self.root))
        return {item["path"]: item for item in json.loads("".join(parser.data))}

    def test_offline_catalog_links_reach_documents_directories_and_duplicate_heading(self):
        catalog = self.catalog()
        self.assertIn("docs", catalog)
        self.assertIn('article-重复-1', catalog['docs/guide.md']['html'])
        from html import unescape
        import re
        for item in catalog.values():
            for href in re.findall(r'href="([^\"]+)"', item['html']):
                params = parse_qs(urlsplit(unescape(href)).fragment)
                destination = catalog[params['read'][0]]
                if 'section' in params:
                    self.assertIn(params['section'][0], {h['id'] for h in destination['headings']})

    def test_embedded_source_preserves_literal_tokens_and_cannot_close_json_script(self):
        original = '# 内容\n\n__TOOL_COUNT__ </script><script>globalThis.injected=true</script>\n'
        (self.root / 'docs/raw.md').write_text(original, encoding='utf-8')
        catalog = self.catalog()
        self.assertEqual(catalog['docs/raw.md']['text'], original)
        self.assertNotIn('<script>', catalog['docs/raw.md']['html'])
        self.assertIn('&lt;script&gt;', catalog['docs/raw.md']['html'])
        self.assertEqual(build(self.root), build(self.root))

    def test_reader_keeps_code_pipes_in_one_table_cell(self):
        rendered, _ = markdown('| 数据 | 说明 |\n|---|---|\n| `a|b` | `x < y` |\n',
                               self.root / 'README.md', self.root, {'README.md'})
        self.assertEqual(rendered.count('<td>'), 2)
        self.assertIn('<code>a|b</code>', rendered)
        self.assertIn('<code>x &lt; y</code>', rendered)

    def test_links_and_source_enumeration_cannot_escape_repository(self):
        source = self.root / 'docs/guide.md'
        for target in ['javascript:alert(1)', 'data:text/html,test', '//example.com/x', '../../outside.md']:
            self.assertIsNone(resolve_link(target, source, self.root, {'README.md'}))
        external = tempfile.TemporaryDirectory()
        self.addCleanup(external.cleanup)
        outside = Path(external.name) / 'outside.md'
        outside.write_text('private external contents', encoding='utf-8')
        (self.root / 'docs/external.md').symlink_to(outside)
        (self.root / 'README.md').unlink()
        (self.root / 'README.md').symlink_to(outside)
        (self.root / '.env').write_text('PRIVATE_VALUE=secret', encoding='utf-8')
        selected = source_files(self.root)
        self.assertNotIn(self.root / 'docs/external.md', selected)
        self.assertNotIn(self.root / 'README.md', selected)
        self.assertNotIn(self.root / '.env', selected)

    def test_check_is_read_only_and_detects_stale_snapshot(self):
        command = [sys.executable, str(ROOT / 'tools/build_guide.py'), '--root', str(self.root)]
        result = subprocess.run(command + ['--check'], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        output = self.root / 'architecture.html'
        self.assertFalse(output.exists())
        self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
        original = output.read_bytes()
        self.assertEqual(subprocess.run(command + ['--check'], capture_output=True).returncode, 0)
        (self.root / 'docs/guide.md').write_text('# changed\n', encoding='utf-8')
        self.assertNotEqual(subprocess.run(command + ['--check'], capture_output=True).returncode, 0)
        self.assertEqual(output.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
