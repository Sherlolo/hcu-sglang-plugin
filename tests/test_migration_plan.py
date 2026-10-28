from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.build_migration_plan import CONTRACT, collect, inventory, render, target_path, validate_plan

ROOT = Path(__file__).resolve().parents[1]


class MigrationPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repos = {name: self.root / name for name in ['official', 'fork', 'plugin']}
        for path in self.repos.values():
            path.mkdir()
            self.git(path, 'init', '-q')
            self.git(path, 'config', 'user.email', 'test@example.com')
            self.git(path, 'config', 'user.name', 'Test')
        module = 'python/sglang/srt/example.py'
        self.write('official', module, 'class Base:\n    pass\n\ndef run(x):\n    return x\n')
        self.write('fork', module, 'class Base:\n    pass\n\ndef run(x):\n    return x + 1\n')
        for name in ['official', 'fork']:
            self.write(name, 'README.md', '# shared\n')
            self.write(name, 'python/sglang/srt/layers/quantization/__init__.py', 'def get_quantization_config(name):\n    return name\n')
        self.write('fork', 'python/sglang/srt/only_fork.py', 'async def added():\n    return 1\n')
        self.write('official', 'python/sglang/srt/only_official.py', 'VALUE = 1\n')
        target = 'sglang.srt.layers.quantization.get_quantization_config'
        self.write('plugin', CONTRACT, json.dumps([target]))
        self.write('plugin', 'sglang_dcu_plugin/hooks.py', 'TARGET = ' + repr(target) + '\n')
        for path in self.repos.values():
            self.commit(path)
        self.plan = {'title': 'test plan', 'reviewed_on': '2026-10-28',
                     'phases': [{'id': 'M0', 'title': 'scope', 'gate': 'review', 'depends_on': []}],
                     'groups': [{'id': 'all', 'title': 'all', 'prefixes': [''], 'role': 'team',
                                 'phase': 'M0', 'priority': 'P0', 'decision': 'review',
                                 'tasks': ['assess'], 'acceptance': ['evidence'],
                                 'depends_on': [], 'destinations': []}], 'historical_evidence': []}

    def git(self, root, *args):
        return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.DEVNULL,
                                       env=dict(os.environ, GIT_OPTIONAL_LOCKS='0')).decode().strip()

    def write(self, repo, name, text):
        path = self.repos[repo] / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')

    def commit(self, repo):
        self.git(repo, 'add', '.')
        self.git(repo, 'commit', '-qm', 'fixture')

    def snapshot(self):
        return inventory(*(collect(self.repos[name]) for name in ['official', 'fork', 'plugin']), self.plan)

    def test_tree_relations_symbols_and_package_targets_are_not_runtime_completion(self):
        report = self.snapshot()
        rows = {row['path']: row for row in report['files']}
        self.assertEqual(rows['README.md']['relation'], 'same')
        self.assertEqual(rows['python/sglang/srt/example.py']['relation'], 'changed')
        self.assertEqual(rows['python/sglang/srt/only_fork.py']['relation'], 'fork_only')
        self.assertEqual(rows['python/sglang/srt/only_official.py']['relation'], 'official_only')
        self.assertEqual(rows['python/sglang/srt/only_fork.py']['symbols'], [['added', 1, 'async function']])
        package = rows['python/sglang/srt/layers/quantization/__init__.py']
        self.assertEqual(len(package['targets']), 1)
        self.assertEqual(package['targets'][0]['literal_sites'][0][0], 'sglang_dcu_plugin/hooks.py')
        self.assertEqual(report['summary']['unmatched_targets'], 0)
        self.assertEqual(report['groups'][0]['tasks'][0]['status'], 'proposed')
        self.assertIsNone(report['groups'][0]['tasks'][0]['assignee'])
        self.assertIsNone(target_path('sglang.missing.submodule.run', set(rows)))

    def test_dirty_inputs_are_recorded_head_content_is_read_and_indexes_are_preserved(self):
        self.write('fork', 'python/sglang/srt/only_fork.py', 'def dirty_change():\n    pass\n')
        self.write('fork', 'untracked.txt', 'preserve me')
        before = {name: (root / '.git/index').read_bytes() for name, root in self.repos.items()}
        report = self.snapshot()
        self.assertTrue(report['baselines']['fork']['dirty'])
        row = next(row for row in report['files'] if row['path'].endswith('only_fork.py'))
        self.assertEqual(row['symbols'][0][0], 'added')
        self.assertTrue((self.repos['fork'] / 'untracked.txt').exists())
        for name, root in self.repos.items():
            self.assertEqual((root / '.git/index').read_bytes(), before[name])

    def test_provenance_mismatch_and_identical_native_bytes_remain_static_clues(self):
        native = 'int native_kernel() { return 1; }\n' * 4
        self.write('fork', 'python/sglang/kernels/example.cu', native)
        self.write('plugin', 'sglang_dcu_plugin/kernels/example.cu', native)
        self.write('plugin', 'sglang_dcu_plugin/source-provenance.json', json.dumps({'source': 'python/sglang/srt/example.py', 'source_sha256': '0' * 64}))
        self.commit(self.repos['fork']); self.commit(self.repos['plugin'])
        report = self.snapshot();rows = {row['path']: row for row in report['files']}
        self.assertEqual(rows['python/sglang/srt/example.py']['provenance'][0]['status'], 'source_hash_mismatch')
        self.assertEqual(rows['python/sglang/kernels/example.cu']['identical_native'], ['sglang_dcu_plugin/kernels/example.cu'])
        self.assertEqual(report['groups'][0]['tasks'][0]['status'], 'proposed')

    def test_gitlinks_and_symlinks_do_not_read_external_source(self):
        outside = self.root / 'private.py';outside.write_text('PRIVATE_DATA = True\n')
        path = self.repos['fork'] / 'external.py';path.symlink_to(outside)
        self.git(self.repos['fork'], 'add', 'external.py')
        head = self.git(self.repos['fork'], 'rev-parse', 'HEAD')
        self.git(self.repos['fork'], 'update-index', '--add', '--cacheinfo', '160000,' + head + ',vendor/submodule')
        self.git(self.repos['fork'], 'commit', '-qm', 'links')
        report = self.snapshot();rows = {row['path']: row for row in report['files']}
        self.assertEqual(rows['external.py']['mode'], '120000')
        self.assertNotIn('symbols', rows['external.py'])
        self.assertEqual(rows['vendor/submodule']['kind'], 'commit')
        self.assertNotIn('PRIVATE_DATA', json.dumps(report))

    def test_dependencies_and_empty_target_contract_fail_explicitly(self):
        invalid = copy.deepcopy(self.plan);invalid['groups'][0]['depends_on'] = ['missing']
        with self.assertRaises(ValueError):validate_plan(invalid)
        invalid = copy.deepcopy(self.plan);invalid['phases'][0]['depends_on'] = ['M0']
        with self.assertRaises(ValueError):validate_plan(invalid)
        self.write('plugin', CONTRACT, '[]');self.commit(self.repos['plugin'])
        with self.assertRaises(ValueError):self.snapshot()

    def test_payload_is_safe_and_cli_check_is_read_only(self):
        snapshot = self.snapshot();snapshot['title'] = '</script><script>unexpected()</script>'
        template = '<script id="migration-data" type="application/json">__MIGRATION_DATA__</script>'
        html = render(snapshot, template)
        self.assertEqual(html.count('</script>'), 1)
        self.assertNotIn('<script>unexpected()', html)
        plan = self.root / 'plan.json';plan.write_text(json.dumps(self.plan))
        source = self.root / 'snapshot.json';source.write_text(json.dumps(snapshot))
        layout = self.root / 'template.html';layout.write_text(template)
        output = self.root / 'output.html'
        args = [sys.executable, str(ROOT / 'tools/build_migration_plan.py'), '--plan', str(plan), '--snapshot', str(source), '--template', str(layout), '--output', str(output)]
        self.assertNotEqual(subprocess.run(args + ['--check'], capture_output=True).returncode, 0)
        self.assertFalse(output.exists())
        self.assertEqual(subprocess.run(args + ['--render-only'], capture_output=True).returncode, 0)
        previous = output.read_bytes()
        self.assertEqual(subprocess.run(args + ['--check'], capture_output=True).returncode, 0)
        self.plan['groups'][0]['tasks'].append('new scope');plan.write_text(json.dumps(self.plan))
        self.assertNotEqual(subprocess.run(args + ['--check'], capture_output=True).returncode, 0)
        self.assertEqual(output.read_bytes(), previous)


if __name__ == '__main__':
    unittest.main()
