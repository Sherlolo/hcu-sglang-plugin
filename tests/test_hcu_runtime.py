from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class HcuRuntimeTests(unittest.TestCase):
    def test_cold_import_keeps_startup_lightweight_and_uses_current_namespace(self):
        code = "import hcu_sglang,sys,json; print(json.dumps({'file':hcu_sglang.__file__,'loaded':[n for n in sys.modules if n.split('.')[0] in ['torch','aiter','sglang_dcu_plugin']]}))"
        report = json.loads(subprocess.check_output([sys.executable, '-c', code], cwd=ROOT, text=True))
        self.assertTrue(Path(report['file']).is_relative_to(ROOT / 'hcu_sglang'))
        self.assertEqual(report['loaded'], [])

    def test_launcher_bootstraps_before_forwarding_arguments_to_official_server(self):
        from hcu_sglang.__main__ import main
        events = []
        args = ['hcu-sglang-serve', '--model-path', 'test-model', '--tp-size', '4']
        with patch('hcu_sglang.compat.aiter.bootstrap_aiter_compat', side_effect=lambda: events.append('bootstrap')), \
             patch('runpy.run_module', side_effect=lambda *a, **kw: events.append((a, kw, sys.argv[1:]))), \
             patch.object(sys, 'argv', args):
            main()
        self.assertEqual(events[0], 'bootstrap')
        self.assertEqual(events[1], (('sglang.launch_server',), {'run_name': '__main__', 'alter_sys': True}, args[1:]))

    def test_runtime_does_not_import_legacy_namespace(self):
        for path in (ROOT / 'hcu_sglang').rglob('*.py'):
            tree = ast.parse(path.read_bytes())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or '').startswith('sglang_dcu_plugin'), str(path))
                if isinstance(node, ast.Import):
                    self.assertFalse(any(alias.name.startswith('sglang_dcu_plugin') for alias in node.names), str(path))

    def test_scoped_launcher_replaces_global_usercustomize(self):
        self.assertFalse((ROOT / 'usercustomize.py').exists())
        self.assertFalse((ROOT / 'sitecustomize.py').exists())
        self.assertTrue((ROOT / 'hcu_sglang/__main__.py').is_file())
        self.assertTrue((ROOT / 'hcu_sglang/jit_overlay/csrc/deep_select/vendor/LICENSE').is_file())
        self.assertTrue(any((ROOT / 'hcu_sglang').rglob('*.hpp')))


if __name__ == '__main__':
    unittest.main()
