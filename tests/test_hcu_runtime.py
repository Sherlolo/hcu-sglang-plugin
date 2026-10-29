from __future__ import annotations

import ast
import json
import os
import tempfile
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

    def test_launcher_preserves_spawn_bootstrap_entry_and_cli_arguments(self):
        from hcu_sglang.__main__ import main
        args = ['hcu-sglang-serve', '--model-path', 'test-model', '--tp-size', '8']
        calls = []
        with patch('runpy.run_module', side_effect=lambda *a, **kw: calls.append((a, kw, sys.argv[1:]))), \
             patch.object(sys, 'argv', args):
            main()
        self.assertEqual(calls, [(('hcu_sglang.launch_server',),
                                  {'run_name': '__main__', 'alter_sys': True}, args[1:])])

    def test_spawn_worker_imports_optional_aiter_api_after_compat_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for package in ['aiter', 'aiter/ops', 'aiter/ops/triton', 'sglang']:
                path = root / package
                path.mkdir(parents=True, exist_ok=True)
                (path / '__init__.py').write_text('')
            (root / 'worker_fixture.py').write_text(
                "from aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_split_cat import fused_gemm_afp4wfp4_split_cat\n"
                "def worker(queue):\n"
                "    queue.put(fused_gemm_afp4wfp4_split_cat.__name__)\n")
            (root / 'sglang/launch_server.py').write_text(
                "import multiprocessing as mp\n"
                "from worker_fixture import worker\n"
                "if __name__ == '__main__':\n"
                "    ctx = mp.get_context('spawn')\n"
                "    queue = ctx.Queue()\n"
                "    process = ctx.Process(target=worker, args=(queue,))\n"
                "    process.start()\n"
                "    process.join(15)\n"
                "    assert process.exitcode == 0, process.exitcode\n"
                "    assert queue.get(timeout=5) == '_unsupported_fused_gemm'\n"
                "    print('spawn bootstrap passed')\n")
            env = {**os.environ, 'PYTHONPATH': str(root) + os.pathsep + str(ROOT)}
            result = subprocess.run([sys.executable, '-m', 'hcu_sglang'], cwd=root,
                                    env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('spawn bootstrap passed', result.stdout)

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
