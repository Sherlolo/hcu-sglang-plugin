from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.plugin_layout import resolve_plugin_package
from tools.build_migration_manifest import plugin_env_names, plugin_targets
from tools.build_disposition_ledger import plugin_targets as ledger_targets


class PluginLayoutTests(unittest.TestCase):
    def test_detects_current_and_legacy_and_refuses_ambiguous_or_missing_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'expected one'):
                resolve_plugin_package(root)
            for name in ('hcu_sglang', 'sglang_dcu_plugin'):
                (root / name).mkdir()
                (root / name / '__init__.py').write_text('')
                self.assertEqual(resolve_plugin_package(root, name), name)
            with self.assertRaisesRegex(ValueError, 'select --plugin-package'):
                resolve_plugin_package(root)
            with self.assertRaises(ValueError):
                resolve_plugin_package(root, '../outside')

    def test_manifest_and_ledger_discover_hcu_targets_and_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / 'hcu_sglang'
            package.mkdir()
            (package / '__init__.py').write_text('')
            (package / 'environ.py').write_text('ENV = "SGLANG_HCU_TEST"\n')
            (package / 'hooks.py').write_text('HookRegistry.register("sglang.srt.test.forward", hook, kind)\n')
            self.assertEqual(plugin_env_names(root), {'SGLANG_HCU_TEST'})
            self.assertEqual(plugin_targets(root), ['sglang.srt.test.forward'])
            self.assertEqual(ledger_targets(root), ['sglang.srt.test.forward'])
