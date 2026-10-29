"""A loader's exit status alone must not make a partial/empty plugin pass."""
import logging
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.validate_plugin_runtime import validate


def fake_runtime(targets, patched, official, plugin, soft_error=False, package="sglang_dcu_plugin"):
    modules = {}
    for name in ["sglang", "sglang.srt", "sglang.srt.plugins",
                 "sglang.srt.plugins.hook_registry", "sglang.srt.arg_groups",
                 "sglang.srt.arg_groups.resolution_hooks", package]:
        module = types.ModuleType(name)
        module.__path__ = []
        modules[name] = module
    for name, module in modules.items():
        if "." in name:
            parent, attr = name.rsplit(".", 1)
            setattr(modules[parent], attr, module)
    modules["sglang"].__file__ = str(official / "sglang" / "__init__.py")
    modules[package].__file__ = str(plugin / package / "__init__.py")

    class Registry:
        _hooks = {}
        _patched = set()

        @classmethod
        def reset(cls):
            cls._hooks = {}
            cls._patched = set()

    def load():
        Registry._hooks = dict.fromkeys(targets, [])
        Registry._patched = set(patched)
        if soft_error:
            logging.getLogger("sglang.srt.plugins").error("Failed to execute general plugin: dcu")

    def wrapper(*args):
        pass
    wrapper.__module__ = package + ".test"
    modules["sglang.srt.plugins"].load_plugins = load
    modules["sglang.srt.plugins.hook_registry"].HookRegistry = Registry
    modules["sglang.srt.arg_groups.resolution_hooks"]._HOOKS = {
        name: [wrapper] for name in ["handle_a2a_moe", "handle_attention_backend_compatibility",
                                     "handle_model_specific_adjustments"]}
    return modules


class RuntimeValidationTests(unittest.TestCase):
    def check(self, targets, patched, soft_error=False):
        with tempfile.TemporaryDirectory(prefix="runtime-validator-") as directory:
            official = Path(directory) / "official"
            plugin = Path(directory) / "plugin"
            with patch.dict(sys.modules, fake_runtime(targets, patched, official, plugin, soft_error)):
                return validate(official, plugin)

    def test_current_package_namespace_is_checked_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            official, plugin = Path(directory) / "official", Path(directory) / "plugin"
            modules = fake_runtime(["sglang.target"], ["sglang.target"], official, plugin, package="hcu_sglang")
            with patch.dict(sys.modules, modules):
                report = validate(official, plugin, plugin_package="hcu_sglang")
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["plugin_package"], "hcu_sglang")

    def test_empty_loader_cannot_pass(self):
        self.assertEqual(self.check([], [])["status"], "failed")

    def test_unapplied_target_cannot_pass(self):
        result = self.check(["sglang.target"], [])
        self.assertEqual(result["unapplied"], ["sglang.target"])
        self.assertEqual(result["status"], "failed")

    def test_logged_registration_error_cannot_pass_even_if_partial_hooks_applied(self):
        self.assertEqual(self.check(["sglang.target"], ["sglang.target"], True)["status"], "failed")

    def test_complete_nonempty_loader_passes(self):
        self.assertEqual(self.check(["sglang.target"], ["sglang.target"])["status"], "passed")


if __name__ == "__main__":
    unittest.main()
