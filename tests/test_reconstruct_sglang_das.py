from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.reconstruct_sglang_das import (
    BOOTSTRAP_BEGIN,
    LOADER_BEGIN,
    MANIFEST_KIND,
    MANIFEST_NAME,
    ReconstructionError,
    reconstruct_sglang_das,
)


def _run(*args: str, cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _init_repo(path: Path) -> None:
    _run("git", "init", "-q", cwd=path)
    _run("git", "config", "user.email", "test@example.com", cwd=path)
    _run("git", "config", "user.name", "Test", cwd=path)
    _run("git", "add", ".", cwd=path)
    _run("git", "commit", "-qm", "fixture", cwd=path)


class ReconstructionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.upstream = self.root / "upstream"
        self.plugin = self.root / "plugin"
        self.output = self.root / "output"
        self.upstream.mkdir()
        self.plugin.mkdir()

        _write(
            self.upstream / "python/sglang/srt/plugins/__init__.py",
            """plugins = {}\n\ndef load_plugins():\n    plugins = {}\n    for name, (func, dist_name) in plugins.items():\n        func()\n""",
        )
        _write(
            self.upstream / "python/sglang/__init__.py",
            '"""SGLang public API."""\n\n'
            "def _redirect_third_party_caches():\n"
            "    pass\n\n"
            "_redirect_third_party_caches()\n\n"
            "VALUE = 1\n",
        )
        _write(
            self.upstream / "python/pyproject.toml",
            '[tool.setuptools.package-data]\n"sglang" = ["srt/**/*"]\n',
        )
        _write(self.upstream / "README.md", "upstream\n")
        _init_repo(self.upstream)

        _write(
            self.plugin / "sglang_dcu_plugin/__init__.py",
            "def register():\n    return None\n",
        )
        _write(
            self.plugin / "sglang_dcu_plugin/compat/aiter.py",
            "def bootstrap_aiter_compat():\n    return False\n",
        )
        _write(self.plugin / "sglang_dcu_plugin/jit_overlay/include/example.h", "x\n")
        _write(self.plugin / "ignored.txt", "not vendored\n")
        _write(self.plugin / "LICENSE", "license\n")
        _write(self.plugin / "requirements-dcu.txt", "vendor-package\n")
        _init_repo(self.plugin)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_reconstructs_repository_with_provenance(self) -> None:
        result = reconstruct_sglang_das(self.upstream, self.plugin, self.output)

        self.assertEqual(result.vendored_files, 3)
        self.assertTrue((self.output / ".git").is_dir())
        self.assertTrue((self.output / "python/sglang_dcu_plugin/__init__.py").is_file())
        self.assertFalse((self.output / "python/ignored.txt").exists())
        loader = (self.output / "python/sglang/srt/plugins/__init__.py").read_text()
        package_init = (self.output / "python/sglang/__init__.py").read_text()
        self.assertIn(LOADER_BEGIN, loader)
        self.assertIn(BOOTSTRAP_BEGIN, package_init)
        self.assertEqual(package_init.count('"""SGLang public API."""'), 1)
        self.assertTrue(package_init.startswith('"""SGLang public API."""'))
        self.assertLess(
            package_init.index("_redirect_third_party_caches()"),
            package_init.index(BOOTSTRAP_BEGIN),
        )
        self.assertIn(
            '"sglang_dcu_plugin"',
            (self.output / "python/pyproject.toml").read_text(),
        )

        manifest = json.loads((self.output / MANIFEST_NAME).read_text())
        self.assertEqual(manifest["kind"], MANIFEST_KIND)
        self.assertEqual(manifest["vendored_file_count"], 3)
        self.assertIn("python/sglang_dcu_plugin/__init__.py", manifest["files"])

    def test_existing_output_requires_force_and_valid_manifest(self) -> None:
        self.output.mkdir()
        with self.assertRaises(ReconstructionError):
            reconstruct_sglang_das(self.upstream, self.plugin, self.output)
        with self.assertRaises(ReconstructionError):
            reconstruct_sglang_das(
                self.upstream, self.plugin, self.output, force=True
            )

    def test_force_replaces_only_generated_output(self) -> None:
        reconstruct_sglang_das(self.upstream, self.plugin, self.output)
        marker = self.output / "stale"
        marker.write_text("stale", encoding="utf-8")

        reconstruct_sglang_das(
            self.upstream, self.plugin, self.output, force=True
        )
        self.assertFalse(marker.exists())

    def test_dirty_source_is_rejected(self) -> None:
        _write(self.plugin / "untracked.py", "x = 1\n")
        with self.assertRaisesRegex(ReconstructionError, "must be clean"):
            reconstruct_sglang_das(self.upstream, self.plugin, self.output)

    def test_loader_drift_is_rejected_atomically(self) -> None:
        loader = self.upstream / "python/sglang/srt/plugins/__init__.py"
        loader.write_text("def load_plugins():\n    pass\n", encoding="utf-8")
        _run("git", "add", ".", cwd=self.upstream)
        _run("git", "commit", "-qm", "drift", cwd=self.upstream)

        with self.assertRaisesRegex(ReconstructionError, "injection anchor"):
            reconstruct_sglang_das(self.upstream, self.plugin, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
