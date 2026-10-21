from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.reconstruct_sglang_das import ReconstructionError, reconstruct_sglang_das
from tools.verify_reconstruction import MANIFEST_NAME, verify
import test_reconstruct_sglang_das as reconstruction_fixture
from test_reconstruct_sglang_das import _init_repo, _write

ROOT = Path(__file__).resolve().parents[1]


class AdditionalReconstructionTests(unittest.TestCase):
    setUp = reconstruction_fixture.ReconstructionTest.setUp
    tearDown = reconstruction_fixture.ReconstructionTest.tearDown

    def test_hash_verification_detects_tamper_and_missing(self):
        reconstruct_sglang_das(self.upstream, self.plugin, self.output)
        self.assertEqual(verify(self.output)["status"], "passed")
        source = self.output / "python/sglang_dcu_plugin/__init__.py"
        source.write_text("tampered = True\n")
        self.assertTrue(any(e["error"] == "SHA256 mismatch" for e in verify(self.output)["errors"]))
        source.unlink()
        self.assertTrue(any(e["error"] == "missing file" for e in verify(self.output)["errors"]))

    def test_output_overlap_is_rejected_before_mutation(self):
        for output in (self.upstream, self.upstream / "generated", self.root):
            with self.assertRaisesRegex(ReconstructionError, "overlap"):
                reconstruct_sglang_das(self.upstream, self.plugin, output, force=True)
        self.assertTrue((self.upstream / "README.md").is_file())
        self.assertFalse((self.upstream / "generated").exists())

    def test_verifier_rejects_manifest_path_escape(self):
        reconstruct_sglang_das(self.upstream, self.plugin, self.output)
        manifest_path = self.output / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["../secret"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
        self.assertTrue(any(e["error"] == "path outside output" for e in verify(self.output)["errors"]))

    def test_verifier_cli_failure_is_nonzero(self):
        result = subprocess.run([sys.executable, str(ROOT / "tools/verify_reconstruction.py"),
                                 "--output", str(self.output)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout)["status"], "failed")


class MigrationPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.official = self.root / "official"
        self.fork = self.root / "fork"
        self.plugin = self.root / "plugin"
        self.official.mkdir()
        self.plugin.mkdir()
        _write(self.official / "python/sglang/runtime.py", "def run(x):\n    return x\n")
        _write(self.official / "python/sglang/inherited.py", "USE_HCU = True\n")
        _init_repo(self.official)
        self.git(self.root, "clone", "-q", str(self.official), str(self.fork))
        self.git(self.fork, "config", "user.name", "Test")
        self.git(self.fork, "config", "user.email", "test@example.com")
        _write(self.official / "python/sglang/inherited.py", "USE_HCU = False\n")
        self.git(self.official, "commit", "-qam", "upstream removes inherited HCU behavior")
        _write(self.fork / "python/sglang/runtime.py", "def run(x):\n    return x + 1\n")
        self.git(self.fork, "commit", "-qam", "new fork behavior")
        _write(self.fork / "dirty.py", "VALUE = 42\n")
        _write(self.plugin / "sglang_dcu_plugin/__init__.py",
               "def register():\n    Registry.register('sglang.runtime.run', hook)\n")
        _init_repo(self.plugin)
        self.sha = self.git(self.official, "rev-parse", "HEAD").strip()
        self.git(self.fork, "fetch", "-q", str(self.official), f"{self.sha}:refs/audit/test")

    def git(self, repo, *args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL)

    def tool(self, name, *args, check=True):
        return subprocess.run([sys.executable, str(ROOT / "tools" / name), *map(str, args)],
                              capture_output=True, text=True, check=check)

    def manifest(self):
        self.tool("build_migration_manifest.py", "--official-repo", self.official,
                  "--fork-repo", self.fork, "--plugin-repo", self.plugin,
                  "--official-ref", "refs/audit/test", "--output-dir", self.root / "manifest")
        return self.root / "manifest/migration_manifest.json"

    def test_manifest_includes_dirty_and_inherited_hardware_drift(self):
        manifest = json.loads(self.manifest().read_text())
        items = {item["path"]: item for item in manifest["items"]}
        self.assertTrue(items["dirty.py"]["working_tree"])
        self.assertEqual(items["python/sglang/inherited.py"]["source"], "direct_hardware_drift")
        self.assertEqual(manifest["baselines"]["official_head"], self.sha)
        self.assertEqual((self.fork / "dirty.py").read_text(), "VALUE = 42\n")

    def test_ledger_does_not_promote_ast_candidates_to_active(self):
        overrides = self.root / "overrides.json"
        overrides.write_text("{}")
        args = ["--manifest", self.manifest(), "--official-repo", self.official,
                "--fork-repo", self.fork, "--plugin-repo", self.plugin,
                "--overrides", overrides, "--output-json", self.root / "ledger.json",
                "--output-markdown", self.root / "ledger.md"]
        self.tool("build_disposition_ledger.py", *args)
        report = json.loads((self.root / "ledger.json").read_text())
        item = next(x for x in report["items"] if x["fork_path"] == "python/sglang/runtime.py")
        self.assertEqual(item["plugin_status"], "candidate")
        self.assertEqual(item["status"], "pending")
        self.assertIsNone(report["summary"]["registered_hook_targets"])
        targets = self.root / "targets.txt"
        targets.write_text("sglang.runtime.run\n")
        self.tool("build_disposition_ledger.py", *args, "--hook-targets-file", targets)
        report = json.loads((self.root / "ledger.json").read_text())
        self.assertEqual(report["summary"]["registered_hook_targets"], 1)
        self.assertTrue(all(x["status"] != "implemented" for x in report["items"]))

    def test_audit_keeps_untracked_file_and_parse_errors(self):
        _write(self.fork / "broken.py", "def broken(:\n")
        self.tool("audit_repositories.py", "--repo", f"fork={self.fork}",
                  "--repo", f"official={self.official}", "--compare", "fork=official",
                  "--output", self.root / "audit.json")
        report = json.loads((self.root / "audit.json").read_text())
        self.assertIn("dirty.py", report["repositories"]["fork"]["files"])
        self.assertIn("broken.py", report["repositories"]["fork"]["python"]["parse_errors"])

    def test_manifest_preserves_dirty_deletion(self):
        (self.fork / "python/sglang/runtime.py").unlink()
        manifest = json.loads(self.manifest().read_text())
        item = next(x for x in manifest["items"] if x["path"] == "python/sglang/runtime.py")
        self.assertTrue(item["working_tree"])
        self.assertIn("run", item["symbol_delta"]["removed"])

    def test_manifest_rejects_ref_worktree_mismatch(self):
        self.git(self.fork, "update-ref", "refs/audit/test", "HEAD")
        result = self.tool("build_migration_manifest.py", "--official-repo", self.official,
                           "--fork-repo", self.fork, "--plugin-repo", self.plugin,
                           "--official-ref", "refs/audit/test", "--output-dir", self.root / "manifest", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must match", result.stderr)


if __name__ == "__main__":
    unittest.main()
