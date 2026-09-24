from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

from gym.binary_runtime_entry import ROLE_MODULES, _allowed_temp_script, _execute, _resolve
from scripts.build_binary_runtime import build_compatibility_metadata, build_manifest, copy_payload
from tests.support import manifest_for, write_directory


class BinaryRuntimeEntryTests(unittest.TestCase):
    def test_module_role_resolves(self):
        self.assertEqual(
            _resolve(["adaptive-train"]),
            ("module", "gym.run_adaptive_training_reference", []),
        )

    def test_artifact_verify_role_resolves_without_general_module_access(self):
        self.assertEqual(
            _resolve(["artifact-verify", "verify", "/tmp/artifact", "--json"]),
            ("module", "artifact_proof", ["verify", "/tmp/artifact", "--json"]),
        )
        with self.assertRaises(SystemExit):
            _resolve(["-m", "json"])

    def test_artifact_verify_role_executes_proof_cli(self):
        with tempfile.TemporaryDirectory(prefix="gym-artifact-verify-") as tmp:
            root = Path(tmp) / "artifact"
            files = {"payload.txt": b"hello"}
            write_directory(root, files, manifest_for(files))
            mode, target, rest = _resolve(["artifact-verify", "verify", str(root), "--json"])
            stdout = StringIO()
            with redirect_stdout(stdout):
                result = _execute(mode, target, rest)
            self.assertEqual(result, 0)
            self.assertIn('"status": "PASS"', stdout.getvalue())

    def test_allowed_temp_script_resolves(self):
        with tempfile.TemporaryDirectory(prefix="gym-runtime-test-") as tmp:
            script = Path(tmp) / "probe.py"
            script.write_text("pass\n")
            self.assertEqual(
                _resolve(["-B", str(script)]),
                ("script", str(script.resolve()), []),
            )

    def test_outside_script_is_rejected(self):
        root = Path(__file__).resolve().parent
        script = root / "_binary_runtime_outside_probe.py"
        script.write_text("pass\n")
        try:
            self.assertIsNone(_allowed_temp_script(str(script)))
            with self.assertRaises(SystemExit):
                _resolve(["-B", str(script)])
        finally:
            script.unlink(missing_ok=True)

    def test_temp_script_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="gym-runtime-test-") as tmp:
            root = Path(tmp)
            target = root / "target.py"
            link = root / "link.py"
            target.write_text("pass\n")
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("symlink unavailable")
            self.assertIsNone(_allowed_temp_script(str(link)))

    def test_script_mode_matches_sibling_import_semantics(self):
        with tempfile.TemporaryDirectory(prefix="gym-runtime-test-") as tmp:
            root = Path(tmp)
            (root / "helper.py").write_text("VALUE = 42\n")
            marker = root / "marker.txt"
            script = root / "probe.py"
            script.write_text(
                "from pathlib import Path\n"
                "from helper import VALUE\n"
                f"Path({str(marker)!r}).write_text(str(VALUE))\n"
            )
            mode, target, rest = _resolve(["-B", str(script)])
            self.assertEqual(_execute(mode, target, rest), 0)
            self.assertEqual(marker.read_text(), "42")


class BinaryRuntimeCompatibilityTests(unittest.TestCase):
    def test_training_surface_metadata_is_stable_and_explicit(self):
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="gym-binary-compat-") as tmp:
            output = Path(tmp)
            copy_payload(project_root, output)
            first = build_compatibility_metadata(output)
            second = build_compatibility_metadata(output)
            self.assertEqual(first, second)
            self.assertEqual(
                first["schema"],
                "proof-gym-binary-runtime-compatibility-v1",
            )
            self.assertEqual(first["contract_version"], 1)
            self.assertEqual(first["training_role"], "external-agent-train")
            self.assertEqual(
                first["agent_protocol"],
                "proof-gym-agent-jsonl-v1",
            )
            self.assertEqual(
                first["external_training_receipt_format"],
                "proof-gym-external-training-receipt-v1",
            )
            self.assertFalse(first["learning_write_authority"])
            self.assertGreater(first["training_surface_file_count"], 20)
            self.assertEqual(len(first["training_surface_sha256"]), 64)
            self.assertEqual(
                first["aggregation_policy"]["mismatch_action"],
                "ISOLATE_EVIDENCE",
            )

    def test_binary_manifest_roles_are_derived_from_bundled_launcher(self):
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="gym-binary-manifest-") as tmp:
            output = Path(tmp)
            copy_payload(project_root, output)
            (output / "proof-gym-runtime").write_bytes(b"test-binary")
            manifest = build_manifest(output, "source", "head")
            self.assertEqual(manifest["runtime_roles"], sorted(ROLE_MODULES))
            self.assertIn("artifact-verify", manifest["runtime_roles"])


if __name__ == "__main__":
    unittest.main()
