from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from artifact_proof.engine import verify_artifact
from artifact_proof.model import Status
from tests.support import manifest_for, write_directory


def _finding(report, check_id: str):
    return next(finding for finding in report.findings if finding.check_id == check_id)


def _binding_manifest(files: dict[str, bytes]) -> dict:
    checks = [{
        "id": "value-binding",
        "type": "value_binding",
        "operands": [
            {"kind": "json_pointer", "path": "left.json", "pointer": "/value"},
            {"kind": "json_pointer", "path": "right.json", "pointer": "/value"},
        ],
    }]
    return manifest_for(files, checks=checks)


class ValueBindingEdgeTests(unittest.TestCase):
    def test_non_standard_numeric_constant_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files = {
                "left.json": b'{"value":NaN}\n',
                "right.json": b'{"value":NaN}\n',
            }
            write_directory(root, files, _binding_manifest(files))

            report = verify_artifact(root)
            finding = _finding(report, "value-binding")

            self.assertFalse(report.passed)
            self.assertEqual(finding.status, Status.FAIL)
            self.assertIn("non-standard JSON constant", finding.message)

    def test_escaped_unpaired_surrogate_is_committed_without_encoder_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files = {
                "left.json": b'{"value":"\\ud800"}\n',
                "right.json": b'{"value":"\\ud800"}\n',
            }
            write_directory(root, files, _binding_manifest(files))

            report = verify_artifact(root)
            finding = _finding(report, "value-binding")

            self.assertTrue(report.passed)
            self.assertEqual(finding.status, Status.PASS)


if __name__ == "__main__":
    unittest.main()
