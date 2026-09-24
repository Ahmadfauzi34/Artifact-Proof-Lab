from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from artifact_proof.engine import verify_artifact
from artifact_proof.model import Status
from tests.support import manifest_for, write_directory


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")


def _finding(report, check_id: str):
    return next(finding for finding in report.findings if finding.check_id == check_id)


class ValueBindingTests(unittest.TestCase):
    def test_json_fields_can_bind_to_physical_file_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            payload = b"durable checkpoint bytes"
            digest = hashlib.sha256(payload).hexdigest()
            files = {
                "AGENT_MANIFEST.json": _json_bytes({"state_contract": {"backup_sha256": digest}}),
                "CAPSULE_CONTRACT.json": _json_bytes({"state": {"backup_sha256": digest}}),
                "state/brain-backup.zip": payload,
            }
            checks = [{
                "id": "backup-identity-coherence",
                "type": "value_binding",
                "operands": [
                    {"kind": "json_pointer", "path": "AGENT_MANIFEST.json", "pointer": "/state_contract/backup_sha256"},
                    {"kind": "json_pointer", "path": "CAPSULE_CONTRACT.json", "pointer": "/state/backup_sha256"},
                    {"kind": "file_sha256", "path": "state/brain-backup.zip"},
                ],
            }]
            write_directory(root, files, manifest_for(files, checks=checks))

            report = verify_artifact(root)
            finding = _finding(report, "backup-identity-coherence")

            self.assertTrue(report.passed)
            self.assertEqual(finding.status, Status.PASS)
            self.assertNotIn(digest, json.dumps(finding.to_dict(), sort_keys=True))
            self.assertEqual(len(finding.observed), 3)

    def test_stale_cross_document_metadata_fails_even_when_each_file_hash_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            payload = b"current checkpoint bytes"
            current_digest = hashlib.sha256(payload).hexdigest()
            stale_digest = hashlib.sha256(b"older checkpoint bytes").hexdigest()
            files = {
                "AGENT_MANIFEST.json": _json_bytes({"state_contract": {"backup_sha256": stale_digest}}),
                "STATE_RECOVERY_CONTRACT.json": _json_bytes({"backup_sha256": current_digest}),
                "state/brain-backup.zip": payload,
            }
            checks = [{
                "id": "backup-identity-coherence",
                "type": "value_binding",
                "operands": [
                    {"kind": "json_pointer", "path": "AGENT_MANIFEST.json", "pointer": "/state_contract/backup_sha256"},
                    {"kind": "json_pointer", "path": "STATE_RECOVERY_CONTRACT.json", "pointer": "/backup_sha256"},
                    {"kind": "file_sha256", "path": "state/brain-backup.zip"},
                ],
            }]
            write_directory(root, files, manifest_for(files, checks=checks))

            report = verify_artifact(root)

            self.assertEqual(_finding(report, "sha256:AGENT_MANIFEST.json").status, Status.PASS)
            self.assertEqual(_finding(report, "sha256:STATE_RECOVERY_CONTRACT.json").status, Status.PASS)
            self.assertEqual(_finding(report, "sha256:state/brain-backup.zip").status, Status.PASS)
            self.assertEqual(_finding(report, "backup-identity-coherence").status, Status.FAIL)
            self.assertFalse(report.passed)

    def test_missing_json_pointer_fails_the_declared_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files = {
                "left.json": _json_bytes({"identity": "same"}),
                "right.json": _json_bytes({"other": "same"}),
            }
            checks = [{
                "id": "identity-binding",
                "type": "value_binding",
                "operands": [
                    {"kind": "json_pointer", "path": "left.json", "pointer": "/identity"},
                    {"kind": "json_pointer", "path": "right.json", "pointer": "/identity"},
                ],
            }]
            write_directory(root, files, manifest_for(files, checks=checks))

            finding = _finding(verify_artifact(root), "identity-binding")

            self.assertEqual(finding.status, Status.FAIL)
            self.assertIn("JSON Pointer member is missing", finding.message)

    def test_duplicate_keys_in_bound_json_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files = {
                "left.json": b'{"identity":"first","identity":"second"}\n',
                "right.json": _json_bytes({"identity": "second"}),
            }
            checks = [{
                "id": "identity-binding",
                "type": "value_binding",
                "operands": [
                    {"kind": "json_pointer", "path": "left.json", "pointer": "/identity"},
                    {"kind": "json_pointer", "path": "right.json", "pointer": "/identity"},
                ],
            }]
            write_directory(root, files, manifest_for(files, checks=checks))

            finding = _finding(verify_artifact(root), "identity-binding")

            self.assertEqual(finding.status, Status.FAIL)
            self.assertIn("duplicate JSON key", finding.message)

    def test_json_pointer_escaping_and_array_index_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files = {
                "left.json": _json_bytes({"a/b": {"~key": ["bound"]}}),
                "right.json": _json_bytes({"identity": "bound"}),
            }
            checks = [{
                "id": "escaped-pointer-binding",
                "type": "value_binding",
                "operands": [
                    {"kind": "json_pointer", "path": "left.json", "pointer": "/a~1b/~0key/0"},
                    {"kind": "json_pointer", "path": "right.json", "pointer": "/identity"},
                ],
            }]
            write_directory(root, files, manifest_for(files, checks=checks))

            self.assertEqual(_finding(verify_artifact(root), "escaped-pointer-binding").status, Status.PASS)

    def test_invalid_pointer_escape_is_rejected_by_manifest_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files = {
                "left.json": _json_bytes({"identity": "bound"}),
                "right.json": _json_bytes({"identity": "bound"}),
            }
            checks = [{
                "id": "identity-binding",
                "type": "value_binding",
                "operands": [
                    {"kind": "json_pointer", "path": "left.json", "pointer": "/bad~2escape"},
                    {"kind": "json_pointer", "path": "right.json", "pointer": "/identity"},
                ],
            }]
            write_directory(root, files, manifest_for(files, checks=checks))

            report = verify_artifact(root)

            self.assertFalse(report.passed)
            self.assertEqual(_finding(report, "artifact-contract").status, Status.FAIL)
            self.assertIn("invalid JSON Pointer escape", _finding(report, "artifact-contract").message)

    def test_binding_operands_must_reference_declared_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files = {"left.json": _json_bytes({"identity": "bound"})}
            checks = [{
                "id": "identity-binding",
                "type": "value_binding",
                "operands": [
                    {"kind": "json_pointer", "path": "left.json", "pointer": "/identity"},
                    {"kind": "file_sha256", "path": "undeclared.bin"},
                ],
            }]
            write_directory(root, files, manifest_for(files, checks=checks))

            report = verify_artifact(root)

            self.assertFalse(report.passed)
            self.assertIn("references undeclared file", _finding(report, "artifact-contract").message)


if __name__ == "__main__":
    unittest.main()
