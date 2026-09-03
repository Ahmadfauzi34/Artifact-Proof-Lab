from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from artifact_proof.engine import verify_artifact
from artifact_proof.model import Finding, Report, Status
from artifact_proof.source import DirectorySource, ZipSource
from tests.support import encode_manifest, manifest_for, nested_zip, sqlite_bytes, write_directory, write_zip


def statuses(report) -> dict[str, Status]:
    return {finding.check_id: finding.status for finding in report.findings}


class VerificationEngineTests(unittest.TestCase):
    def test_valid_directory_passes_with_unanchored_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files = {"payload.txt": b"hello"}
            write_directory(root, files, manifest_for(files))
            report = verify_artifact(root)
            self.assertTrue(report.passed)
            self.assertEqual(statuses(report)["manifest-anchor"], Status.WARN)
            self.assertEqual(statuses(report)["sha256:payload.txt"], Status.PASS)
            self.assertEqual(statuses(report)["source-snapshot"], Status.PASS)
            self.assertRegex(report.source_snapshot_sha256 or "", r"^[0-9a-f]{64}$")

    def test_valid_zip_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.zip"
            files = {"payload.txt": b"hello"}
            write_zip(path, files, manifest_for(files))
            report = verify_artifact(path)
            self.assertTrue(report.passed)
            self.assertEqual(statuses(report)["source-snapshot"], Status.PASS)

    def test_directory_change_after_earlier_hash_fails_finalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files = {"payload.txt": b"original", "z-trigger.txt": b"trigger"}
            write_directory(root, files, manifest_for(files))
            original_read = DirectorySource.read_bytes
            mutated = False

            def racing_read(source, relative, *, max_bytes=None):
                nonlocal mutated
                data = original_read(source, relative, max_bytes=max_bytes)
                if relative == "z-trigger.txt" and not mutated:
                    (root / "payload.txt").write_bytes(b"tampered")
                    mutated = True
                return data

            with patch.object(DirectorySource, "read_bytes", racing_read):
                report = verify_artifact(root)

            self.assertTrue(mutated)
            self.assertFalse(report.passed)
            self.assertEqual(statuses(report)["artifact-contract"], Status.FAIL)
            self.assertIsNone(report.source_snapshot_sha256)

    def test_zip_path_replacement_during_validation_fails_finalization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.zip"
            replacement = Path(directory) / "replacement.zip"
            files = {"payload.txt": b"original", "z-trigger.txt": b"trigger"}
            write_zip(path, files, manifest_for(files))
            changed = {"payload.txt": b"tampered", "z-trigger.txt": b"trigger"}
            write_zip(replacement, changed, manifest_for(changed))
            original_read = ZipSource.read_bytes
            replaced = False

            def racing_read(source, relative, *, max_bytes=None):
                nonlocal replaced
                data = original_read(source, relative, max_bytes=max_bytes)
                if relative == "z-trigger.txt" and not replaced:
                    replacement.replace(path)
                    replaced = True
                return data

            with patch.object(ZipSource, "read_bytes", racing_read):
                report = verify_artifact(path)

            self.assertTrue(replaced)
            self.assertFalse(report.passed)
            self.assertEqual(statuses(report)["artifact-contract"], Status.FAIL)
            self.assertIsNone(report.source_snapshot_sha256)

    def test_tampered_file_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files = {"payload.txt": b"original"}
            write_directory(root, files, manifest_for(files))
            (root / "payload.txt").write_bytes(b"tampered")
            report = verify_artifact(root)
            self.assertFalse(report.passed)
            self.assertEqual(statuses(report)["sha256:payload.txt"], Status.FAIL)

    def test_unlisted_file_fails_complete_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files = {"payload.txt": b"hello"}
            write_directory(root, files, manifest_for(files))
            (root / "hidden.txt").write_text("not declared")
            report = verify_artifact(root)
            self.assertEqual(statuses(report)["file-coverage"], Status.FAIL)

    def test_explicitly_allowed_unlisted_file_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files = {"payload.txt": b"hello"}
            manifest = manifest_for(files, allow_unlisted=["runtime.log"])
            write_directory(root, files, manifest)
            (root / "runtime.log").write_text("live output")
            self.assertTrue(verify_artifact(root).passed)

    def test_mutable_hash_drift_fails_sealed_but_skips_live(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files = {"state.bin": b"packaged"}
            write_directory(root, files, manifest_for(files, mutable={"state.bin"}))
            (root / "state.bin").write_bytes(b"learned-later")
            sealed = verify_artifact(root, profile="sealed")
            live = verify_artifact(root, profile="live")
            self.assertFalse(sealed.passed)
            self.assertTrue(live.passed)
            self.assertEqual(statuses(live)["sha256:state.bin"], Status.SKIP)

    def test_matching_detached_anchor_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files = {"payload.txt": b"hello"}
            write_directory(root, files, manifest_for(files))
            digest = hashlib.sha256((root / "ARTIFACT_PROOF.json").read_bytes()).hexdigest()
            report = verify_artifact(root, manifest_sha256=digest, require_trust_anchor=True)
            self.assertTrue(report.passed)
            self.assertEqual(statuses(report)["manifest-anchor"], Status.PASS)

    def test_missing_required_anchor_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files = {"payload.txt": b"hello"}
            write_directory(root, files, manifest_for(files))
            report = verify_artifact(root, require_trust_anchor=True)
            self.assertEqual(statuses(report)["manifest-anchor"], Status.FAIL)

    def test_wrong_anchor_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files = {"payload.txt": b"hello"}
            write_directory(root, files, manifest_for(files))
            report = verify_artifact(root, manifest_sha256="0" * 64)
            self.assertEqual(statuses(report)["manifest-anchor"], Status.FAIL)

    def test_valid_sqlite_passes_full_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files = {"state/brain.db": sqlite_bytes()}
            checks = [{"id": "brain", "type": "sqlite_integrity", "path": "state/brain.db", "mode": "full"}]
            write_directory(root, files, manifest_for(files, checks=checks))
            report = verify_artifact(root)
            self.assertEqual(statuses(report)["brain"], Status.PASS)

    def test_invalid_sqlite_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files = {"state/brain.db": b"not sqlite"}
            checks = [{"id": "brain", "type": "sqlite_integrity", "path": "state/brain.db"}]
            write_directory(root, files, manifest_for(files, checks=checks))
            report = verify_artifact(root)
            self.assertEqual(statuses(report)["brain"], Status.FAIL)

    def test_sealed_backup_identity_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            brain = sqlite_bytes()
            files = {
                "state/brain.db": brain,
                "state/brain-backup.zip": nested_zip("brain.db", brain),
            }
            checks = [{
                "id": "cold-backup",
                "type": "zip_member_matches",
                "archive": "state/brain-backup.zip",
                "member": "brain.db",
                "target": "state/brain.db",
                "only_member": True,
                "profiles": ["sealed"],
            }]
            write_directory(root, files, manifest_for(files, checks=checks))
            report = verify_artifact(root)
            self.assertEqual(statuses(report)["cold-backup"], Status.PASS)

    def test_stale_backup_fails_sealed_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            brain = sqlite_bytes()
            files = {
                "state/brain.db": brain,
                "state/brain-backup.zip": nested_zip("brain.db", b"older state"),
            }
            checks = [{
                "id": "cold-backup",
                "type": "zip_member_matches",
                "archive": "state/brain-backup.zip",
                "member": "brain.db",
                "target": "state/brain.db",
                "only_member": True,
            }]
            write_directory(root, files, manifest_for(files, checks=checks))
            report = verify_artifact(root)
            self.assertEqual(statuses(report)["cold-backup"], Status.FAIL)

    def test_nested_backup_extra_member_fails_exclusive_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            brain = b"brain"
            files = {
                "brain.db": brain,
                "backup.zip": nested_zip("brain.db", brain, extra={"extra.txt": b"no"}),
            }
            checks = [{
                "id": "backup-shape",
                "type": "zip_member_matches",
                "archive": "backup.zip",
                "member": "brain.db",
                "target": "brain.db",
                "only_member": True,
            }]
            write_directory(root, files, manifest_for(files, checks=checks))
            self.assertEqual(statuses(verify_artifact(root))["backup-shape"], Status.FAIL)

    def test_nonbinding_environment_difference_warns_not_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            checks = [{
                "id": "future-reference",
                "type": "environment",
                "python_implementation": "imaginary-python",
                "python_major_minor": [99, 0],
                "platform_system": "MoonOS",
                "enforce": False,
            }]
            write_directory(root, {}, manifest_for({}, checks=checks))
            report = verify_artifact(root)
            self.assertTrue(report.passed)
            self.assertEqual(statuses(report)["future-reference"], Status.WARN)

    def test_enforced_environment_difference_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            checks = [{
                "id": "required-environment",
                "type": "environment",
                "python_implementation": "imaginary-python",
                "python_major_minor": [99, 0],
                "platform_system": "MoonOS",
                "enforce": True,
            }]
            write_directory(root, {}, manifest_for({}, checks=checks))
            self.assertEqual(statuses(verify_artifact(root))["required-environment"], Status.FAIL)

    def test_malicious_outer_zip_becomes_failed_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("../escape", b"no")
            report = verify_artifact(path)
            self.assertFalse(report.passed)
            self.assertEqual(statuses(report)["artifact-contract"], Status.FAIL)


class ReportDeterminismTests(unittest.TestCase):
    def test_report_is_json_serializable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            write_directory(root, {}, manifest_for({}))
            rendered = json.dumps(verify_artifact(root).to_dict(), sort_keys=True)
            self.assertIn('"format": "artifact-proof-report-v1"', rendered)
            self.assertIn('"source_snapshot_sha256"', rendered)

    def test_failed_report_verdict_cannot_be_rewritten_after_finalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files = {"payload.txt": b"original"}
            write_directory(root, files, manifest_for(files))
            (root / "payload.txt").write_bytes(b"tampered")

            report = verify_artifact(root)
            original = report.to_dict()
            self.assertEqual(report.status, "FAIL")

            with self.assertRaises(AttributeError):
                report.findings.clear()
            with self.assertRaises(FrozenInstanceError):
                report.findings = ()

            self.assertEqual(report.to_dict(), original)
            self.assertEqual(report.status, "FAIL")

    def test_report_snapshots_builder_findings_without_hidden_size_limit(self):
        builder_findings = [
            Finding("contract", "contract", Status.FAIL, "rejected")
        ]
        report = Report("artifact", "sealed", findings=builder_findings)
        builder_findings.clear()

        self.assertEqual(len(report.findings), 1)
        self.assertEqual(report.status, "FAIL")
