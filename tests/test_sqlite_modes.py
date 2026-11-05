from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from artifact_proof.engine import verify_artifact
from artifact_proof.model import Status
from tests.support import manifest_for, sqlite_bytes, write_directory


def statuses(report) -> dict[str, Status]:
    return {finding.check_id: finding.status for finding in report.findings}


class SqliteModeTests(unittest.TestCase):
    def test_quick_mode_passes_on_valid_db(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files = {"state/brain.db": sqlite_bytes()}
            checks = [{"id": "brain", "type": "sqlite_integrity", "path": "state/brain.db", "mode": "quick"}]
            write_directory(root, files, manifest_for(files, checks=checks))
            self.assertEqual(statuses(verify_artifact(root))["brain"], Status.PASS)

    def test_quick_and_full_agree_on_corrupt_db(self):
        for mode in ("quick", "full"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "artifact"
                files = {"state/brain.db": b"not sqlite"}
                checks = [{"id": "brain", "type": "sqlite_integrity", "path": "state/brain.db", "mode": mode}]
                write_directory(root, files, manifest_for(files, checks=checks))
                self.assertEqual(statuses(verify_artifact(root))["brain"], Status.FAIL)

    def test_truncated_db_fails_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            valid = sqlite_bytes()
            files = {"state/brain.db": valid[: len(valid) // 2]}
            checks = [{"id": "brain", "type": "sqlite_integrity", "path": "state/brain.db", "mode": "full"}]
            write_directory(root, files, manifest_for(files, checks=checks))
            self.assertEqual(statuses(verify_artifact(root))["brain"], Status.FAIL)


if __name__ == "__main__":
    unittest.main()
