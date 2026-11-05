from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from artifact_proof.model import Finding, Report, Status


class ReportImmutabilityTests(unittest.TestCase):
    def test_finding_assignment_raises(self):
        finding = Finding("x", "coverage", Status.PASS, "ok")
        with self.assertRaises(FrozenInstanceError):
            finding.status = Status.FAIL

    def test_report_assignment_raises(self):
        report = Report(artifact="a", profile="sealed")
        with self.assertRaises(FrozenInstanceError):
            report.artifact = "b"


if __name__ == "__main__":
    unittest.main()
