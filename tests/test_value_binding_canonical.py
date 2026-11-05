from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from artifact_proof.engine import verify_artifact
from artifact_proof.model import Status
from tests.support import manifest_for, write_directory


def _finding(report, check_id: str):
    return next(finding for finding in report.findings if finding.check_id == check_id)


def _binding(left: bytes, right: bytes, left_pointer: str = "/value", right_pointer: str = "/value"):
    files = {"left.json": left, "right.json": right}
    checks = [{
        "id": "value-binding",
        "type": "value_binding",
        "operands": [
            {"kind": "json_pointer", "path": "left.json", "pointer": left_pointer},
            {"kind": "json_pointer", "path": "right.json", "pointer": right_pointer},
        ],
    }]
    return files, checks


class ValueBindingCanonicalTests(unittest.TestCase):
    def test_null_and_false_do_not_bind(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files, checks = _binding(b'{"value":null}\n', b'{"value":false}\n')
            write_directory(root, files, manifest_for(files, checks=checks))
            self.assertEqual(_finding(verify_artifact(root), "value-binding").status, Status.FAIL)

    def test_false_and_zero_do_not_bind(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files, checks = _binding(b'{"value":false}\n', b'{"value":0}\n')
            write_directory(root, files, manifest_for(files, checks=checks))
            self.assertEqual(_finding(verify_artifact(root), "value-binding").status, Status.FAIL)

    def test_string_one_and_number_one_do_not_bind(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files, checks = _binding(b'{"value":"1"}\n', b'{"value":1}\n')
            write_directory(root, files, manifest_for(files, checks=checks))
            self.assertEqual(_finding(verify_artifact(root), "value-binding").status, Status.FAIL)

    def test_object_member_order_does_not_break_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files, checks = _binding(b'{"value":{"a":1,"b":2}}\n', b'{"value":{"b":2,"a":1}}\n')
            write_directory(root, files, manifest_for(files, checks=checks))
            self.assertEqual(_finding(verify_artifact(root), "value-binding").status, Status.PASS)

    def test_array_order_breaks_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files, checks = _binding(b'{"value":[1,2]}\n', b'{"value":[2,1]}\n')
            write_directory(root, files, manifest_for(files, checks=checks))
            self.assertEqual(_finding(verify_artifact(root), "value-binding").status, Status.FAIL)

    def test_empty_pointer_binds_whole_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files, checks = _binding(b'{"x":1}\n', b'{"x":1}\n', "", "")
            write_directory(root, files, manifest_for(files, checks=checks))
            self.assertEqual(_finding(verify_artifact(root), "value-binding").status, Status.PASS)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files, checks = _binding(b'{"x":1}\n', b'{"x":2}\n', "", "")
            write_directory(root, files, manifest_for(files, checks=checks))
            self.assertEqual(_finding(verify_artifact(root), "value-binding").status, Status.FAIL)

    def test_binding_report_contains_no_raw_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files, checks = _binding(b'{"value":"SECRET-XYZ"}\n', b'{"value":"SECRET-XYZ"}\n')
            write_directory(root, files, manifest_for(files, checks=checks))
            report = verify_artifact(root).to_dict()
            self.assertNotIn("SECRET-XYZ", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
