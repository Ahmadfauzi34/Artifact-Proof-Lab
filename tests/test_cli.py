from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from artifact_proof.cli import main
from tests.support import manifest_for, write_directory


class CliTests(unittest.TestCase):
    def test_json_success_returns_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            write_directory(root, {"payload.txt": b"hello"}, manifest_for({"payload.txt": b"hello"}))
            output = StringIO()
            with redirect_stdout(output):
                code = main(["verify", str(root), "--json"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.getvalue())["status"], "PASS")

    def test_tamper_returns_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            write_directory(root, {"payload.txt": b"original"}, manifest_for({"payload.txt": b"original"}))
            (root / "payload.txt").write_bytes(b"tampered")
            with redirect_stdout(StringIO()):
                code = main(["verify", str(root)])
            self.assertEqual(code, 1)
