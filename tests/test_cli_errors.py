from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

from artifact_proof.cli import main
from tests.support import manifest_for, write_directory


class CliErrorTests(unittest.TestCase):
    def test_require_trust_anchor_without_anchor_exits_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            write_directory(root, {"payload.txt": b"hello"}, manifest_for({"payload.txt": b"hello"}))
            with redirect_stdout(StringIO()):
                code = main(["verify", str(root), "--require-trust-anchor"])
            self.assertEqual(code, 1)

    def test_missing_artifact_path_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "no-such-artifact"
            with redirect_stdout(StringIO()):
                code = main(["verify", str(missing)])
            self.assertEqual(code, 1)

    def test_malformed_manifest_sha256_flag_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            write_directory(root, {"payload.txt": b"hello"}, manifest_for({"payload.txt": b"hello"}))
            with redirect_stdout(StringIO()):
                code = main(["verify", str(root), "--manifest-sha256", "not-hex"])
            self.assertEqual(code, 1)

    def test_invalid_json_manifest_via_cli_exits_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            root.mkdir(parents=True, exist_ok=True)
            (root / "payload.txt").write_bytes(b"hello")
            (root / "ARTIFACT_PROOF.json").write_bytes(b"{not json")
            with redirect_stdout(StringIO()):
                code = main(["verify", str(root)])
            self.assertEqual(code, 1)

    def test_unknown_argument_exits_two(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            write_directory(root, {"payload.txt": b"hello"}, manifest_for({"payload.txt": b"hello"}))
            with redirect_stdout(StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    main(["verify", str(root), "--bogus"])
            self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
