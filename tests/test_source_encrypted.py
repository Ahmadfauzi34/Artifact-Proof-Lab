from __future__ import annotations

import struct
import tempfile
from pathlib import Path
import unittest
import zipfile
from io import BytesIO

from artifact_proof.engine import verify_artifact
from artifact_proof.model import Status
from artifact_proof.source import SourceLimits
from tests.support import encode_manifest, manifest_for, write_directory, write_zip


def statuses(report) -> dict[str, Status]:
    return {finding.check_id: finding.status for finding in report.findings}


def _set_encrypted_flag(raw: bytes, member: str) -> bytes:
    """Set the encrypted flag (0x1) on one member of a ZIP byte string.

    Python's zipfile clears flag_bits on writestr, so the flag must be
    patched into the raw local header (+6) and central entry (+8) after
    writing. Only the named member is flagged; other members are untouched.
    """
    data = bytearray(raw)
    target = member.encode("utf-8")
    # Local file headers: PK\x03\x04, flag at +6, name length at +26, extra at +28.
    offset = 0
    while True:
        at = data.find(b"PK\x03\x04", offset)
        if at < 0:
            break
        name_len, extra_len = struct.unpack_from("<HH", data, at + 26)
        name = bytes(data[at + 30: at + 30 + name_len])
        if name == target:
            struct.pack_into("<H", data, at + 6, struct.unpack_from("<H", data, at + 6)[0] | 0x1)
        offset = at + 30 + name_len + extra_len
    # Central directory entries: PK\x01\x02, flag at +8, name at +28, extra at +30, comment at +32.
    offset = 0
    while True:
        at = data.find(b"PK\x01\x02", offset)
        if at < 0:
            break
        name_len, extra_len, comment_len = struct.unpack_from("<HHH", data, at + 28)
        name = bytes(data[at + 46: at + 46 + name_len])
        if name == target:
            struct.pack_into("<H", data, at + 8, struct.unpack_from("<H", data, at + 8)[0] | 0x1)
        offset = at + 46 + name_len + extra_len + comment_len
    return bytes(data)


class EncryptedSourceTests(unittest.TestCase):
    def test_encrypted_zip_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            zpath = Path(directory) / "artifact.zip"
            files = {"payload.txt": b"hello"}
            write_zip(zpath, files, manifest_for(files))
            zpath.write_bytes(_set_encrypted_flag(zpath.read_bytes(), "payload.txt"))
            report = verify_artifact(zpath)
            self.assertFalse(report.passed)
            self.assertEqual(statuses(report)["artifact-contract"], Status.FAIL)

    def test_encrypted_nested_zip_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            nested = BytesIO()
            with zipfile.ZipFile(nested, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("payload.txt", b"payload-data")
            nested_bytes = _set_encrypted_flag(nested.getvalue(), "payload.txt")
            files = {"archive.zip": nested_bytes, "target.txt": b"payload-data"}
            checks = [{
                "id": "backup",
                "type": "zip_member_matches",
                "archive": "archive.zip",
                "member": "payload.txt",
                "target": "target.txt",
            }]
            write_directory(root, files, manifest_for(files, checks=checks))
            report = verify_artifact(root)
            self.assertEqual(statuses(report)["backup"], Status.FAIL)

    def test_zip_member_at_exact_size_limit_passes(self):
        # The limit applies to every file in the artifact, manifest included
        # (~250 bytes), so the payload must clear the manifest, not just the limit.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files = {"payload.txt": b"x" * 500}
            write_directory(root, files, manifest_for(files))
            report = verify_artifact(root, limits=SourceLimits(max_file_bytes=500))
            self.assertTrue(report.passed)

    def test_zip_member_above_size_limit_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            files = {"payload.txt": b"x" * 501}
            write_directory(root, files, manifest_for(files))
            report = verify_artifact(root, limits=SourceLimits(max_file_bytes=500))
            self.assertFalse(report.passed)
            self.assertEqual(statuses(report)["artifact-contract"], Status.FAIL)

    def test_compression_ratio_at_limit_passes_and_above_fails(self):
        payload = b"\x00" * 1000
        with tempfile.TemporaryDirectory() as directory:
            zpath = Path(directory) / "artifact.zip"
            files = {"payload.txt": payload}
            write_zip(zpath, files, manifest_for(files))
            generous = verify_artifact(zpath, limits=SourceLimits(max_compression_ratio=200.0))
            self.assertTrue(generous.passed)
            strict = verify_artifact(zpath, limits=SourceLimits(max_compression_ratio=10.0))
            self.assertFalse(strict.passed)
            self.assertEqual(statuses(strict)["artifact-contract"], Status.FAIL)


if __name__ == "__main__":
    unittest.main()
