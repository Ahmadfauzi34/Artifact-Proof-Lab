from pathlib import Path
import stat
import tempfile
import unittest
import warnings
import zipfile

from artifact_proof.errors import SourceError
from artifact_proof.source import DirectorySource, SourceLimits, ZipSource


class SourceBoundaryTests(unittest.TestCase):
    def test_directory_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "target.txt").write_text("target")
            (root / "link.txt").symlink_to(root / "target.txt")
            with self.assertRaises(SourceError):
                DirectorySource(root, SourceLimits())

    def test_zip_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("../escape.txt", b"no")
            with self.assertRaises(SourceError):
                ZipSource(path, SourceLimits())

    def test_duplicate_zip_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.zip"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("same.txt", b"one")
                    archive.writestr("same.txt", b"two")
            with self.assertRaises(SourceError):
                ZipSource(path, SourceLimits())

    def test_zip_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "symlink.zip"
            info = zipfile.ZipInfo("link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(info, "target")
            with self.assertRaises(SourceError):
                ZipSource(path, SourceLimits())

    def test_non_regular_zip_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fifo.zip"
            info = zipfile.ZipInfo("pipe")
            info.create_system = 3
            info.external_attr = (stat.S_IFIFO | 0o600) << 16
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(info, b"")
            with self.assertRaises(SourceError):
                ZipSource(path, SourceLimits())

    def test_compression_ratio_limit_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bomb.zip"
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("zeros.bin", b"0" * 1024 * 1024)
            with self.assertRaises(SourceError):
                ZipSource(path, SourceLimits(max_compression_ratio=10.0))

    def test_directory_file_replacement_after_inventory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "payload.txt"
            path.write_text("original")
            source = DirectorySource(root, SourceLimits())
            replacement = root / "replacement.txt"
            replacement.write_text("replacement")
            replacement.replace(path)
            with self.assertRaises(SourceError):
                source.read_bytes("payload.txt")
