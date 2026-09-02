from pathlib import Path
import tempfile
import unittest

from artifact_proof.errors import SourceError
from artifact_proof.paths import canonical_relative_path, safe_join


class PortablePathTests(unittest.TestCase):
    def test_nested_portable_path_is_accepted(self):
        self.assertEqual(canonical_relative_path("state/brain.db"), "state/brain.db")

    def test_parent_escape_is_rejected(self):
        with self.assertRaises(SourceError):
            canonical_relative_path("../brain.db")

    def test_embedded_parent_escape_is_rejected(self):
        with self.assertRaises(SourceError):
            canonical_relative_path("state/../brain.db")

    def test_absolute_path_is_rejected(self):
        with self.assertRaises(SourceError):
            canonical_relative_path("/state/brain.db")

    def test_windows_drive_is_rejected(self):
        with self.assertRaises(SourceError):
            canonical_relative_path("C:/state/brain.db")

    def test_backslash_is_rejected(self):
        with self.assertRaises(SourceError):
            canonical_relative_path("state\\brain.db")

    def test_empty_segment_is_rejected(self):
        with self.assertRaises(SourceError):
            canonical_relative_path("state//brain.db")

    def test_safe_join_stays_inside_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(safe_join(root, "a/b.txt"), root / "a" / "b.txt")
