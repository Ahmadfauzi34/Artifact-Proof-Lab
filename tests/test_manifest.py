import json
import unittest

from artifact_proof.errors import ManifestError
from artifact_proof.manifest import parse_manifest
from tests.support import encode_manifest, manifest_for


class ManifestContractTests(unittest.TestCase):
    def parse(self, value: dict):
        return parse_manifest(encode_manifest(value), manifest_path="ARTIFACT_PROOF.json")

    def test_minimal_manifest_parses(self):
        parsed = self.parse(manifest_for({"payload.txt": b"hello"}))
        self.assertEqual(parsed.name, "test-artifact")
        self.assertIn("payload.txt", parsed.files)

    def test_unknown_top_level_key_is_rejected(self):
        value = manifest_for({"payload.txt": b"hello"})
        value["surprise"] = True
        with self.assertRaises(ManifestError):
            self.parse(value)

    def test_uppercase_digest_is_rejected(self):
        value = manifest_for({"payload.txt": b"hello"})
        value["files"]["payload.txt"]["sha256"] = "A" * 64
        with self.assertRaises(ManifestError):
            self.parse(value)

    def test_self_hash_is_rejected(self):
        value = manifest_for({"ARTIFACT_PROOF.json": b"impossible"})
        with self.assertRaises(ManifestError):
            self.parse(value)

    def test_duplicate_check_id_is_rejected(self):
        check = {
            "id": "reference",
            "type": "environment",
            "python_implementation": "cpython",
            "python_major_minor": [3, 13],
            "platform_system": "Linux",
        }
        value = manifest_for({}, checks=[check, check])
        with self.assertRaises(ManifestError):
            self.parse(value)

    def test_check_cannot_reference_undeclared_file(self):
        check = {"id": "db", "type": "sqlite_integrity", "path": "brain.db"}
        with self.assertRaises(ManifestError):
            self.parse(manifest_for({}, checks=[check]))

    def test_duplicate_json_key_is_rejected(self):
        raw = b'{"format":"artifact-proof-manifest-v1","format":"duplicate"}'
        with self.assertRaises(ManifestError):
            parse_manifest(raw, manifest_path="ARTIFACT_PROOF.json")

    def test_invalid_profile_is_rejected(self):
        check = {
            "id": "db",
            "type": "sqlite_integrity",
            "path": "brain.db",
            "profiles": ["future"],
        }
        with self.assertRaises(ManifestError):
            self.parse(manifest_for({"brain.db": b"x"}, checks=[check]))
