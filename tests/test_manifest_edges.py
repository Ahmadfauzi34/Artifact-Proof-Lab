from __future__ import annotations

import unittest

from artifact_proof.errors import ManifestError
from artifact_proof.manifest import parse_manifest
from tests.support import encode_manifest, manifest_for


class ManifestEdgeTests(unittest.TestCase):
    def _parse(self, manifest: dict) -> None:
        parse_manifest(encode_manifest(manifest), manifest_path="ARTIFACT_PROOF.json")

    def test_binding_with_fewer_than_two_operands_is_rejected(self):
        manifest = manifest_for({"a.json": b'{"value":1}\n'}, checks=[{
            "id": "vb",
            "type": "value_binding",
            "operands": [{"kind": "json_pointer", "path": "a.json", "pointer": "/value"}],
        }])
        with self.assertRaises(ManifestError):
            self._parse(manifest)

    def test_unsupported_binding_operand_kind_is_rejected(self):
        manifest = manifest_for({"a.json": b'{"value":1}\n'}, checks=[{
            "id": "vb",
            "type": "value_binding",
            "operands": [
                {"kind": "magic", "path": "a.json", "pointer": "/value"},
                {"kind": "json_pointer", "path": "a.json", "pointer": "/value"},
            ],
        }])
        with self.assertRaises(ManifestError):
            self._parse(manifest)

    def test_invalid_sqlite_mode_is_rejected(self):
        manifest = manifest_for({"state/brain.db": b"db"}, checks=[{
            "id": "brain",
            "type": "sqlite_integrity",
            "path": "state/brain.db",
            "mode": "sometimes",
        }])
        with self.assertRaises(ManifestError):
            self._parse(manifest)

    def test_invalid_python_major_minor_is_rejected(self):
        manifest = manifest_for({}, checks=[{
            "id": "env",
            "type": "environment",
            "python_implementation": "cpython",
            "python_major_minor": [3],
            "platform_system": "Linux",
        }])
        with self.assertRaises(ManifestError):
            self._parse(manifest)

    def test_non_boolean_enforce_is_rejected(self):
        manifest = manifest_for({}, checks=[{
            "id": "env",
            "type": "environment",
            "python_implementation": "cpython",
            "python_major_minor": [3, 12],
            "platform_system": "Linux",
            "enforce": "yes",
        }])
        with self.assertRaises(ManifestError):
            self._parse(manifest)

    def test_manifest_over_1mib_is_rejected(self):
        with self.assertRaises(ManifestError):
            parse_manifest(b"x" * (1024 * 1024 + 1), manifest_path="ARTIFACT_PROOF.json")

    def test_non_utf8_manifest_is_rejected(self):
        with self.assertRaises(ManifestError):
            parse_manifest(b"\xff\xfe{}", manifest_path="ARTIFACT_PROOF.json")

    def test_wrong_manifest_format_is_rejected(self):
        manifest = manifest_for({"payload.txt": b"hello"})
        manifest["format"] = "artifact-proof-manifest-v2"
        with self.assertRaises(ManifestError):
            self._parse(manifest)


if __name__ == "__main__":
    unittest.main()
