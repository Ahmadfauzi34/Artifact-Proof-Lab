from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import zipfile


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_for(
    files: dict[str, bytes],
    *,
    mutable: set[str] | None = None,
    checks: list[dict] | None = None,
    complete: bool = True,
    allow_unlisted: list[str] | None = None,
) -> dict:
    mutable = mutable or set()
    return {
        "format": "artifact-proof-manifest-v1",
        "artifact": {"name": "test-artifact", "version": "1"},
        "files": {
            path: {"sha256": sha256(data), **({"mutable": True} if path in mutable else {})}
            for path, data in sorted(files.items())
        },
        "checks": checks or [],
        "coverage": {"complete": complete, "allow_unlisted": allow_unlisted or []},
    }


def encode_manifest(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def write_directory(root: Path, files: dict[str, bytes], manifest: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for relative, data in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (root / "ARTIFACT_PROOF.json").write_bytes(encode_manifest(manifest))
    return root


def write_zip(path: Path, files: dict[str, bytes], manifest: dict) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative, data in files.items():
            archive.writestr(relative, data)
        archive.writestr("ARTIFACT_PROOF.json", encode_manifest(manifest))
    return path


def sqlite_bytes() -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".db") as handle:
        connection = sqlite3.connect(handle.name)
        connection.execute("CREATE TABLE proof_state(id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO proof_state(value) VALUES ('validated')")
        connection.commit()
        connection.close()
        return Path(handle.name).read_bytes()


def nested_zip(member: str, data: bytes, *, extra: dict[str, bytes] | None = None) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, data)
        for path, content in (extra or {}).items():
            archive.writestr(path, content)
    return output.getvalue()
