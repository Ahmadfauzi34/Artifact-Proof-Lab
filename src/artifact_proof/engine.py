from __future__ import annotations

import hashlib
from pathlib import Path
import re

from .checks import run_declared_check, verify_coverage, verify_file_hashes, verify_manifest_anchor
from .errors import ArtifactProofError, ManifestError
from .manifest import SUPPORTED_PROFILES, parse_manifest
from .model import Finding, Report, Status
from .paths import canonical_relative_path
from .source import SourceLimits, open_source


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def verify_artifact(
    artifact: Path | str,
    *,
    manifest_path: str = "ARTIFACT_PROOF.json",
    profile: str = "sealed",
    manifest_sha256: str | None = None,
    require_trust_anchor: bool = False,
    limits: SourceLimits | None = None,
) -> Report:
    findings: list[Finding] = []
    observed_manifest_sha256: str | None = None
    artifact_name: str | None = None
    artifact_version: str | None = None
    try:
        if profile not in SUPPORTED_PROFILES:
            raise ManifestError(f"unsupported profile: {profile}")
        canonical_manifest = canonical_relative_path(manifest_path)
        if manifest_sha256 is not None and not _SHA256.fullmatch(manifest_sha256):
            raise ManifestError("detached manifest SHA-256 must be 64 lowercase hexadecimal characters")
        with open_source(artifact, limits) as source:
            manifest_bytes = source.read_bytes(canonical_manifest, max_bytes=1024 * 1024)
            observed_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            manifest = parse_manifest(manifest_bytes, manifest_path=canonical_manifest)
            artifact_name = manifest.name
            artifact_version = manifest.version
            findings.append(verify_manifest_anchor(manifest_bytes, manifest_sha256, require_trust_anchor))
            findings.append(verify_coverage(source, manifest, canonical_manifest))
            findings.extend(verify_file_hashes(source, manifest, profile))
            for check in manifest.checks:
                findings.append(run_declared_check(source, check, profile))
    except ArtifactProofError as exc:
        findings.append(
            Finding(
                "artifact-contract",
                "contract",
                Status.FAIL,
                f"{type(exc).__name__}: {exc}",
            )
        )
    return Report(
        artifact=str(artifact),
        profile=profile,
        manifest_sha256=observed_manifest_sha256,
        artifact_name=artifact_name,
        artifact_version=artifact_version,
        findings=tuple(findings),
    )
