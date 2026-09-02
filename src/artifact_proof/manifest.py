from __future__ import annotations

from dataclasses import dataclass
import json
import re
from types import MappingProxyType
from typing import Any, Mapping

from .errors import ManifestError, SourceError
from .paths import canonical_relative_path


FORMAT = "artifact-proof-manifest-v1"
SUPPORTED_PROFILES = frozenset({"sealed", "live"})
SUPPORTED_CHECKS = frozenset({"sqlite_integrity", "zip_member_matches", "environment"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FileSpec:
    sha256: str
    mutable: bool = False


@dataclass(frozen=True)
class CheckSpec:
    check_id: str
    check_type: str
    profiles: frozenset[str]
    config: Mapping[str, Any]


@dataclass(frozen=True)
class Manifest:
    name: str
    version: str
    files: Mapping[str, FileSpec]
    checks: tuple[CheckSpec, ...]
    complete_coverage: bool
    allow_unlisted: frozenset[str]


def parse_manifest(data: bytes, *, manifest_path: str) -> Manifest:
    if len(data) > 1024 * 1024:
        raise ManifestError("manifest exceeds 1 MiB")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestError("manifest must be UTF-8") from exc
    try:
        raw = json.loads(text, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, ManifestError) as exc:
        raise ManifestError(f"invalid manifest JSON: {exc}") from exc
    root = _object(raw, "manifest")
    _keys(root, {"format", "artifact", "files", "checks", "coverage"}, "manifest")
    if root.get("format") != FORMAT:
        raise ManifestError(f"unsupported manifest format: {root.get('format')!r}")

    artifact = _object(root.get("artifact"), "artifact")
    _keys(artifact, {"name", "version"}, "artifact")
    name = _nonempty(artifact.get("name"), "artifact.name")
    version = _nonempty(artifact.get("version"), "artifact.version")

    files_raw = _object(root.get("files"), "files")
    files: dict[str, FileSpec] = {}
    for raw_path, raw_spec in files_raw.items():
        path = _path(raw_path, f"files[{raw_path!r}]")
        if path == manifest_path:
            raise ManifestError("manifest cannot hash itself; use a detached manifest SHA-256")
        spec = _object(raw_spec, f"files[{path!r}]")
        _keys(spec, {"sha256", "mutable"}, f"files[{path!r}]")
        digest = spec.get("sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ManifestError(f"invalid SHA-256 for {path}")
        mutable = spec.get("mutable", False)
        if not isinstance(mutable, bool):
            raise ManifestError(f"mutable must be boolean for {path}")
        files[path] = FileSpec(digest, mutable)

    coverage = _object(root.get("coverage", {}), "coverage")
    _keys(coverage, {"complete", "allow_unlisted"}, "coverage")
    complete = coverage.get("complete", True)
    if not isinstance(complete, bool):
        raise ManifestError("coverage.complete must be boolean")
    allow_raw = coverage.get("allow_unlisted", [])
    if not isinstance(allow_raw, list) or not all(isinstance(item, str) for item in allow_raw):
        raise ManifestError("coverage.allow_unlisted must be a string list")
    allow = frozenset(_path(item, "coverage.allow_unlisted") for item in allow_raw)

    checks_raw = root.get("checks", [])
    if not isinstance(checks_raw, list):
        raise ManifestError("checks must be a list")
    checks: list[CheckSpec] = []
    ids: set[str] = set()
    for index, value in enumerate(checks_raw):
        check = _parse_check(value, index, files)
        if check.check_id in ids:
            raise ManifestError(f"duplicate check id: {check.check_id}")
        ids.add(check.check_id)
        checks.append(check)
    return Manifest(
        name=name,
        version=version,
        files=MappingProxyType(files),
        checks=tuple(checks),
        complete_coverage=complete,
        allow_unlisted=allow,
    )


def _parse_check(raw: Any, index: int, files: Mapping[str, FileSpec]) -> CheckSpec:
    context = f"checks[{index}]"
    value = _object(raw, context)
    check_id = _nonempty(value.get("id"), f"{context}.id")
    check_type = _nonempty(value.get("type"), f"{context}.type")
    if check_type not in SUPPORTED_CHECKS:
        raise ManifestError(f"unsupported check type: {check_type}")
    profiles_raw = value.get("profiles", ["sealed", "live"])
    if not isinstance(profiles_raw, list) or not profiles_raw:
        raise ManifestError(f"{context}.profiles must be a non-empty list")
    profiles = frozenset(profiles_raw)
    if not all(isinstance(item, str) for item in profiles) or not profiles <= SUPPORTED_PROFILES:
        raise ManifestError(f"invalid profiles in {context}")

    common = {"id", "type", "profiles"}
    config: dict[str, Any]
    if check_type == "sqlite_integrity":
        _keys(value, common | {"path", "mode"}, context)
        path = _declared_path(value.get("path"), files, f"{context}.path")
        mode = value.get("mode", "full")
        if mode not in {"quick", "full"}:
            raise ManifestError(f"invalid SQLite integrity mode in {context}")
        config = {"path": path, "mode": mode}
    elif check_type == "zip_member_matches":
        _keys(value, common | {"archive", "member", "target", "only_member"}, context)
        archive = _declared_path(value.get("archive"), files, f"{context}.archive")
        target = _declared_path(value.get("target"), files, f"{context}.target")
        member = _path(value.get("member"), f"{context}.member")
        only_member = value.get("only_member", False)
        if not isinstance(only_member, bool):
            raise ManifestError(f"{context}.only_member must be boolean")
        config = {"archive": archive, "member": member, "target": target, "only_member": only_member}
    else:
        _keys(
            value,
            common | {"python_implementation", "python_major_minor", "platform_system", "platform_machine", "enforce"},
            context,
        )
        implementation = _nonempty(value.get("python_implementation"), f"{context}.python_implementation")
        version = value.get("python_major_minor")
        if not isinstance(version, list) or len(version) != 2 or not all(type(part) is int and part >= 0 for part in version):
            raise ManifestError(f"{context}.python_major_minor must contain two integers")
        system = _nonempty(value.get("platform_system"), f"{context}.platform_system")
        machine = value.get("platform_machine")
        if machine is not None:
            machine = _nonempty(machine, f"{context}.platform_machine")
        enforce = value.get("enforce", False)
        if not isinstance(enforce, bool):
            raise ManifestError(f"{context}.enforce must be boolean")
        config = {
            "python_implementation": implementation,
            "python_major_minor": tuple(version),
            "platform_system": system,
            "platform_machine": machine,
            "enforce": enforce,
        }
    return CheckSpec(check_id, check_type, profiles, MappingProxyType(config))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{context} must be an object")
    return value


def _keys(value: Mapping[str, Any], allowed: set[str], context: str) -> None:
    extra = set(value) - allowed
    if extra:
        raise ManifestError(f"unknown keys in {context}: {sorted(extra)}")


def _nonempty(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{context} must be a non-empty string")
    return value


def _path(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise ManifestError(f"{context} must be a string path")
    try:
        return canonical_relative_path(value)
    except SourceError as exc:
        raise ManifestError(f"invalid path in {context}: {exc}") from exc


def _declared_path(value: Any, files: Mapping[str, FileSpec], context: str) -> str:
    path = _path(value, context)
    if path not in files:
        raise ManifestError(f"{context} references undeclared file: {path}")
    return path
