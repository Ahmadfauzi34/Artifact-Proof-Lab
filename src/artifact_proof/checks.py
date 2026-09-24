from __future__ import annotations

from decimal import Decimal, DecimalException
from io import BytesIO
import hashlib
import json
import platform
import sqlite3
import stat
import sys
import tempfile
from pathlib import Path
import zipfile

from .errors import SourceError
from .manifest import CheckSpec, Manifest
from .model import Finding, Status
from .paths import canonical_relative_path
from .source import ArtifactSource


def verify_manifest_anchor(
    manifest_bytes: bytes,
    expected_sha256: str | None,
    require_anchor: bool,
) -> Finding:
    observed = hashlib.sha256(manifest_bytes).hexdigest()
    if expected_sha256 is None:
        return Finding(
            "manifest-anchor",
            "trust_anchor",
            Status.FAIL if require_anchor else Status.WARN,
            "no detached manifest SHA-256 was supplied",
            observed=observed,
        )
    if expected_sha256 == observed:
        return Finding(
            "manifest-anchor",
            "trust_anchor",
            Status.PASS,
            "detached manifest SHA-256 matches",
            expected=expected_sha256,
            observed=observed,
        )
    return Finding(
        "manifest-anchor",
        "trust_anchor",
        Status.FAIL,
        "detached manifest SHA-256 mismatch",
        expected=expected_sha256,
        observed=observed,
    )


def verify_coverage(
    source: ArtifactSource,
    manifest: Manifest,
    manifest_path: str,
) -> Finding:
    actual = set(source.list_files())
    expected = set(manifest.files)
    missing = sorted(expected - actual)
    unlisted = sorted(actual - expected - {manifest_path} - set(manifest.allow_unlisted))
    failed = bool(missing or (manifest.complete_coverage and unlisted))
    message = "artifact file coverage is complete" if not failed else "artifact file coverage mismatch"
    return Finding(
        "file-coverage",
        "coverage",
        Status.FAIL if failed else Status.PASS,
        message,
        expected={"declared": len(expected), "complete": manifest.complete_coverage},
        observed={"actual": len(actual), "missing": missing, "unlisted": unlisted},
    )


def verify_file_hashes(source: ArtifactSource, manifest: Manifest, profile: str) -> list[Finding]:
    actual = set(source.list_files())
    findings: list[Finding] = []
    for path, spec in sorted(manifest.files.items()):
        check_id = f"sha256:{path}"
        if path not in actual:
            findings.append(Finding(check_id, "sha256", Status.FAIL, "declared file is missing", expected=spec.sha256))
            continue
        observed = source.sha256(path)
        if profile == "live" and spec.mutable:
            findings.append(
                Finding(
                    check_id,
                    "sha256",
                    Status.SKIP,
                    "mutable file hash is not frozen in live profile",
                    expected=spec.sha256,
                    observed=observed,
                )
            )
        elif observed == spec.sha256:
            findings.append(Finding(check_id, "sha256", Status.PASS, "file SHA-256 matches", expected=spec.sha256, observed=observed))
        else:
            findings.append(Finding(check_id, "sha256", Status.FAIL, "file SHA-256 mismatch", expected=spec.sha256, observed=observed))
    return findings


def run_declared_check(source: ArtifactSource, check: CheckSpec, profile: str) -> Finding:
    if profile not in check.profiles:
        return Finding(check.check_id, check.check_type, Status.SKIP, f"check is inactive for {profile} profile")
    if check.check_type == "sqlite_integrity":
        return _sqlite_integrity(source, check)
    if check.check_type == "zip_member_matches":
        return _zip_member_matches(source, check)
    if check.check_type == "environment":
        return _environment(check)
    if check.check_type == "value_binding":
        return _value_binding(source, check)
    raise AssertionError(f"unhandled check type: {check.check_type}")


def _sqlite_integrity(source: ArtifactSource, check: CheckSpec) -> Finding:
    path = check.config["path"]
    mode = check.config["mode"]
    try:
        data = source.read_bytes(path)
        with tempfile.NamedTemporaryFile(prefix="artifact-proof-", suffix=".db") as handle:
            handle.write(data)
            handle.flush()
            uri = f"file:{Path(handle.name).resolve()}?mode=ro&immutable=1"
            connection = sqlite3.connect(uri, uri=True)
            try:
                pragma = "quick_check" if mode == "quick" else "integrity_check"
                rows = [row[0] for row in connection.execute(f"PRAGMA {pragma}")]
            finally:
                connection.close()
        passed = rows == ["ok"]
        return Finding(
            check.check_id,
            check.check_type,
            Status.PASS if passed else Status.FAIL,
            f"SQLite {mode} integrity {'passed' if passed else 'failed'}",
            expected=["ok"],
            observed=rows,
        )
    except (sqlite3.DatabaseError, SourceError) as exc:
        return Finding(check.check_id, check.check_type, Status.FAIL, f"SQLite inspection failed: {type(exc).__name__}: {exc}")


def _zip_member_matches(source: ArtifactSource, check: CheckSpec) -> Finding:
    archive_path = check.config["archive"]
    member_path = check.config["member"]
    target_path = check.config["target"]
    only_member = check.config["only_member"]
    try:
        archive_bytes = source.read_bytes(archive_path, max_bytes=64 * 1024 * 1024)
        target_digest = source.sha256(target_path)
        with zipfile.ZipFile(BytesIO(archive_bytes), "r") as nested:
            files: dict[str, zipfile.ZipInfo] = {}
            for info in nested.infolist():
                raw = info.filename[:-1] if info.is_dir() and info.filename.endswith("/") else info.filename
                canonical_relative_path(raw)
                if info.is_dir():
                    continue
                if raw in files:
                    raise SourceError(f"duplicate nested ZIP member: {raw}")
                file_type = stat.S_IFMT(info.external_attr >> 16)
                if file_type == stat.S_IFLNK:
                    raise SourceError(f"nested ZIP symlink is forbidden: {raw}")
                if file_type not in {0, stat.S_IFREG}:
                    raise SourceError(f"non-regular nested ZIP member is forbidden: {raw}")
                if info.flag_bits & 0x1:
                    raise SourceError(f"encrypted nested ZIP member is unsupported: {raw}")
                if info.file_size > 64 * 1024 * 1024:
                    raise SourceError(f"nested ZIP member exceeds size limit: {raw}")
                files[raw] = info
            if member_path not in files:
                raise SourceError(f"nested ZIP member is missing: {member_path}")
            if only_member and set(files) != {member_path}:
                raise SourceError(f"nested ZIP must contain only {member_path}")
            with nested.open(files[member_path]) as member:
                data = member.read(64 * 1024 * 1024 + 1)
            if len(data) > 64 * 1024 * 1024:
                raise SourceError("nested ZIP member exceeds read limit")
            observed = hashlib.sha256(data).hexdigest()
        passed = observed == target_digest
        return Finding(
            check.check_id,
            check.check_type,
            Status.PASS if passed else Status.FAIL,
            "nested ZIP member matches target" if passed else "nested ZIP member differs from target",
            expected=target_digest,
            observed=observed,
        )
    except (zipfile.BadZipFile, SourceError, KeyError, RuntimeError) as exc:
        return Finding(check.check_id, check.check_type, Status.FAIL, f"nested ZIP inspection failed: {type(exc).__name__}: {exc}")


def _environment(check: CheckSpec) -> Finding:
    expected = {
        "python_implementation": check.config["python_implementation"],
        "python_major_minor": list(check.config["python_major_minor"]),
        "platform_system": check.config["platform_system"],
    }
    observed = {
        "python_implementation": sys.implementation.name,
        "python_major_minor": list(sys.version_info[:2]),
        "platform_system": platform.system(),
    }
    machine = check.config["platform_machine"]
    if machine is not None:
        expected["platform_machine"] = machine
        observed["platform_machine"] = platform.machine()
    matched = expected == observed
    if matched:
        status = Status.PASS
        message = "execution environment matches reference"
    elif check.config["enforce"]:
        status = Status.FAIL
        message = "execution environment violates enforced contract"
    else:
        status = Status.WARN
        message = "execution environment differs from non-binding reference"
    return Finding(check.check_id, check.check_type, status, message, expected=expected, observed=observed)


def _value_binding(source: ArtifactSource, check: CheckSpec) -> Finding:
    try:
        bound: list[tuple[str, bytes]] = []
        documents: dict[str, object] = {}
        for operand in check.config["operands"]:
            kind = operand["kind"]
            path = operand["path"]
            if kind == "file_sha256":
                label = f"file_sha256:{path}"
                value: object = source.sha256(path)
            elif kind == "json_pointer":
                pointer = operand["pointer"]
                label = f"json_pointer:{path}#{pointer}"
                if path not in documents:
                    documents[path] = _read_strict_json(source, path)
                value = _resolve_json_pointer(documents[path], pointer)
            else:
                raise SourceError(f"unsupported value-binding operand kind: {kind}")
            canonical = _canonical_json_value(value)
            bound.append((label, canonical))

        reference = bound[0][1]
        passed = all(value == reference for _, value in bound[1:])
        observed = [
            {
                "operand": label,
                "value_sha256": hashlib.sha256(value).hexdigest(),
            }
            for label, value in bound
        ]
        return Finding(
            check.check_id,
            check.check_type,
            Status.PASS if passed else Status.FAIL,
            "all bound values are semantically identical" if passed else "bound values differ",
            expected={"operand": bound[0][0], "value_sha256": observed[0]["value_sha256"]},
            observed=observed,
        )
    except (SourceError, UnicodeDecodeError, json.JSONDecodeError, DecimalException, RecursionError, ValueError) as exc:
        return Finding(
            check.check_id,
            check.check_type,
            Status.FAIL,
            f"value binding inspection failed: {type(exc).__name__}: {exc}",
        )


def _read_strict_json(source: ArtifactSource, path: str) -> object:
    data = source.read_bytes(path)
    text = data.decode("utf-8")
    return json.loads(
        text,
        object_pairs_hook=_unique_json_object,
        parse_int=Decimal,
        parse_float=Decimal,
        parse_constant=_reject_json_constant,
    )


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(text: str) -> object:
    raise ValueError(f"non-standard JSON constant is forbidden: {text}")


def _resolve_json_pointer(document: object, pointer: str) -> object:
    if pointer == "":
        return document
    current = document
    for raw_token in pointer.split("/")[1:]:
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if token not in current:
                raise ValueError(f"JSON Pointer member is missing: {pointer}")
            current = current[token]
            continue
        if isinstance(current, list):
            if token == "-" or not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                raise ValueError(f"JSON Pointer array index is invalid: {pointer}")
            index = int(token)
            if index >= len(current):
                raise ValueError(f"JSON Pointer array index is out of range: {pointer}")
            current = current[index]
            continue
        raise ValueError(f"JSON Pointer descends through a scalar: {pointer}")
    return current


def _canonical_json_value(value: object) -> bytes:
    normalized = _normalize_json_value(value)
    return json.dumps(
        normalized,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _normalize_json_value(value: object) -> object:
    if value is None:
        return ["null"]
    if type(value) is bool:
        return ["bool", value]
    if isinstance(value, Decimal):
        return ["number", *_canonical_decimal(value)]
    if isinstance(value, str):
        return ["string", value]
    if isinstance(value, list):
        return ["array", [_normalize_json_value(item) for item in value]]
    if isinstance(value, dict):
        return [
            "object",
            [[key, _normalize_json_value(value[key])] for key in sorted(value)],
        ]
    raise ValueError(f"unsupported JSON value type: {type(value).__name__}")


def _canonical_decimal(value: Decimal) -> tuple[int, str, int]:
    sign, raw_digits, exponent = value.as_tuple()
    digits = list(raw_digits)
    if not any(digits):
        return 0, "0", 0
    while digits and digits[-1] == 0:
        digits.pop()
        exponent += 1
    return sign, "".join(str(digit) for digit in digits), exponent
