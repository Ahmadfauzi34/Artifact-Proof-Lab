from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .core import HostTaskDescriptor
from .ledger import sha256_json


PACK_FORMAT = "proof-gym-private-holdout-pack-v1"
SNAPSHOT_FORMAT = "proof-gym-private-snapshot-v1"
RUNTIME_DESCRIPTOR_FORMAT = "proof-gym-private-runtime-descriptor-v1"
PACK_COMMITMENT_FORMAT = "proof-gym-private-pack-commitment-v1"
PRIVATE_SPLITS = frozenset({"holdout", "external_real"})

_RESERVED_METADATA = frozenset(
    {
        "private_holdout_verified",
        "private_pack_id",
        "private_task_file_sha256",
        "private_snapshot_sha256",
        "private_runtime_descriptor_sha256",
    }
)


class PrivatePackError(RuntimeError):
    """Fail-closed private holdout pack validation error."""


@dataclass(frozen=True)
class PrivatePackTask:
    task: HostTaskDescriptor
    task_file_sha256: str
    snapshot_sha256: str
    descriptor_sha256: str


@dataclass(frozen=True)
class PrivateHoldoutPack:
    root: Path
    pack_id: str
    native_competence_claim: bool
    manifest_sha256: str
    runtime_id: str
    runtime_descriptor_sha256: str
    tasks: Mapping[str, PrivatePackTask]
    commitment_sha256: str

    @classmethod
    def load(
        cls,
        root: Path | str,
        *,
        expected_runtime_id: str | None = None,
    ) -> "PrivateHoldoutPack":
        base = Path(root)
        if not base.exists() or not base.is_dir():
            raise PrivatePackError("private pack root must be an existing directory")
        if base.is_symlink():
            raise PrivatePackError("private pack root must not be a symlink")
        base = base.resolve()

        manifest_path = base / "PRIVATE_HOLDOUT_PACK.json"
        if not manifest_path.exists() or not manifest_path.is_file() or manifest_path.is_symlink():
            raise PrivatePackError("PRIVATE_HOLDOUT_PACK.json is missing or unsafe")
        manifest_bytes = manifest_path.read_bytes()
        manifest = _strict_json_object(manifest_bytes, field="private pack manifest")
        _require_exact_fields(
            manifest,
            {"format", "pack_id", "native_competence_claim", "runtime", "tasks"},
            field="private pack manifest",
        )
        if manifest.get("format") != PACK_FORMAT:
            raise PrivatePackError("unsupported private pack format")
        pack_id = _required_string(manifest, "pack_id")
        native_claim = manifest.get("native_competence_claim")
        if native_claim is not False:
            raise PrivatePackError(
                "private pack cannot self-assert native competence"
            )

        runtime_raw = _required_mapping(manifest, "runtime")
        _require_exact_fields(
            runtime_raw,
            {"runtime_id", "path", "sha256"},
            field="private pack runtime",
        )
        runtime_id = _required_string(runtime_raw, "runtime_id")
        if expected_runtime_id is not None and runtime_id != expected_runtime_id:
            raise PrivatePackError(
                f"private pack runtime mismatch: {runtime_id!r} != {expected_runtime_id!r}"
            )
        runtime_rel = _required_string(runtime_raw, "path")
        runtime_sha = _required_sha256(runtime_raw, "sha256")
        runtime_path = _safe_file(base, runtime_rel)
        runtime_bytes = runtime_path.read_bytes()
        _require_digest(runtime_bytes, runtime_sha, field="runtime descriptor")
        runtime_doc = _strict_json_object(runtime_bytes, field="runtime descriptor")
        _require_exact_fields(
            runtime_doc,
            {"format", "runtime_id", "environment_contract", "attestation_scope"},
            field="runtime descriptor",
        )
        if runtime_doc.get("format") != RUNTIME_DESCRIPTOR_FORMAT:
            raise PrivatePackError("unsupported runtime descriptor format")
        if runtime_doc.get("runtime_id") != runtime_id:
            raise PrivatePackError("runtime descriptor id mismatch")
        if runtime_doc.get("attestation_scope") != "reference_identity_only":
            raise PrivatePackError("reference pack runtime descriptor scope is invalid")

        tasks_raw = manifest.get("tasks")
        if not isinstance(tasks_raw, list) or not tasks_raw:
            raise PrivatePackError("private pack tasks must be a non-empty list")

        declared_files = {
            "PRIVATE_HOLDOUT_PACK.json",
            _normalize_rel(runtime_rel),
        }
        loaded: dict[str, PrivatePackTask] = {}
        for index, item in enumerate(tasks_raw):
            if not isinstance(item, dict):
                raise PrivatePackError(f"private pack task entry {index} must be an object")
            _require_exact_fields(
                item,
                {"task_id", "path", "sha256", "snapshot"},
                field=f"private pack task entry {index}",
            )
            task_id = _required_string(item, "task_id")
            if task_id in loaded:
                raise PrivatePackError(f"duplicate private pack task_id: {task_id}")

            task_rel = _required_string(item, "path")
            task_sha = _required_sha256(item, "sha256")
            task_path = _safe_file(base, task_rel)
            task_bytes = task_path.read_bytes()
            _require_digest(task_bytes, task_sha, field=f"task {task_id}")
            raw_task = _strict_json_object(task_bytes, field=f"task {task_id}")
            try:
                task = HostTaskDescriptor.from_dict(raw_task)
            except (KeyError, TypeError, ValueError) as exc:
                raise PrivatePackError(f"task {task_id} descriptor is invalid: {exc}") from exc
            if task.task_id != task_id:
                raise PrivatePackError(f"task id mismatch for {task_id}")
            if task.split not in PRIVATE_SPLITS:
                raise PrivatePackError(
                    f"private pack task {task_id} has non-private split: {task.split}"
                )
            if (
                "native_competence_claim" in task.metadata
                and task.metadata["native_competence_claim"] is not False
            ):
                raise PrivatePackError(
                    f"task {task_id} cannot self-assert native competence"
                )
            reserved = _RESERVED_METADATA.intersection(task.metadata)
            if reserved:
                raise PrivatePackError(
                    f"task {task_id} predefines reserved private metadata: {sorted(reserved)}"
                )
            if "private_snapshot" in task.environment:
                raise PrivatePackError(f"task {task_id} predefines private_snapshot")

            snapshot_raw = _required_mapping(item, "snapshot")
            _require_exact_fields(
                snapshot_raw,
                {"snapshot_id", "path", "sha256"},
                field=f"task {task_id} snapshot entry",
            )
            snapshot_id = _required_string(snapshot_raw, "snapshot_id")
            snapshot_rel = _required_string(snapshot_raw, "path")
            snapshot_sha = _required_sha256(snapshot_raw, "sha256")
            if task.environment.get("snapshot_id") != snapshot_id:
                raise PrivatePackError(f"task {task_id} snapshot id does not match descriptor")

            snapshot_path = _safe_file(base, snapshot_rel)
            snapshot_bytes = snapshot_path.read_bytes()
            _require_digest(snapshot_bytes, snapshot_sha, field=f"snapshot {snapshot_id}")
            snapshot_doc = _strict_json_object(
                snapshot_bytes,
                field=f"snapshot {snapshot_id}",
            )
            _validate_snapshot(snapshot_doc, task_id=task_id, domain=task.domain, snapshot_id=snapshot_id)

            environment = dict(task.environment)
            environment["private_snapshot"] = snapshot_doc
            metadata = dict(task.metadata)
            metadata.update(
                {
                    "private_holdout_verified": True,
                    "private_pack_id": pack_id,
                    "private_task_file_sha256": task_sha,
                    "private_snapshot_sha256": snapshot_sha,
                    "private_runtime_descriptor_sha256": runtime_sha,
                }
            )
            sealed_task = replace(task, environment=environment, metadata=metadata)
            descriptor_sha = sha256_json(asdict(sealed_task))
            loaded[task_id] = PrivatePackTask(
                task=sealed_task,
                task_file_sha256=task_sha,
                snapshot_sha256=snapshot_sha,
                descriptor_sha256=descriptor_sha,
            )
            declared_files.add(_normalize_rel(task_rel))
            declared_files.add(_normalize_rel(snapshot_rel))

        _verify_complete_file_coverage(base, declared_files)

        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        commitment = sha256_json(
            {
                "format": PACK_COMMITMENT_FORMAT,
                "pack_id": pack_id,
                "manifest_sha256": manifest_sha,
                "runtime_id": runtime_id,
                "runtime_descriptor_sha256": runtime_sha,
                "tasks": {
                    task_id: loaded[task_id].descriptor_sha256
                    for task_id in sorted(loaded)
                },
            }
        )
        return cls(
            root=base,
            pack_id=pack_id,
            native_competence_claim=native_claim,
            manifest_sha256=manifest_sha,
            runtime_id=runtime_id,
            runtime_descriptor_sha256=runtime_sha,
            tasks=dict(loaded),
            commitment_sha256=commitment,
        )

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.tasks))

    def task(self, task_id: str) -> HostTaskDescriptor:
        try:
            return self.tasks[task_id].task
        except KeyError as exc:
            raise PrivatePackError(f"unknown private pack task: {task_id}") from exc

    def task_commitment(self, task_id: str) -> str:
        try:
            return self.tasks[task_id].descriptor_sha256
        except KeyError as exc:
            raise PrivatePackError(f"unknown private pack task: {task_id}") from exc


def _validate_snapshot(
    snapshot: Mapping[str, Any],
    *,
    task_id: str,
    domain: str,
    snapshot_id: str,
) -> None:
    _require_exact_fields(
        snapshot,
        {"format", "snapshot_id", "domain", "state", "oracle"},
        field=f"snapshot {snapshot_id}",
    )
    if snapshot.get("format") != SNAPSHOT_FORMAT:
        raise PrivatePackError(f"snapshot {snapshot_id} has unsupported format")
    if snapshot.get("snapshot_id") != snapshot_id:
        raise PrivatePackError(f"snapshot {snapshot_id} identity mismatch")
    if snapshot.get("domain") != domain:
        raise PrivatePackError(f"snapshot {snapshot_id} domain mismatch for task {task_id}")
    if not isinstance(snapshot.get("state"), dict):
        raise PrivatePackError(f"snapshot {snapshot_id} state must be an object")
    if not isinstance(snapshot.get("oracle"), dict):
        raise PrivatePackError(f"snapshot {snapshot_id} oracle must be an object")


def _safe_file(root: Path, relative: str) -> Path:
    normalized = _normalize_rel(relative)
    parts = PurePosixPath(normalized).parts
    candidate = root.joinpath(*parts)

    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise PrivatePackError(f"private pack path uses symlink: {relative}")

    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PrivatePackError(f"private pack file is missing: {relative}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PrivatePackError(f"private pack path escapes root: {relative}") from exc
    if not resolved.is_file():
        raise PrivatePackError(f"private pack path is not a regular file: {relative}")
    return resolved


def _normalize_rel(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise PrivatePackError("private pack path must be a non-empty string")
    if "\\" in value:
        raise PrivatePackError(f"private pack path must use POSIX separators: {value}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PrivatePackError(f"unsafe private pack path: {value}")
    normalized = path.as_posix()
    if normalized != value:
        raise PrivatePackError(f"private pack path is not canonical: {value}")
    return normalized


def _verify_complete_file_coverage(root: Path, declared: set[str]) -> None:
    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise PrivatePackError(
                f"private pack contains symlink: {path.relative_to(root).as_posix()}"
            )
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != declared:
        missing = sorted(declared - actual)
        unlisted = sorted(actual - declared)
        raise PrivatePackError(
            f"private pack file coverage mismatch; missing={missing}, unlisted={unlisted}"
        )


def _strict_json_object(data: bytes, *, field: str) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PrivatePackError(f"{field} is not UTF-8") from exc

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        raw = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise PrivatePackError(f"{field} is not strict JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise PrivatePackError(f"{field} must be a JSON object")
    _reject_nonfinite_numbers(raw, field=field)
    return raw


def _reject_nonfinite_numbers(value: Any, *, field: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PrivatePackError(f"{field} contains non-finite number")
        return
    if isinstance(value, list):
        for item in value:
            _reject_nonfinite_numbers(item, field=field)
        return
    if isinstance(value, dict):
        for item in value.values():
            _reject_nonfinite_numbers(item, field=field)
        return
    raise PrivatePackError(f"{field} contains unsupported JSON value")


def _require_digest(data: bytes, expected: str, *, field: str) -> None:
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise PrivatePackError(f"{field} sha256 mismatch")


def _required_sha256(value: Mapping[str, Any], name: str) -> str:
    item = _required_string(value, name)
    if len(item) != 64 or any(ch not in "0123456789abcdef" for ch in item):
        raise PrivatePackError(f"{name} must be 64 lowercase hex characters")
    return item


def _required_string(value: Mapping[str, Any], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item:
        raise PrivatePackError(f"{name} must be a non-empty string")
    return item


def _required_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    item = value.get(name)
    if not isinstance(item, dict):
        raise PrivatePackError(f"{name} must be an object")
    return dict(item)


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    field: str,
) -> None:
    keys = set(value)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise PrivatePackError(
            f"{field} fields mismatch; missing={missing}, extra={extra}"
        )
