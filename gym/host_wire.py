from __future__ import annotations

import json
import math
from typing import Any, Mapping

from .core import AgentTaskView, Observation, StepOutcome


PROTOCOL = "proof-gym-host-jsonl-v1"
DEFAULT_MAX_MESSAGE_BYTES = 1024 * 1024


class HostProtocolError(RuntimeError):
    """Fail-closed wire/protocol violation."""


class HostProtocolTimeout(HostProtocolError):
    """The private host did not answer within the configured request timeout."""


class HostProtocolClosed(HostProtocolError):
    """The private host transport closed before a complete response arrived."""


def encode_message(message: Mapping[str, Any], *, max_bytes: int = DEFAULT_MAX_MESSAGE_BYTES) -> bytes:
    try:
        encoded = json.dumps(
            dict(message),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HostProtocolError(f"message is not strict JSON: {exc}") from exc
    if len(encoded) > max_bytes:
        raise HostProtocolError(f"message exceeds max size: {len(encoded)} > {max_bytes}")
    if b"\n" in encoded or b"\r" in encoded:
        raise HostProtocolError("encoded JSONL frame contains a raw newline")
    return encoded


def decode_message(raw: bytes, *, max_bytes: int = DEFAULT_MAX_MESSAGE_BYTES) -> dict[str, Any]:
    if not raw:
        raise HostProtocolClosed("empty host protocol frame")
    if len(raw) > max_bytes + 1:
        raise HostProtocolError(f"message exceeds max size: {len(raw)} > {max_bytes + 1}")
    if raw.endswith(b"\n"):
        raw = raw[:-1]
    if raw.endswith(b"\r"):
        raw = raw[:-1]
    if len(raw) > max_bytes:
        raise HostProtocolError(f"message exceeds max size: {len(raw)} > {max_bytes}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HostProtocolError("host protocol frame is not UTF-8") from exc

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def reject_duplicate_keys(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    try:
        value = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise HostProtocolError(f"invalid strict JSON frame: {exc}") from exc
    if not isinstance(value, dict):
        raise HostProtocolError("host protocol frame must be a JSON object")
    _reject_nonfinite_numbers(value)
    if value.get("protocol") != PROTOCOL:
        raise HostProtocolError("host protocol version mismatch")
    request_id = value.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise HostProtocolError("host protocol frame missing request_id")
    return value


def _reject_nonfinite_numbers(value: Any) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HostProtocolError("host protocol frame contains non-finite number")
        return
    if isinstance(value, list):
        for item in value:
            _reject_nonfinite_numbers(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _reject_nonfinite_numbers(item)
        return
    raise HostProtocolError("host protocol frame contains unsupported JSON value")


def agent_task_to_wire(task: AgentTaskView) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "domain": task.domain,
        "description": task.description,
        "allowed_actions": list(task.allowed_actions),
        "budget": dict(task.budget),
        "public_tests": list(task.public_tests),
        "public_inputs": list(task.public_inputs),
    }


def agent_task_from_wire(raw: Mapping[str, Any]) -> AgentTaskView:
    required = {
        "task_id",
        "domain",
        "description",
        "allowed_actions",
        "budget",
        "public_tests",
        "public_inputs",
    }
    _require_exact_fields(raw, required, field="agent task")
    task_id = _require_string(raw, "task_id")
    domain = _require_string(raw, "domain")
    description = _require_string(raw, "description")
    allowed_actions = _require_string_list(raw.get("allowed_actions"), field="allowed_actions")
    public_tests = _require_string_list(raw.get("public_tests"), field="public_tests")
    public_inputs = _require_string_list(raw.get("public_inputs"), field="public_inputs")
    budget = raw.get("budget")
    if not isinstance(budget, dict):
        raise HostProtocolError("agent task budget must be an object")
    return AgentTaskView(
        task_id=task_id,
        domain=domain,
        description=description,
        allowed_actions=allowed_actions,
        budget=dict(budget),
        public_tests=public_tests,
        public_inputs=public_inputs,
    )


def observation_to_wire(observation: Observation) -> dict[str, Any]:
    return observation.to_dict()


def observation_from_wire(raw: Mapping[str, Any]) -> Observation:
    required = {
        "observation_id",
        "source_lineage_root",
        "check_id",
        "probe_id",
        "attempt_id",
        "sequence_index",
        "logical_tick",
        "observation_type",
        "status",
        "value",
        "cost",
        "raw_payload_ref",
    }
    _require_exact_fields(raw, required, field="observation")
    attempt_id = _require_int(raw, "attempt_id")
    sequence_index = _require_int(raw, "sequence_index")
    logical_tick = _require_int(raw, "logical_tick")
    cost_raw = raw.get("cost")
    if isinstance(cost_raw, bool) or not isinstance(cost_raw, (int, float)):
        raise HostProtocolError("observation cost must be numeric")
    cost = float(cost_raw)
    if not math.isfinite(cost):
        raise HostProtocolError("observation cost must be finite")
    payload_ref = raw.get("raw_payload_ref")
    if payload_ref is not None and not isinstance(payload_ref, str):
        raise HostProtocolError("raw_payload_ref must be string or null")
    return Observation(
        observation_id=_require_string(raw, "observation_id"),
        source_lineage_root=_require_string(raw, "source_lineage_root"),
        check_id=_require_string(raw, "check_id"),
        probe_id=_require_string(raw, "probe_id"),
        attempt_id=attempt_id,
        sequence_index=sequence_index,
        logical_tick=logical_tick,
        observation_type=_require_string(raw, "observation_type"),
        status=_require_string(raw, "status"),
        value=raw["value"],
        cost=cost,
        raw_payload_ref=payload_ref,
    )


def outcome_to_wire(outcome: StepOutcome) -> dict[str, Any]:
    return {
        "observations": [observation_to_wire(item) for item in outcome.observations],
        "done": bool(outcome.done),
        "accepted_candidate": outcome.accepted_candidate,
    }


def outcome_from_wire(raw: Mapping[str, Any]) -> StepOutcome:
    _require_exact_fields(
        raw,
        {"observations", "done", "accepted_candidate"},
        field="step outcome",
    )
    observations = raw.get("observations")
    if not isinstance(observations, list):
        raise HostProtocolError("step outcome observations must be a list")
    parsed_observations = []
    for item in observations:
        if not isinstance(item, dict):
            raise HostProtocolError("step outcome observation must be an object")
        parsed_observations.append(observation_from_wire(item))
    accepted = raw.get("accepted_candidate")
    if accepted is not None and not isinstance(accepted, bool):
        raise HostProtocolError("accepted_candidate must be boolean or null")
    if not isinstance(raw.get("done"), bool):
        raise HostProtocolError("step outcome done must be boolean")
    return StepOutcome(
        observations=tuple(parsed_observations),
        done=raw["done"],
        accepted_candidate=accepted,
    )


def strict_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HostProtocolError(f"{field} must be an object")
    return dict(value)


def require_exact_fields(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    _require_exact_fields(value, expected, field=field)


def required_string(value: Mapping[str, Any], name: str) -> str:
    return _require_string(value, name)


def _require_exact_fields(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    keys = set(value)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise HostProtocolError(f"{field} fields mismatch; missing={missing}, extra={extra}")


def _require_string(value: Mapping[str, Any], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item:
        raise HostProtocolError(f"{name} must be a non-empty string")
    return item


def _require_int(value: Mapping[str, Any], name: str) -> int:
    item = value.get(name)
    if isinstance(item, bool) or not isinstance(item, int):
        raise HostProtocolError(f"{name} must be an integer")
    return item


def _require_string_list(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise HostProtocolError(f"{field} must be a string list")
    return tuple(value)
