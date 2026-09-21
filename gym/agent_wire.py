from __future__ import annotations

import json
import math
from typing import Any, Mapping

from .core import AgentTaskView, DecisionKind, Observation, PolicyDecision
from .host_wire import (
    HostProtocolError,
    agent_task_from_wire as _host_agent_task_from_wire,
    agent_task_to_wire,
    observation_from_wire as _host_observation_from_wire,
    observation_to_wire,
)


AGENT_PROTOCOL = "proof-gym-agent-jsonl-v1"
DEFAULT_MAX_MESSAGE_BYTES = 1024 * 1024


class AgentProtocolError(RuntimeError):
    """Fail-closed agent endpoint protocol violation."""


class AgentProtocolTimeout(AgentProtocolError):
    """The agent endpoint did not answer within the configured deadline."""


class AgentProtocolClosed(AgentProtocolError):
    """The agent endpoint transport closed before a valid response arrived."""


def encode_message(
    message: Mapping[str, Any],
    *,
    max_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
) -> bytes:
    try:
        encoded = json.dumps(
            dict(message),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AgentProtocolError(f"message is not strict JSON: {exc}") from exc
    if len(encoded) > max_bytes:
        raise AgentProtocolError(
            f"message exceeds max size: {len(encoded)} > {max_bytes}"
        )
    if b"\n" in encoded or b"\r" in encoded:
        raise AgentProtocolError("encoded JSONL frame contains a raw newline")
    return encoded


def decode_message(
    raw: bytes,
    *,
    max_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
) -> dict[str, Any]:
    if not raw:
        raise AgentProtocolClosed("empty agent protocol frame")
    if len(raw) > max_bytes + 1:
        raise AgentProtocolError(
            f"message exceeds max size: {len(raw)} > {max_bytes + 1}"
        )
    if raw.endswith(b"\n"):
        raw = raw[:-1]
    if raw.endswith(b"\r"):
        raw = raw[:-1]
    if len(raw) > max_bytes:
        raise AgentProtocolError(
            f"message exceeds max size: {len(raw)} > {max_bytes}"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AgentProtocolError("agent protocol frame is not UTF-8") from exc

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
        raise AgentProtocolError(f"invalid strict JSON frame: {exc}") from exc
    if not isinstance(value, dict):
        raise AgentProtocolError("agent protocol frame must be a JSON object")
    _reject_nonfinite_numbers(value)
    if value.get("protocol") != AGENT_PROTOCOL:
        raise AgentProtocolError("agent protocol version mismatch")
    request_id = value.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise AgentProtocolError("agent protocol frame missing request_id")
    return value


def agent_task_from_wire(raw: Mapping[str, Any]) -> AgentTaskView:
    try:
        return _host_agent_task_from_wire(raw)
    except HostProtocolError as exc:
        raise AgentProtocolError(str(exc)) from exc


def observation_from_wire(raw: Mapping[str, Any]) -> Observation:
    try:
        return _host_observation_from_wire(raw)
    except HostProtocolError as exc:
        raise AgentProtocolError(str(exc)) from exc


def decision_to_wire(decision: PolicyDecision) -> dict[str, Any]:
    _validate_decision(decision)
    return {
        "kind": decision.kind.value,
        "action": decision.action,
    }


def decision_from_wire(raw: Mapping[str, Any]) -> PolicyDecision:
    require_exact_fields(raw, {"kind", "action"}, field="agent decision")
    kind_raw = raw.get("kind")
    if not isinstance(kind_raw, str):
        raise AgentProtocolError("agent decision kind must be a string")
    try:
        kind = DecisionKind(kind_raw)
    except ValueError as exc:
        raise AgentProtocolError(f"unsupported decision kind: {kind_raw}") from exc
    action = raw.get("action")
    if action is not None and not isinstance(action, str):
        raise AgentProtocolError("agent decision action must be string or null")
    decision = PolicyDecision(kind=kind, action=action, rationale="")
    _validate_decision(decision)
    return decision


def observations_to_wire(
    observations: tuple[Observation, ...],
) -> list[dict[str, Any]]:
    return [observation_to_wire(item) for item in observations]


def observations_from_wire(value: Any) -> tuple[Observation, ...]:
    if not isinstance(value, list):
        raise AgentProtocolError("agent observations must be a list")
    parsed = []
    for item in value:
        if not isinstance(item, dict):
            raise AgentProtocolError("agent observation must be an object")
        parsed.append(observation_from_wire(item))
    return tuple(parsed)


def trajectory_from_wire(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise AgentProtocolError("agent trajectory must be a string list")
    return tuple(value)


def strict_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AgentProtocolError(f"{field} must be an object")
    return dict(value)


def require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    field: str,
) -> None:
    keys = set(value)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise AgentProtocolError(
            f"{field} fields mismatch; missing={missing}, extra={extra}"
        )


def required_string(value: Mapping[str, Any], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item:
        raise AgentProtocolError(f"{name} must be a non-empty string")
    return item


def required_turn(value: Mapping[str, Any]) -> int:
    item = value.get("turn")
    if isinstance(item, bool) or not isinstance(item, int) or item < 1:
        raise AgentProtocolError("turn must be a positive integer")
    return item


def required_attempt(value: Mapping[str, Any]) -> int:
    item = value.get("attempt")
    if isinstance(item, bool) or not isinstance(item, int) or item < 1:
        raise AgentProtocolError("attempt must be a positive integer")
    return item


def _validate_decision(decision: PolicyDecision) -> None:
    if decision.kind is DecisionKind.SELECT:
        if not isinstance(decision.action, str) or not decision.action:
            raise AgentProtocolError("SELECT requires a non-empty action")
        return
    if decision.action is not None:
        raise AgentProtocolError(
            f"{decision.kind.value} must not carry an action"
        )


def _reject_nonfinite_numbers(value: Any) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AgentProtocolError(
                "agent protocol frame contains non-finite number"
            )
        return
    if isinstance(value, list):
        for item in value:
            _reject_nonfinite_numbers(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _reject_nonfinite_numbers(item)
        return
    raise AgentProtocolError(
        "agent protocol frame contains unsupported JSON value"
    )
