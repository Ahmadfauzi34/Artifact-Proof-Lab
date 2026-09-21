from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import sys
from typing import Any, BinaryIO, Mapping

from .agent_wire import (
    AGENT_PROTOCOL,
    DEFAULT_MAX_MESSAGE_BYTES,
    AgentProtocolError,
    agent_task_from_wire,
    decode_message,
    decision_to_wire,
    encode_message,
    observations_from_wire,
    require_exact_fields,
    required_attempt,
    required_string,
    required_turn,
    strict_mapping,
    trajectory_from_wire,
)
from .core import AgentTaskView
from .ledger import sha256_json
from .reference_policy import ReferenceGenerator, ReferencePolicy


OPERATIONS = (
    "HELLO",
    "START",
    "DECIDE",
    "GENERATE",
    "CLOSE",
    "PING",
    "SHUTDOWN",
)


@dataclass
class _WorkerSession:
    task: AgentTaskView
    next_turn: int = 1


class ReferenceAgentWorker:
    """Inspectable protocol worker; not native competence evidence."""

    def __init__(self) -> None:
        self._policy = ReferencePolicy()
        self._generator = ReferenceGenerator()
        self._sessions: dict[str, _WorkerSession] = {}
        self._counter = 0
        self._shutdown = False

    @property
    def should_shutdown(self) -> bool:
        return self._shutdown

    def dispatch(
        self,
        op: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        if op == "HELLO":
            require_exact_fields(payload, set(), field="HELLO payload")
            return {
                "protocol": AGENT_PROTOCOL,
                "operations": list(OPERATIONS),
                "endpoint_role": "reference_agent_worker",
                "reasoning_surface": "typed_decision_and_candidate_only",
            }
        if op == "PING":
            require_exact_fields(payload, set(), field="PING payload")
            return {"status": "ok"}
        if op == "START":
            require_exact_fields(payload, {"task"}, field="START payload")
            task_raw = strict_mapping(
                payload.get("task"),
                field="agent START task",
            )
            task = agent_task_from_wire(task_raw)
            self._counter += 1
            task_sha = sha256_json(asdict(task))
            material = (
                f"{task.task_id}|{task_sha}|{self._counter}".encode("utf-8")
            )
            session_id = (
                "agent-session-"
                + hashlib.sha256(material).hexdigest()[:20]
            )
            if session_id in self._sessions:
                raise AgentProtocolError("agent worker session id collision")
            self._sessions[session_id] = _WorkerSession(task=task)
            return {
                "session_id": session_id,
                "agent_task_view_sha256": task_sha,
            }
        if op == "DECIDE":
            require_exact_fields(
                payload,
                {"session_id", "turn", "observations", "trajectory"},
                field="DECIDE payload",
            )
            session_id = required_string(payload, "session_id")
            turn = required_turn(payload)
            session = self._session(session_id)
            self._require_turn(session, turn)
            observations = observations_from_wire(
                payload.get("observations")
            )
            trajectory = trajectory_from_wire(
                payload.get("trajectory")
            )
            decision = self._policy.decide(
                session.task,
                observations,
                trajectory,
            )
            result = {
                "session_id": session_id,
                "turn": turn,
                "decision": decision_to_wire(decision),
            }
            session.next_turn += 1
            return result
        if op == "GENERATE":
            require_exact_fields(
                payload,
                {"session_id", "turn", "observations", "attempt"},
                field="GENERATE payload",
            )
            session_id = required_string(payload, "session_id")
            turn = required_turn(payload)
            attempt = required_attempt(payload)
            session = self._session(session_id)
            self._require_turn(session, turn)
            observations = observations_from_wire(
                payload.get("observations")
            )
            candidate = self._generator.generate(
                session.task,
                observations,
                attempt,
            )
            if not isinstance(candidate, str):
                raise AgentProtocolError(
                    "reference generator returned non-string candidate"
                )
            result = {
                "session_id": session_id,
                "turn": turn,
                "candidate": candidate,
            }
            session.next_turn += 1
            return result
        if op == "CLOSE":
            require_exact_fields(
                payload,
                {"session_id"},
                field="CLOSE payload",
            )
            session_id = required_string(payload, "session_id")
            if session_id not in self._sessions:
                raise AgentProtocolError(
                    "unknown or closed agent session"
                )
            del self._sessions[session_id]
            return {
                "closed": True,
                "session_id": session_id,
            }
        if op == "SHUTDOWN":
            require_exact_fields(
                payload,
                set(),
                field="SHUTDOWN payload",
            )
            self._sessions.clear()
            self._shutdown = True
            return {"shutdown": True}
        raise AgentProtocolError(
            f"unsupported agent operation: {op}"
        )

    def _session(self, session_id: str) -> _WorkerSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise AgentProtocolError(
                "unknown or closed agent session"
            ) from exc

    @staticmethod
    def _require_turn(
        session: _WorkerSession,
        turn: int,
    ) -> None:
        if turn != session.next_turn:
            raise AgentProtocolError(
                f"agent turn mismatch: {turn} != {session.next_turn}"
            )


def serve(
    *,
    input_stream: BinaryIO,
    output_stream: BinaryIO,
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
) -> int:
    worker = ReferenceAgentWorker()

    while not worker.should_shutdown:
        raw = input_stream.readline(max_message_bytes + 2)
        if not raw:
            return 0
        if len(raw) > max_message_bytes + 1:
            print(
                "agent worker rejected oversized frame",
                file=sys.stderr,
            )
            return 2

        request_id = _request_id_best_effort(raw)
        try:
            request = decode_message(
                raw,
                max_bytes=max_message_bytes,
            )
            if set(request) != {
                "protocol",
                "request_id",
                "op",
                "payload",
            }:
                raise AgentProtocolError(
                    "request envelope has unexpected fields"
                )
            op = request.get("op")
            if not isinstance(op, str) or not op:
                raise AgentProtocolError("request missing op")
            payload = strict_mapping(
                request.get("payload"),
                field="request payload",
            )
            result = worker.dispatch(op, payload)
            response = {
                "protocol": AGENT_PROTOCOL,
                "request_id": request["request_id"],
                "ok": True,
                "result": result,
            }
        except AgentProtocolError as exc:
            response = _error_response(
                request_id,
                code="PROTOCOL_REJECTED",
                message=str(exc),
            )
        except Exception:
            print("agent worker internal failure", file=sys.stderr)
            response = _error_response(
                request_id,
                code="AGENT_INTERNAL",
                message="agent worker internal failure",
            )

        try:
            output_stream.write(
                encode_message(
                    response,
                    max_bytes=max_message_bytes,
                )
                + b"\n"
            )
            output_stream.flush()
        except AgentProtocolError as exc:
            print(
                f"agent worker response encoding failed: {exc}",
                file=sys.stderr,
            )
            return 2
    return 0


def _error_response(
    request_id: str,
    *,
    code: str,
    message: str,
) -> dict[str, Any]:
    return {
        "protocol": AGENT_PROTOCOL,
        "request_id": request_id,
        "ok": False,
        "error": {
            "code": code,
            "message": message,
        },
    }


def _request_id_best_effort(raw: bytes) -> str:
    try:
        request = decode_message(
            raw,
            max_bytes=max(len(raw), DEFAULT_MAX_MESSAGE_BYTES),
        )
        value = request.get("request_id")
        if isinstance(value, str) and value:
            return value
    except AgentProtocolError:
        pass
    return "invalid-request"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Proof-Gated Gym reference agent endpoint over JSONL stdio"
    )
    parser.add_argument(
        "--max-message-bytes",
        type=int,
        default=DEFAULT_MAX_MESSAGE_BYTES,
    )
    args = parser.parse_args(argv)
    if args.max_message_bytes < 1024:
        parser.error("--max-message-bytes must be >= 1024")
    return serve(
        input_stream=sys.stdin.buffer,
        output_stream=sys.stdout.buffer,
        max_message_bytes=args.max_message_bytes,
    )


if __name__ == "__main__":
    raise SystemExit(main())
