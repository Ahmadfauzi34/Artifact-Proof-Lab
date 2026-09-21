from __future__ import annotations

import argparse
from dataclasses import asdict
import sys
from pathlib import Path
from typing import Any, BinaryIO, Mapping

from .core import HostActionRejected, HostTaskDescriptor
from .host import create_environment
from .host_boundary import LocalReferenceHost
from .host_wire import (
    DEFAULT_MAX_MESSAGE_BYTES,
    PROTOCOL,
    HostProtocolError,
    agent_task_to_wire,
    decode_message,
    encode_message,
    observation_to_wire,
    outcome_to_wire,
    require_exact_fields,
    required_string,
    strict_mapping,
)
from .io import load_tasks
from .ledger import sha256_json
from .private_pack import PrivateHoldoutPack, PrivatePackError


OPERATIONS = ("HELLO", "START", "STEP", "SEMANTIC_VERDICT", "CLOSE", "PING", "SHUTDOWN")


class ReferencePrivateHostServer:
    """Inspectable reference server; protocol evidence, not native competence."""

    def __init__(self, tasks: Mapping[str, HostTaskDescriptor]) -> None:
        self._tasks = dict(tasks)
        self._host = LocalReferenceHost(create_environment)
        self._active_sessions: set[str] = set()
        self._shutdown = False

    def dispatch(self, op: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if op == "HELLO":
            require_exact_fields(payload, set(), field="HELLO payload")
            return {
                "protocol": PROTOCOL,
                "operations": list(OPERATIONS),
                "server_role": "reference_private_host",
                "native_competence_claim": False,
            }
        if op == "PING":
            require_exact_fields(payload, set(), field="PING payload")
            return {"status": "ok"}
        if op == "START":
            require_exact_fields(
                payload,
                {"task_id", "host_task_commitment"},
                field="START payload",
            )
            return self._start(payload)
        if op == "STEP":
            require_exact_fields(
                payload,
                {"session_id", "action", "candidate"},
                field="STEP payload",
            )
            session_id = required_string(payload, "session_id")
            action = required_string(payload, "action")
            candidate = payload.get("candidate")
            if candidate is not None and not isinstance(candidate, str):
                raise HostProtocolError("candidate must be string or null")
            if candidate is not None and action != "CALL_GENERATIVE":
                raise HostProtocolError("candidate is only valid for CALL_GENERATIVE")
            outcome = self._host.step(session_id, action, candidate=candidate)
            return outcome_to_wire(outcome)
        if op == "SEMANTIC_VERDICT":
            require_exact_fields(payload, {"session_id"}, field="SEMANTIC_VERDICT payload")
            session_id = required_string(payload, "session_id")
            return {"semantic_valid": self._host.semantic_verdict(session_id)}
        if op == "CLOSE":
            require_exact_fields(payload, {"session_id"}, field="CLOSE payload")
            session_id = required_string(payload, "session_id")
            self._host.close(session_id)
            self._active_sessions.discard(session_id)
            return {"closed": True}
        if op == "SHUTDOWN":
            require_exact_fields(payload, set(), field="SHUTDOWN payload")
            for session_id in tuple(self._active_sessions):
                self._host.close(session_id)
            self._active_sessions.clear()
            self._shutdown = True
            return {"shutdown": True}
        raise HostProtocolError(f"unsupported host operation: {op}")

    @property
    def should_shutdown(self) -> bool:
        return self._shutdown

    def _start(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        task_id = required_string(payload, "task_id")
        supplied_commitment = required_string(payload, "host_task_commitment")
        try:
            task = self._tasks[task_id]
        except KeyError as exc:
            raise HostProtocolError("unknown private host task") from exc
        expected_commitment = _task_commitment(task)
        if supplied_commitment != expected_commitment:
            raise HostProtocolError("host task commitment mismatch")

        start = self._host.start(task)
        self._active_sessions.add(start.session_id)
        return {
            "session_id": start.session_id,
            "task": agent_task_to_wire(start.agent_task),
            "observations": [observation_to_wire(item) for item in start.observations],
            "host_task_commitment": expected_commitment,
        }


def serve(
    tasks_root: Path | None,
    *,
    pack_root: Path | None = None,
    input_stream: BinaryIO,
    output_stream: BinaryIO,
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
) -> int:
    try:
        if pack_root is not None:
            if tasks_root is not None:
                raise PrivatePackError("provide tasks_root or pack_root, not both")
            pack = PrivateHoldoutPack.load(
                pack_root,
                expected_runtime_id="reference-v1",
            )
            tasks = {task_id: pack.task(task_id) for task_id in pack.task_ids}
        else:
            if tasks_root is None:
                raise PrivatePackError("private host registry source is missing")
            tasks = {task.task_id: task for task in load_tasks(tasks_root)}
    except (OSError, ValueError, PrivatePackError) as exc:
        print(f"private host registry rejected: {exc}", file=sys.stderr)
        return 2

    if not tasks:
        print("private host task registry is empty", file=sys.stderr)
        return 2
    server = ReferencePrivateHostServer(tasks)

    while not server.should_shutdown:
        raw = input_stream.readline(max_message_bytes + 2)
        if not raw:
            return 0
        if len(raw) > max_message_bytes + 1:
            print("private host rejected oversized frame", file=sys.stderr)
            return 2

        request_id = _request_id_best_effort(raw)
        try:
            request = decode_message(raw, max_bytes=max_message_bytes)
            if set(request) != {"protocol", "request_id", "op", "payload"}:
                raise HostProtocolError("request envelope has unexpected fields")
            op = request.get("op")
            if not isinstance(op, str) or not op:
                raise HostProtocolError("request missing op")
            payload = strict_mapping(request.get("payload"), field="request payload")
            result = server.dispatch(op, payload)
            response = {
                "protocol": PROTOCOL,
                "request_id": request["request_id"],
                "ok": True,
                "result": result,
            }
        except HostActionRejected as exc:
            response = _error_response(request_id, code="ACTION_REJECTED", message=str(exc))
        except KeyError:
            response = _error_response(
                request_id,
                code="UNKNOWN_SESSION",
                message="unknown or closed host session",
            )
        except HostProtocolError as exc:
            response = _error_response(
                request_id,
                code="PROTOCOL_REJECTED",
                message=str(exc),
            )
        except Exception:
            print("private host internal failure", file=sys.stderr)
            response = _error_response(
                request_id,
                code="HOST_INTERNAL",
                message="private host internal failure",
            )

        try:
            output_stream.write(encode_message(response, max_bytes=max_message_bytes) + b"\n")
            output_stream.flush()
        except HostProtocolError as exc:
            print(f"private host response encoding failed: {exc}", file=sys.stderr)
            return 2
    return 0


def _error_response(request_id: str, *, code: str, message: str) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "request_id": request_id,
        "ok": False,
        "error": {
            "code": code,
            "message": message,
        },
    }


def _request_id_best_effort(raw: bytes) -> str:
    try:
        request = decode_message(raw, max_bytes=max(len(raw), DEFAULT_MAX_MESSAGE_BYTES))
        value = request.get("request_id")
        if isinstance(value, str) and value:
            return value
    except HostProtocolError:
        pass
    return "invalid-request"


def _task_commitment(task: HostTaskDescriptor) -> str:
    return sha256_json(asdict(task))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Proof-Gated Gym reference private host over JSONL stdio")
    registry = parser.add_mutually_exclusive_group()
    registry.add_argument(
        "--tasks",
        type=Path,
        help="host-side raw task registry root (G2.1 reference mode)",
    )
    registry.add_argument(
        "--pack",
        type=Path,
        help="sealed private holdout pack root (G2.2 reference mode)",
    )
    parser.add_argument(
        "--max-message-bytes",
        type=int,
        default=DEFAULT_MAX_MESSAGE_BYTES,
    )
    args = parser.parse_args(argv)
    if args.max_message_bytes < 1024:
        parser.error("--max-message-bytes must be >= 1024")
    tasks_root = args.tasks
    if tasks_root is None and args.pack is None:
        tasks_root = Path(__file__).parent / "tasks"
    return serve(
        tasks_root,
        pack_root=args.pack,
        input_stream=sys.stdin.buffer,
        output_stream=sys.stdout.buffer,
        max_message_bytes=args.max_message_bytes,
    )


if __name__ == "__main__":
    raise SystemExit(main())
