from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import queue
import subprocess
import threading
from typing import Mapping, Protocol, Sequence

from .agent_wire import (
    AGENT_PROTOCOL,
    DEFAULT_MAX_MESSAGE_BYTES,
    AgentProtocolClosed,
    AgentProtocolError,
    AgentProtocolTimeout,
    agent_task_to_wire,
    decode_message,
    decision_from_wire,
    decision_to_wire,
    encode_message,
    observations_to_wire,
    strict_mapping,
)
from .core import AgentTaskView, Generator, Observation, Policy, PolicyDecision
from .ledger import sha256_json


_EOF = object()


@dataclass(frozen=True)
class AgentSessionStart:
    session_id: str
    agent_task_view_sha256: str


@dataclass(frozen=True)
class AgentDecisionReceipt:
    turn: int
    decision: PolicyDecision


@dataclass(frozen=True)
class AgentGenerationReceipt:
    turn: int
    candidate: str


class AgentEndpoint(Protocol):
    """Sanitized policy/generator boundary owned by the trusted controller."""

    def start(self, task: AgentTaskView) -> AgentSessionStart: ...

    def decide(
        self,
        session_id: str,
        observations: tuple[Observation, ...],
        trajectory: tuple[str, ...],
    ) -> AgentDecisionReceipt: ...

    def generate(
        self,
        session_id: str,
        observations: tuple[Observation, ...],
        attempt: int,
    ) -> AgentGenerationReceipt: ...

    def close(self, session_id: str) -> None: ...


@dataclass
class _LocalAgentSession:
    task: AgentTaskView
    next_turn: int = 1


class LocalReferenceAgentEndpoint:
    """In-process reference endpoint for contract/regression tests only."""

    def __init__(
        self,
        policy: Policy,
        *,
        generator: Generator | None = None,
    ) -> None:
        self._policy = policy
        self._generator = generator
        self._sessions: dict[str, _LocalAgentSession] = {}
        self._counter = 0

    def start(self, task: AgentTaskView) -> AgentSessionStart:
        self._counter += 1
        task_sha = sha256_json(asdict(task))
        material = f"{task.task_id}|{task_sha}|{self._counter}".encode("utf-8")
        session_id = "agent-session-" + hashlib.sha256(material).hexdigest()[:20]
        if session_id in self._sessions:
            raise AgentProtocolError("local agent session id collision")
        self._sessions[session_id] = _LocalAgentSession(task=task)
        return AgentSessionStart(
            session_id=session_id,
            agent_task_view_sha256=task_sha,
        )

    def decide(
        self,
        session_id: str,
        observations: tuple[Observation, ...],
        trajectory: tuple[str, ...],
    ) -> AgentDecisionReceipt:
        session = self._session(session_id)
        turn = session.next_turn
        decision = self._policy.decide(
            session.task,
            observations,
            trajectory,
        )
        if not isinstance(decision, PolicyDecision):
            raise AgentProtocolError("local agent policy returned invalid decision type")
        decision_to_wire(decision)
        session.next_turn += 1
        return AgentDecisionReceipt(turn=turn, decision=decision)

    def generate(
        self,
        session_id: str,
        observations: tuple[Observation, ...],
        attempt: int,
    ) -> AgentGenerationReceipt:
        session = self._session(session_id)
        if self._generator is None:
            raise AgentProtocolError("local agent endpoint has no generator")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise AgentProtocolError("generation attempt must be a positive integer")
        turn = session.next_turn
        candidate = self._generator.generate(
            session.task,
            observations,
            attempt,
        )
        if not isinstance(candidate, str):
            raise AgentProtocolError("local agent generator returned non-string candidate")
        session.next_turn += 1
        return AgentGenerationReceipt(turn=turn, candidate=candidate)

    def close(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _session(self, session_id: str) -> _LocalAgentSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise AgentProtocolError(
                f"unknown or closed agent session: {session_id}"
            ) from exc


class SubprocessAgentEndpoint:
    """Reference AgentEndpoint over strict JSONL subprocess transport.

    The subprocess boundary proves protocol separation, not filesystem or kernel
    isolation. Native evaluation should run an equivalent endpoint in a sandbox,
    container, VM, remote service, or other authority boundary that cannot read
    trusted host/private-pack state.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: Path | str | None = None,
        env: Mapping[str, str] | None = None,
        request_timeout: float = 10.0,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ) -> None:
        if not command:
            raise ValueError("agent endpoint command must not be empty")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if max_message_bytes < 1024:
            raise ValueError("max_message_bytes is too small for the protocol")
        self._command = tuple(str(item) for item in command)
        self._cwd = None if cwd is None else str(cwd)
        self._env = None if env is None else {**os.environ, **dict(env)}
        self._request_timeout = float(request_timeout)
        self._max_message_bytes = int(max_message_bytes)
        self._process: subprocess.Popen[bytes] | None = None
        self._stdout_queue: queue.Queue[bytes | object] = queue.Queue()
        self._request_counter = 0
        self._write_lock = threading.Lock()
        self._handshake_complete = False
        self._closed = False
        self._turns: dict[str, int] = {}
        self._tasks: dict[str, AgentTaskView] = {}

    def __enter__(self) -> "SubprocessAgentEndpoint":
        self._ensure_started()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    def start(self, task: AgentTaskView) -> AgentSessionStart:
        result = self._request(
            "START",
            {
                "task": agent_task_to_wire(task),
            },
        )
        if set(result) != {"session_id", "agent_task_view_sha256"}:
            self._reject_protocol("agent START response fields are invalid")
        session_id = result.get("session_id")
        task_sha = result.get("agent_task_view_sha256")
        if not isinstance(session_id, str) or not session_id:
            self._reject_protocol("agent START response missing session_id")
        expected_sha = sha256_json(asdict(task))
        if task_sha != expected_sha:
            self._reject_protocol("agent task-view commitment mismatch")
        if session_id in self._turns:
            self._reject_protocol("agent endpoint reused active session_id")
        self._turns[session_id] = 1
        self._tasks[session_id] = task
        return AgentSessionStart(
            session_id=session_id,
            agent_task_view_sha256=expected_sha,
        )

    def decide(
        self,
        session_id: str,
        observations: tuple[Observation, ...],
        trajectory: tuple[str, ...],
    ) -> AgentDecisionReceipt:
        turn = self._turn(session_id)
        result = self._request(
            "DECIDE",
            {
                "session_id": session_id,
                "turn": turn,
                "observations": observations_to_wire(observations),
                "trajectory": list(trajectory),
            },
        )
        if set(result) != {"session_id", "turn", "decision"}:
            self._reject_protocol("agent DECIDE response fields are invalid")
        if result.get("session_id") != session_id:
            self._reject_protocol("agent DECIDE session mismatch")
        if result.get("turn") != turn:
            self._reject_protocol("agent DECIDE turn mismatch")
        decision_raw = result.get("decision")
        if not isinstance(decision_raw, dict):
            self._reject_protocol("agent DECIDE decision must be an object")
        try:
            decision = decision_from_wire(decision_raw)
        except AgentProtocolError as exc:
            self._reject_protocol(str(exc))
        self._turns[session_id] = turn + 1
        return AgentDecisionReceipt(turn=turn, decision=decision)

    def generate(
        self,
        session_id: str,
        observations: tuple[Observation, ...],
        attempt: int,
    ) -> AgentGenerationReceipt:
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise AgentProtocolError("generation attempt must be a positive integer")
        turn = self._turn(session_id)
        result = self._request(
            "GENERATE",
            {
                "session_id": session_id,
                "turn": turn,
                "observations": observations_to_wire(observations),
                "attempt": attempt,
            },
        )
        if set(result) != {"session_id", "turn", "candidate"}:
            self._reject_protocol("agent GENERATE response fields are invalid")
        if result.get("session_id") != session_id:
            self._reject_protocol("agent GENERATE session mismatch")
        if result.get("turn") != turn:
            self._reject_protocol("agent GENERATE turn mismatch")
        candidate = result.get("candidate")
        if not isinstance(candidate, str):
            self._reject_protocol("agent GENERATE candidate must be a string")
        self._turns[session_id] = turn + 1
        return AgentGenerationReceipt(turn=turn, candidate=candidate)

    def close(self, session_id: str) -> None:
        if session_id not in self._turns:
            return
        result = self._request("CLOSE", {"session_id": session_id})
        if result != {"closed": True, "session_id": session_id}:
            self._reject_protocol("agent CLOSE acknowledgement is invalid")
        self._turns.pop(session_id, None)
        self._tasks.pop(session_id, None)

    def ping(self) -> bool:
        result = self._request("PING", {})
        if result != {"status": "ok"}:
            self._reject_protocol("agent PING response is invalid")
        return True

    def shutdown(self) -> None:
        if self._closed:
            return
        process = self._process
        if process is not None and process.poll() is None:
            try:
                result = self._request(
                    "SHUTDOWN",
                    {},
                    require_handshake=False,
                )
                if result != {"shutdown": True}:
                    raise AgentProtocolError(
                        "agent SHUTDOWN acknowledgement is invalid"
                    )
            except AgentProtocolError:
                self._terminate_process()
        self._turns.clear()
        self._tasks.clear()
        self._close_process()
        self._closed = True

    def _turn(self, session_id: str) -> int:
        try:
            return self._turns[session_id]
        except KeyError as exc:
            raise AgentProtocolError(
                f"unknown or closed agent session: {session_id}"
            ) from exc

    def _ensure_started(self) -> None:
        if self._closed:
            raise AgentProtocolClosed("agent endpoint is closed")
        if self._process is not None:
            if self._process.poll() is None:
                if not self._handshake_complete:
                    self._handshake()
                return
            raise AgentProtocolClosed(
                f"agent endpoint already exited with code {self._process.returncode}"
            )
        self._spawn_process()
        self._handshake()

    def _spawn_process(self) -> None:
        self._process = subprocess.Popen(
            self._command,
            cwd=self._cwd,
            env=self._env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        threading.Thread(
            target=self._stdout_reader,
            args=(self._process.stdout,),
            name="proof-gym-agent-stdout",
            daemon=True,
        ).start()
        threading.Thread(
            target=self._stderr_reader,
            args=(self._process.stderr,),
            name="proof-gym-agent-stderr",
            daemon=True,
        ).start()

    def _handshake(self) -> None:
        result = self._request("HELLO", {}, require_handshake=False)
        expected_fields = {
            "protocol",
            "operations",
            "endpoint_role",
            "reasoning_surface",
        }
        if set(result) != expected_fields:
            self._reject_protocol("agent handshake fields are invalid")
        if result.get("protocol") != AGENT_PROTOCOL:
            self._reject_protocol("agent handshake protocol mismatch")
        required = {
            "START",
            "DECIDE",
            "GENERATE",
            "CLOSE",
            "PING",
            "SHUTDOWN",
        }
        operations = result.get("operations")
        if not isinstance(operations, list) or not required <= set(operations):
            self._reject_protocol("agent endpoint does not expose required operations")
        if result.get("reasoning_surface") != "typed_decision_and_candidate_only":
            self._reject_protocol("agent endpoint reasoning surface is invalid")
        self._handshake_complete = True

    def _request(
        self,
        op: str,
        payload: Mapping[str, object],
        *,
        require_handshake: bool = True,
    ) -> dict[str, object]:
        if require_handshake:
            self._ensure_started()
        elif self._process is None:
            if self._closed:
                raise AgentProtocolClosed("agent endpoint is closed")
            self._spawn_process()

        process = self._process
        if process is None or process.poll() is not None:
            raise AgentProtocolClosed("agent endpoint process is not running")
        assert process.stdin is not None

        self._request_counter += 1
        request_id = f"agent-req-{self._request_counter:08d}"
        frame = encode_message(
            {
                "protocol": AGENT_PROTOCOL,
                "request_id": request_id,
                "op": op,
                "payload": dict(payload),
            },
            max_bytes=self._max_message_bytes,
        )
        try:
            with self._write_lock:
                process.stdin.write(frame + b"\n")
                process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise AgentProtocolClosed("agent endpoint stdin closed") from exc

        raw = self._read_response()
        try:
            response = decode_message(
                raw,
                max_bytes=self._max_message_bytes,
            )
            if response.get("request_id") != request_id:
                raise AgentProtocolError(
                    "agent endpoint response request_id mismatch"
                )
            ok = response.get("ok")
            if ok is True:
                if set(response) != {
                    "protocol",
                    "request_id",
                    "ok",
                    "result",
                }:
                    raise AgentProtocolError(
                        "agent success envelope has unexpected fields"
                    )
                return strict_mapping(
                    response.get("result"),
                    field="agent result",
                )
            if ok is not False:
                raise AgentProtocolError(
                    "agent endpoint response missing boolean ok"
                )
            if set(response) != {
                "protocol",
                "request_id",
                "ok",
                "error",
            }:
                raise AgentProtocolError(
                    "agent error envelope has unexpected fields"
                )
            error = strict_mapping(
                response.get("error"),
                field="agent error",
            )
            if set(error) != {"code", "message"}:
                raise AgentProtocolError(
                    "agent error object has unexpected fields"
                )
            code = error.get("code")
            message = error.get("message")
            if not isinstance(code, str) or not isinstance(message, str):
                raise AgentProtocolError(
                    "agent error code/message must be strings"
                )
            raise AgentProtocolError(
                f"agent endpoint error {code}: {message}"
            )
        except AgentProtocolError:
            self._terminate_process()
            raise

    def _reject_protocol(self, message: str):
        self._terminate_process()
        raise AgentProtocolError(message)

    def _read_response(self) -> bytes:
        try:
            item = self._stdout_queue.get(timeout=self._request_timeout)
        except queue.Empty as exc:
            process = self._process
            if process is not None and process.poll() is not None:
                raise AgentProtocolClosed(
                    f"agent endpoint stdout closed; exit={process.returncode}"
                ) from exc
            self._terminate_process()
            raise AgentProtocolTimeout(
                "agent endpoint request timed out"
            ) from exc
        if item is _EOF:
            process = self._process
            code = None if process is None else process.poll()
            raise AgentProtocolClosed(
                f"agent endpoint stdout closed; exit={code}"
            )
        assert isinstance(item, bytes)
        if len(item) > self._max_message_bytes + 1:
            self._terminate_process()
            raise AgentProtocolError(
                "agent endpoint response exceeds max message size"
            )
        return item

    def _stdout_reader(self, stream) -> None:
        try:
            while True:
                line = stream.readline(self._max_message_bytes + 2)
                if not line:
                    break
                self._stdout_queue.put(line)
                if len(line) > self._max_message_bytes + 1:
                    break
        finally:
            self._stdout_queue.put(_EOF)

    def _stderr_reader(self, stream) -> None:
        for _raw in iter(stream.readline, b""):
            pass

    def _terminate_process(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)

    def _close_process(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass
        if process.poll() is None:
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._terminate_process()
        for stream in (process.stdout, process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        self._process = None
        self._handshake_complete = False
