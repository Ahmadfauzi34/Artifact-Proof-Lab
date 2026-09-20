from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import queue
import subprocess
import threading
from typing import Mapping, Sequence

from .core import HostActionRejected, HostTaskDescriptor, StepOutcome
from .host_boundary import HostSessionStart
from .host_wire import (
    DEFAULT_MAX_MESSAGE_BYTES,
    PROTOCOL,
    HostProtocolClosed,
    HostProtocolError,
    HostProtocolTimeout,
    agent_task_from_wire,
    decode_message,
    encode_message,
    observation_from_wire,
    outcome_from_wire,
    strict_mapping,
)
from .ledger import sha256_json


_EOF = object()


class SubprocessHostGateway:
    """Reference HostGateway over a strict JSONL subprocess transport."""

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
            raise ValueError("private host command must not be empty")
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

    def __enter__(self) -> "SubprocessHostGateway":
        self._ensure_started()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    def start(self, task: HostTaskDescriptor) -> HostSessionStart:
        result = self._request(
            "START",
            {
                "task_id": task.task_id,
                "host_task_commitment": _task_commitment(task),
            },
        )
        commitment = result.get("host_task_commitment")
        if commitment != _task_commitment(task):
            self._reject_protocol("private host task commitment mismatch")
        try:
            agent_task_raw = strict_mapping(result.get("task"), field="session task")
            observations_raw = result.get("observations")
            if not isinstance(observations_raw, list):
                raise HostProtocolError("session observations must be a list")
            session_id = result.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                raise HostProtocolError("session start missing session_id")
            agent_task = agent_task_from_wire(agent_task_raw)
            if agent_task != task.agent_view():
                raise HostProtocolError("private host returned a different sanitized task surface")
            observations = []
            for raw in observations_raw:
                if not isinstance(raw, dict):
                    raise HostProtocolError("session observation must be an object")
                observations.append(observation_from_wire(raw))
        except HostProtocolError as exc:
            self._reject_protocol(str(exc))
        return HostSessionStart(
            session_id=session_id,
            agent_task=agent_task,
            observations=tuple(observations),
        )

    def step(self, session_id: str, action: str, *, candidate: str | None = None) -> StepOutcome:
        if candidate is not None and action != "CALL_GENERATIVE":
            raise HostProtocolError("candidate is only valid for CALL_GENERATIVE")
        result = self._request(
            "STEP",
            {
                "session_id": session_id,
                "action": action,
                "candidate": candidate,
            },
        )
        try:
            return outcome_from_wire(result)
        except HostProtocolError as exc:
            self._reject_protocol(str(exc))

    def semantic_verdict(self, session_id: str) -> bool:
        result = self._request("SEMANTIC_VERDICT", {"session_id": session_id})
        verdict = result.get("semantic_valid")
        if set(result) != {"semantic_valid"} or not isinstance(verdict, bool):
            self._reject_protocol("semantic verdict response is invalid")
        return verdict

    def close(self, session_id: str) -> None:
        result = self._request("CLOSE", {"session_id": session_id})
        if result != {"closed": True}:
            self._reject_protocol("private host close acknowledgement is invalid")

    def ping(self) -> bool:
        result = self._request("PING", {})
        if set(result) != {"status"}:
            self._reject_protocol("private host ping response is invalid")
        return result.get("status") == "ok"

    def shutdown(self) -> None:
        if self._closed:
            return
        process = self._process
        if process is not None and process.poll() is None:
            try:
                result = self._request("SHUTDOWN", {}, require_handshake=False)
                if result != {"shutdown": True}:
                    raise HostProtocolError("private host shutdown acknowledgement is invalid")
            except HostProtocolError:
                self._terminate_process()
        self._close_process()
        self._closed = True

    def _ensure_started(self) -> None:
        if self._closed:
            raise HostProtocolClosed("private host gateway is closed")
        if self._process is not None:
            if self._process.poll() is None:
                if not self._handshake_complete:
                    self._handshake()
                return
            raise HostProtocolClosed(
                f"private host already exited with code {self._process.returncode}"
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
            name="proof-gym-host-stdout",
            daemon=True,
        ).start()
        threading.Thread(
            target=self._stderr_reader,
            args=(self._process.stderr,),
            name="proof-gym-host-stderr",
            daemon=True,
        ).start()

    def _handshake(self) -> None:
        result = self._request("HELLO", {}, require_handshake=False)
        if result.get("protocol") != PROTOCOL:
            self._reject_protocol("private host handshake protocol mismatch")
        operations = result.get("operations")
        required = {"START", "STEP", "SEMANTIC_VERDICT", "CLOSE", "PING", "SHUTDOWN"}
        if set(result) != {"protocol", "operations", "server_role", "native_competence_claim"}:
            self._reject_protocol("private host handshake fields are invalid")
        if not isinstance(operations, list) or not required <= set(operations):
            self._reject_protocol("private host does not expose required operations")
        if result.get("native_competence_claim") is not False:
            self._reject_protocol("reference private host must not claim native competence")
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
                raise HostProtocolClosed("private host gateway is closed")
            self._spawn_process()

        process = self._process
        if process is None or process.poll() is not None:
            raise HostProtocolClosed("private host process is not running")
        assert process.stdin is not None

        self._request_counter += 1
        request_id = f"req-{self._request_counter:08d}"
        frame = encode_message(
            {
                "protocol": PROTOCOL,
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
            raise HostProtocolClosed("private host stdin closed") from exc

        raw = self._read_response()
        try:
            response = decode_message(raw, max_bytes=self._max_message_bytes)
            if response.get("request_id") != request_id:
                raise HostProtocolError("private host response request_id mismatch")
            ok = response.get("ok")
            if ok is True:
                if set(response) != {"protocol", "request_id", "ok", "result"}:
                    raise HostProtocolError("private host success envelope has unexpected fields")
                return strict_mapping(response.get("result"), field="host result")
            if ok is not False:
                raise HostProtocolError("private host response missing boolean ok")
            if set(response) != {"protocol", "request_id", "ok", "error"}:
                raise HostProtocolError("private host error envelope has unexpected fields")
            error = strict_mapping(response.get("error"), field="host error")
            if set(error) != {"code", "message"}:
                raise HostProtocolError("private host error object has unexpected fields")
            code = error.get("code")
            message = error.get("message")
            if not isinstance(code, str) or not isinstance(message, str):
                raise HostProtocolError("private host error code/message must be strings")
            if code == "ACTION_REJECTED":
                raise HostActionRejected(message)
            raise HostProtocolError(f"private host error {code}: {message}")
        except HostActionRejected:
            raise
        except HostProtocolError:
            self._terminate_process()
            raise

    def _reject_protocol(self, message: str):
        self._terminate_process()
        raise HostProtocolError(message)

    def _read_response(self) -> bytes:
        try:
            item = self._stdout_queue.get(timeout=self._request_timeout)
        except queue.Empty as exc:
            process = self._process
            if process is not None and process.poll() is not None:
                raise HostProtocolClosed(
                    f"private host stdout closed; exit={process.returncode}"
                ) from exc
            self._terminate_process()
            raise HostProtocolTimeout("private host request timed out") from exc
        if item is _EOF:
            process = self._process
            code = None if process is None else process.poll()
            raise HostProtocolClosed(
                f"private host stdout closed; exit={code}"
            )
        assert isinstance(item, bytes)
        if len(item) > self._max_message_bytes + 1:
            self._terminate_process()
            raise HostProtocolError("private host response exceeds max message size")
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


def _task_commitment(task: HostTaskDescriptor) -> str:
    return sha256_json(asdict(task))
