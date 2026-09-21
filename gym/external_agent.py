from __future__ import annotations

from dataclasses import asdict
import socket
from typing import Any

from .agent_endpoint import (
    AgentDecisionReceipt,
    AgentEndpoint,
    AgentGenerationReceipt,
    AgentSessionStart,
)
from .agent_wire import (
    AGENT_PROTOCOL,
    DEFAULT_MAX_MESSAGE_BYTES,
    AgentProtocolClosed,
    AgentProtocolError,
    AgentProtocolTimeout,
    agent_task_to_wire,
    decode_message,
    decision_from_wire,
    encode_message,
    observations_to_wire,
    strict_mapping,
)
from .core import AgentTaskView, Observation
from .ledger import sha256_json


class ExternalAgentEndpoint(AgentEndpoint):
    """AgentEndpoint over a pre-existing TCP service.

    This class deliberately has no command/cwd/env launcher surface. It proves
    controller-side launch decoupling only. Filesystem, kernel, credential,
    network, model, or machine identity isolation requires independent external
    evidence and is not inferred from a TCP connection.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        connect_timeout: float = 5.0,
        request_timeout: float = 10.0,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ) -> None:
        if not isinstance(host, str) or not host:
            raise ValueError("external agent host must be a non-empty string")
        if isinstance(port, bool) or not isinstance(port, int) or not (1 <= port <= 65535):
            raise ValueError("external agent port must be in 1..65535")
        if connect_timeout <= 0:
            raise ValueError("connect_timeout must be positive")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if max_message_bytes < 1024:
            raise ValueError("max_message_bytes is too small for the protocol")

        self._host = host
        self._port = port
        self._connect_timeout = float(connect_timeout)
        self._request_timeout = float(request_timeout)
        self._max_message_bytes = int(max_message_bytes)

        self._socket: socket.socket | None = None
        self._request_counter = 0
        self._handshake_complete = False
        self._closed = False
        self._poisoned = False
        self._turns: dict[str, int] = {}

    def __enter__(self) -> "ExternalAgentEndpoint":
        self._ensure_connected()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

    @property
    def peer(self) -> tuple[str, int]:
        return self._host, self._port

    @property
    def controller_launched_agent(self) -> bool:
        return False

    def start(self, task: AgentTaskView) -> AgentSessionStart:
        result = self._request(
            "START",
            {"task": agent_task_to_wire(task)},
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
            self._reject_protocol("external agent reused active session_id")
        self._turns[session_id] = 1
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

    def ping(self) -> bool:
        result = self._request("PING", {})
        if result != {"status": "ok"}:
            self._reject_protocol("agent PING response is invalid")
        return True

    def disconnect(self) -> None:
        """Close only the controller-side transport.

        Unlike SubprocessAgentEndpoint.shutdown(), this never sends SHUTDOWN:
        the trusted controller does not own the external agent process/service.
        """
        self._turns.clear()
        sock = self._socket
        self._socket = None
        self._handshake_complete = False
        self._closed = True
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    def _turn(self, session_id: str) -> int:
        try:
            return self._turns[session_id]
        except KeyError as exc:
            raise AgentProtocolError(
                f"unknown or closed external agent session: {session_id}"
            ) from exc

    def _ensure_connected(self) -> None:
        if self._closed:
            raise AgentProtocolClosed("external agent endpoint is closed")
        if self._poisoned:
            raise AgentProtocolClosed("external agent endpoint is poisoned")
        if self._socket is not None:
            if not self._handshake_complete:
                self._handshake()
            return
        try:
            sock = socket.create_connection(
                (self._host, self._port),
                timeout=self._connect_timeout,
            )
        except TimeoutError as exc:
            self._poison()
            raise AgentProtocolTimeout(
                "external agent connection timed out"
            ) from exc
        except OSError as exc:
            self._poison()
            raise AgentProtocolClosed(
                f"external agent connection failed: {type(exc).__name__}"
            ) from exc
        sock.settimeout(self._request_timeout)
        self._socket = sock
        self._handshake()

    def _handshake(self) -> None:
        result = self._request(
            "HELLO",
            {},
            require_handshake=False,
        )
        expected_fields = {
            "protocol",
            "operations",
            "endpoint_role",
            "reasoning_surface",
        }
        if set(result) != expected_fields:
            self._reject_protocol("external agent handshake fields are invalid")
        if result.get("protocol") != AGENT_PROTOCOL:
            self._reject_protocol("external agent handshake protocol mismatch")
        required = {
            "START",
            "DECIDE",
            "GENERATE",
            "CLOSE",
            "PING",
        }
        operations = result.get("operations")
        if not isinstance(operations, list) or not required <= set(operations):
            self._reject_protocol(
                "external agent does not expose required operations"
            )
        if result.get("reasoning_surface") != "typed_decision_and_candidate_only":
            self._reject_protocol(
                "external agent reasoning surface is invalid"
            )
        self._handshake_complete = True

    def _request(
        self,
        op: str,
        payload: dict[str, object],
        *,
        require_handshake: bool = True,
    ) -> dict[str, object]:
        if require_handshake:
            self._ensure_connected()
        elif self._socket is None:
            if self._closed or self._poisoned:
                raise AgentProtocolClosed("external agent endpoint is unavailable")
            try:
                sock = socket.create_connection(
                    (self._host, self._port),
                    timeout=self._connect_timeout,
                )
            except TimeoutError as exc:
                self._poison()
                raise AgentProtocolTimeout(
                    "external agent connection timed out"
                ) from exc
            except OSError as exc:
                self._poison()
                raise AgentProtocolClosed(
                    f"external agent connection failed: {type(exc).__name__}"
                ) from exc
            sock.settimeout(self._request_timeout)
            self._socket = sock

        sock = self._socket
        if sock is None:
            raise AgentProtocolClosed("external agent socket is not connected")

        self._request_counter += 1
        request_id = f"external-agent-req-{self._request_counter:08d}"
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
            sock.sendall(frame + b"\n")
        except socket.timeout as exc:
            self._poison()
            raise AgentProtocolTimeout(
                "external agent request write timed out"
            ) from exc
        except OSError as exc:
            self._poison()
            raise AgentProtocolClosed(
                f"external agent write failed: {type(exc).__name__}"
            ) from exc

        raw = self._read_line()
        try:
            response = decode_message(
                raw,
                max_bytes=self._max_message_bytes,
            )
            if response.get("request_id") != request_id:
                raise AgentProtocolError(
                    "external agent response request_id mismatch"
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
                        "external agent success envelope has unexpected fields"
                    )
                return strict_mapping(
                    response.get("result"),
                    field="external agent result",
                )
            if ok is not False:
                raise AgentProtocolError(
                    "external agent response missing boolean ok"
                )
            if set(response) != {
                "protocol",
                "request_id",
                "ok",
                "error",
            }:
                raise AgentProtocolError(
                    "external agent error envelope has unexpected fields"
                )
            error = strict_mapping(
                response.get("error"),
                field="external agent error",
            )
            if set(error) != {"code", "message"}:
                raise AgentProtocolError(
                    "external agent error object has unexpected fields"
                )
            code = error.get("code")
            message = error.get("message")
            if not isinstance(code, str) or not isinstance(message, str):
                raise AgentProtocolError(
                    "external agent error code/message must be strings"
                )
            raise AgentProtocolError(
                f"external agent error {code}: {message}"
            )
        except AgentProtocolError:
            self._poison()
            raise

    def _read_line(self) -> bytes:
        sock = self._socket
        if sock is None:
            raise AgentProtocolClosed("external agent socket is not connected")
        chunks = bytearray()
        try:
            while True:
                if len(chunks) > self._max_message_bytes:
                    self._poison()
                    raise AgentProtocolError(
                        "external agent response exceeds max message size"
                    )
                piece = sock.recv(1)
                if not piece:
                    self._poison()
                    raise AgentProtocolClosed(
                        "external agent connection closed before response"
                    )
                chunks.extend(piece)
                if piece == b"\n":
                    break
        except socket.timeout as exc:
            self._poison()
            raise AgentProtocolTimeout(
                "external agent request timed out"
            ) from exc
        except OSError as exc:
            self._poison()
            raise AgentProtocolClosed(
                f"external agent read failed: {type(exc).__name__}"
            ) from exc
        return bytes(chunks)

    def _reject_protocol(self, message: str) -> None:
        self._poison()
        raise AgentProtocolError(message)

    def _poison(self) -> None:
        self._poisoned = True
        self._turns.clear()
        sock = self._socket
        self._socket = None
        self._handshake_complete = False
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
