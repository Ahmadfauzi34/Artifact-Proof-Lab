from __future__ import annotations

import argparse
import socketserver
import sys
from typing import Any

from .agent_wire import (
    AGENT_PROTOCOL,
    DEFAULT_MAX_MESSAGE_BYTES,
    AgentProtocolError,
    decode_message,
    encode_message,
    strict_mapping,
)
from .agent_worker import ReferenceAgentWorker


class _AgentRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        worker = ReferenceAgentWorker()
        max_bytes = self.server.max_message_bytes  # type: ignore[attr-defined]

        while True:
            raw = self.rfile.readline(max_bytes + 2)
            if not raw:
                return
            if len(raw) > max_bytes + 1:
                return

            request_id = "invalid-request"
            try:
                request = decode_message(raw, max_bytes=max_bytes)
                request_id = str(request["request_id"])
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
                if op == "SHUTDOWN":
                    raise AgentProtocolError(
                        "external controller cannot shut down agent service"
                    )
                payload = strict_mapping(
                    request.get("payload"),
                    field="request payload",
                )
                result = worker.dispatch(op, payload)
                if op == "HELLO":
                    operations = result.get("operations")
                    if isinstance(operations, list):
                        result = dict(result)
                        result["operations"] = [
                            item for item in operations if item != "SHUTDOWN"
                        ]
                response = {
                    "protocol": AGENT_PROTOCOL,
                    "request_id": request_id,
                    "ok": True,
                    "result": result,
                }
            except AgentProtocolError as exc:
                response = {
                    "protocol": AGENT_PROTOCOL,
                    "request_id": request_id,
                    "ok": False,
                    "error": {
                        "code": "PROTOCOL_REJECTED",
                        "message": str(exc),
                    },
                }
            except Exception:
                print(
                    "reference external agent internal failure",
                    file=sys.stderr,
                )
                response = {
                    "protocol": AGENT_PROTOCOL,
                    "request_id": request_id,
                    "ok": False,
                    "error": {
                        "code": "AGENT_INTERNAL",
                        "message": "reference external agent internal failure",
                    },
                }

            try:
                self.wfile.write(
                    encode_message(response, max_bytes=max_bytes) + b"\n"
                )
                self.wfile.flush()
            except (AgentProtocolError, OSError):
                return


class ReferenceAgentTCPServer(socketserver.ThreadingTCPServer):
    """Inspectable reference TCP server for external-endpoint conformance only."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ) -> None:
        if max_message_bytes < 1024:
            raise ValueError("max_message_bytes is too small for the protocol")
        self.max_message_bytes = int(max_message_bytes)
        super().__init__(server_address, _AgentRequestHandler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run inspectable Proof-Gated Gym reference agent as a TCP service"
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--max-message-bytes",
        type=int,
        default=DEFAULT_MAX_MESSAGE_BYTES,
    )
    args = parser.parse_args(argv)

    if not (1 <= args.port <= 65535):
        parser.error("--port must be in 1..65535")
    if args.max_message_bytes < 1024:
        parser.error("--max-message-bytes must be >= 1024")

    with ReferenceAgentTCPServer(
        (args.host, args.port),
        max_message_bytes=args.max_message_bytes,
    ) as server:
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
