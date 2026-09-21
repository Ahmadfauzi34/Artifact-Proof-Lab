from __future__ import annotations

import inspect
import os
from pathlib import Path
import socket
import socketserver
import sys
import threading
import time
import unittest

import gym.external_agent as external_agent_module
from gym.agent_socket_server import ReferenceAgentTCPServer
from gym.agent_wire import (
    AGENT_PROTOCOL,
    AgentProtocolClosed,
    AgentProtocolError,
    AgentProtocolTimeout,
    decode_message,
    encode_message,
)
from gym.core import ReferenceGym
from gym.external_agent import ExternalAgentEndpoint
from gym.host import create_environment
from gym.io import load_task
from gym.isolated_agent import run_isolated_episode
from gym.private_pack import PrivateHoldoutPack
from gym.subprocess_host import SubprocessHostGateway


ROOT = Path(__file__).resolve().parents[1]
GYM_ROOT = ROOT / "gym"
REFERENCE_PACK = GYM_ROOT / "private_holdout_reference"


def _python_env():
    current = os.environ.get("PYTHONPATH", "")
    parts = [str(ROOT), str(ROOT / "src")]
    if current:
        parts.append(current)
    return {"PYTHONPATH": os.pathsep.join(parts)}


def _host_command(pack: Path):
    return [
        sys.executable,
        "-B",
        "-m",
        "gym.private_host_server",
        "--pack",
        str(pack),
    ]


class _RunningReferenceServer:
    def __init__(self):
        self.server = ReferenceAgentTCPServer(("127.0.0.1", 0))
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        )

    @property
    def address(self):
        host, port = self.server.server_address
        return str(host), int(port)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)


class _WrongRequestIdHandler(socketserver.StreamRequestHandler):
    def handle(self):
        raw = self.rfile.readline(1024 * 1024 + 2)
        request = decode_message(raw)
        response = {
            "protocol": AGENT_PROTOCOL,
            "request_id": "wrong-request-id",
            "ok": True,
            "result": {
                "protocol": AGENT_PROTOCOL,
                "operations": [
                    "START",
                    "DECIDE",
                    "GENERATE",
                    "CLOSE",
                    "PING",
                ],
                "endpoint_role": "fake-external-agent",
                "reasoning_surface": "typed_decision_and_candidate_only",
            },
        }
        self.wfile.write(encode_message(response) + b"\n")
        self.wfile.flush()


class _TimeoutHandler(socketserver.StreamRequestHandler):
    request_count = 0

    def handle(self):
        type(self).request_count += 1
        self.rfile.readline(1024 * 1024 + 2)
        time.sleep(2.0)


class GymG24Tests(unittest.TestCase):
    def test_external_endpoint_has_no_agent_launcher_surface(self):
        endpoint = ExternalAgentEndpoint("127.0.0.1", 12345)
        self.assertFalse(endpoint.controller_launched_agent)
        self.assertFalse(hasattr(endpoint, "_command"))
        self.assertFalse(hasattr(endpoint, "_cwd"))
        self.assertFalse(hasattr(endpoint, "_env"))
        source = inspect.getsource(external_agent_module)
        self.assertNotIn("subprocess", source)
        self.assertNotIn("Popen", source)

    def test_reference_external_server_does_not_advertise_shutdown(self):
        with _RunningReferenceServer() as running:
            host, port = running.address
            with socket.create_connection((host, port), timeout=1.0) as sock:
                request = {
                    "protocol": AGENT_PROTOCOL,
                    "request_id": "hello-1",
                    "op": "HELLO",
                    "payload": {},
                }
                sock.sendall(encode_message(request) + b"\n")
                raw = self._recv_line(sock)
                response = decode_message(raw)
                self.assertTrue(response["ok"])
                operations = response["result"]["operations"]
                self.assertNotIn("SHUTDOWN", operations)

    def test_external_controller_cannot_shutdown_agent_service(self):
        with _RunningReferenceServer() as running:
            host, port = running.address
            with socket.create_connection((host, port), timeout=1.0) as sock:
                request = {
                    "protocol": AGENT_PROTOCOL,
                    "request_id": "shutdown-1",
                    "op": "SHUTDOWN",
                    "payload": {},
                }
                sock.sendall(encode_message(request) + b"\n")
                response = decode_message(self._recv_line(sock))
                self.assertFalse(response["ok"])
                self.assertIn(
                    "cannot shut down",
                    response["error"]["message"],
                )

            with ExternalAgentEndpoint(host, port) as endpoint:
                self.assertTrue(endpoint.ping())

    def test_disconnect_does_not_terminate_external_service(self):
        task = load_task(GYM_ROOT / "tasks" / "train" / "tg_a7.json")
        with _RunningReferenceServer() as running:
            host, port = running.address
            with ExternalAgentEndpoint(host, port) as first:
                result = run_isolated_episode(
                    ReferenceGym(create_environment),
                    task,
                    first,
                )
                self.assertTrue(result.accepted)
                self.assertTrue(first.ping())

            with ExternalAgentEndpoint(host, port) as second:
                self.assertTrue(second.ping())
                result = run_isolated_episode(
                    ReferenceGym(create_environment),
                    task,
                    second,
                )
                self.assertTrue(result.accepted)

    def test_wrong_request_id_poisons_external_connection(self):
        with socketserver.ThreadingTCPServer(
            ("127.0.0.1", 0),
            _WrongRequestIdHandler,
        ) as server:
            thread = threading.Thread(
                target=server.serve_forever,
                daemon=True,
            )
            thread.start()
            host, port = server.server_address
            endpoint = ExternalAgentEndpoint(
                str(host),
                int(port),
                request_timeout=1.0,
            )
            try:
                with self.assertRaises(AgentProtocolError) as caught:
                    endpoint.ping()
                self.assertIn("request_id mismatch", str(caught.exception))
                with self.assertRaises(AgentProtocolClosed):
                    endpoint.ping()
            finally:
                endpoint.disconnect()
                server.shutdown()
                thread.join(timeout=2.0)

    def test_timeout_does_not_retry_or_reconnect(self):
        _TimeoutHandler.request_count = 0
        with socketserver.ThreadingTCPServer(
            ("127.0.0.1", 0),
            _TimeoutHandler,
        ) as server:
            thread = threading.Thread(
                target=server.serve_forever,
                daemon=True,
            )
            thread.start()
            host, port = server.server_address
            endpoint = ExternalAgentEndpoint(
                str(host),
                int(port),
                request_timeout=0.1,
            )
            try:
                with self.assertRaises(AgentProtocolTimeout):
                    endpoint.ping()
                with self.assertRaises(AgentProtocolClosed):
                    endpoint.ping()
                self.assertEqual(_TimeoutHandler.request_count, 1)
            finally:
                endpoint.disconnect()
                server.shutdown()
                thread.join(timeout=2.0)

    def test_external_agent_and_private_host_run_end_to_end(self):
        pack = PrivateHoldoutPack.load(
            REFERENCE_PACK,
            expected_runtime_id="reference-v1",
        )
        task = pack.task("hg_f8")

        with _RunningReferenceServer() as running:
            host_addr, port = running.address
            with (
                SubprocessHostGateway(
                    _host_command(REFERENCE_PACK),
                    cwd=ROOT,
                    env=_python_env(),
                ) as host,
                ExternalAgentEndpoint(host_addr, port) as agent,
            ):
                isolated = run_isolated_episode(
                    ReferenceGym(host_gateway=host),
                    task,
                    agent,
                )
                self.assertTrue(isolated.accepted)
                self.assertTrue(isolated.agent_transcript_valid)
                self.assertFalse(isolated.episode.learning_updated)
                self.assertFalse(agent.controller_launched_agent)
                self.assertTrue(host.ping())
                self.assertTrue(agent.ping())

    @staticmethod
    def _recv_line(sock: socket.socket) -> bytes:
        data = bytearray()
        while True:
            piece = sock.recv(1)
            if not piece:
                break
            data.extend(piece)
            if piece == b"\n":
                break
        return bytes(data)


if __name__ == "__main__":
    unittest.main()
