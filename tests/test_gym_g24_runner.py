from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
import threading
import unittest

from gym.agent_socket_server import ReferenceAgentTCPServer
from gym.run_external_agent_reference import main as run_reference


class _RunningReferenceServer:
    """Minimal copy of the reference-server harness from test_gym_g24."""

    def __init__(self):
        self.server = ReferenceAgentTCPServer(("127.0.0.1", 0))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

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


class GymG24RunnerTests(unittest.TestCase):
    def test_reference_runner_reports_launch_and_attestation_fields(self):
        with _RunningReferenceServer() as server:
            host, port = server.address
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = run_reference(["--agent-host", host, "--agent-port", str(port)])
            summary = json.loads(buffer.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(summary["agent_launch_mode"], "external_preexisting_endpoint")
        self.assertEqual(summary["authority_attestation"], "not_present")
        self.assertIs(summary["native_competence_claim"], False)
        self.assertIs(summary["controller_launched_agent"], False)
        self.assertTrue(summary["tasks"])
        self.assertTrue(all(row["accepted"] for row in summary["tasks"]))


if __name__ == "__main__":
    unittest.main()
