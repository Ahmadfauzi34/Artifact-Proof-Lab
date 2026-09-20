from dataclasses import replace
import os
from pathlib import Path
import sys
import unittest

from gym.core import HostActionRejected, ReferenceGym
from gym.host_wire import (
    PROTOCOL,
    HostProtocolClosed,
    HostProtocolError,
    HostProtocolTimeout,
    decode_message,
    encode_message,
)
from gym.host import create_environment
from gym.host_boundary import HostSessionStart, LocalReferenceHost
from gym.io import load_task
from gym.reference_policy import ReferenceGenerator, ReferencePolicy
from gym.subprocess_host import SubprocessHostGateway


ROOT = Path(__file__).resolve().parents[1]
GYM_ROOT = ROOT / "gym"


def server_command():
    return [
        sys.executable,
        "-B",
        "-m",
        "gym.private_host_server",
        "--tasks",
        str(GYM_ROOT / "tasks"),
    ]


def server_env():
    current = os.environ.get("PYTHONPATH", "")
    parts = [str(ROOT), str(ROOT / "src")]
    if current:
        parts.append(current)
    return {"PYTHONPATH": os.pathsep.join(parts)}


class GymG21WireTests(unittest.TestCase):
    def test_wire_is_strict_json_and_protocol_versioned(self):
        frame = encode_message(
            {
                "protocol": PROTOCOL,
                "request_id": "r1",
                "op": "PING",
                "payload": {},
            }
        )
        decoded = decode_message(frame)
        self.assertEqual(decoded["request_id"], "r1")

        with self.assertRaises(HostProtocolError):
            encode_message(
                {
                    "protocol": PROTOCOL,
                    "request_id": "r2",
                    "op": "PING",
                    "payload": {"bad": float("nan")},
                }
            )
        with self.assertRaises(HostProtocolError):
            decode_message(
                b'{"protocol":"wrong","request_id":"r3","op":"PING","payload":{}}'
            )

    def test_subprocess_start_surface_is_sanitized(self):
        task = load_task(GYM_ROOT / "tasks" / "train" / "tg_a7.json")
        with SubprocessHostGateway(server_command(), cwd=ROOT, env=server_env()) as host:
            self.assertTrue(host.ping())
            start = host.start(task)
            try:
                self.assertEqual(start.agent_task, task.agent_view())
                self.assertFalse(hasattr(start.agent_task, "split"))
                self.assertFalse(hasattr(start.agent_task, "skill_targets"))
                self.assertFalse(hasattr(start.agent_task, "environment"))
                self.assertFalse(hasattr(start.agent_task, "metadata"))
                self.assertGreater(len(start.observations), 0)
            finally:
                host.close(start.session_id)

    def test_reference_gym_runs_end_to_end_through_subprocess_host(self):
        task = load_task(GYM_ROOT / "tasks" / "train" / "tg_b3.json")
        with SubprocessHostGateway(server_command(), cwd=ROOT, env=server_env()) as host:
            result = ReferenceGym(host_gateway=host).run(
                task,
                ReferencePolicy(),
                generator=ReferenceGenerator(),
            )
            self.assertTrue(result.accepted, result.admission.reason)
            self.assertTrue(result.admission.semantic_valid)
            self.assertTrue(result.admission.provenance_valid)
            self.assertTrue(result.admission.trajectory_valid)
            self.assertTrue(result.admission.integrity_valid)

    def test_one_host_process_can_serve_multiple_episodes(self):
        first = load_task(GYM_ROOT / "tasks" / "train" / "tg_a7.json")
        second = load_task(GYM_ROOT / "tasks" / "validation" / "vg_d2.json")
        with SubprocessHostGateway(server_command(), cwd=ROOT, env=server_env()) as host:
            gym = ReferenceGym(host_gateway=host)
            first_result = gym.run(first, ReferencePolicy())
            second_result = gym.run(second, ReferencePolicy())
            self.assertTrue(first_result.accepted)
            self.assertTrue(second_result.accepted)
            self.assertTrue(host.ping())

    def test_descriptor_commitment_mismatch_fails_closed(self):
        task = load_task(GYM_ROOT / "tasks" / "train" / "tg_a7.json")
        altered = replace(
            task,
            metadata={**task.metadata, "local_only_mutation": True},
        )
        with SubprocessHostGateway(server_command(), cwd=ROOT, env=server_env()) as host:
            with self.assertRaises(HostProtocolError) as caught:
                host.start(altered)
            self.assertIn("PROTOCOL_REJECTED", str(caught.exception))

    def test_action_rejection_remains_typed(self):
        task = load_task(GYM_ROOT / "tasks" / "train" / "tg_a7.json")
        with SubprocessHostGateway(server_command(), cwd=ROOT, env=server_env()) as host:
            start = host.start(task)
            try:
                with self.assertRaises(HostActionRejected):
                    host.step(start.session_id, "NOT_AN_ALLOWED_ACTION")
            finally:
                host.close(start.session_id)

    def test_candidate_cannot_cross_on_non_generative_action(self):
        task = load_task(GYM_ROOT / "tasks" / "train" / "tg_a7.json")
        with SubprocessHostGateway(server_command(), cwd=ROOT, env=server_env()) as host:
            start = host.start(task)
            try:
                with self.assertRaises(HostProtocolError):
                    host.step(
                        start.session_id,
                        "INSPECT_PROCESS_STATE",
                        candidate="hidden-payload",
                    )
            finally:
                host.close(start.session_id)

    def test_wrong_response_request_id_is_rejected(self):
        script = (
            "import json,sys;"
            "r=json.loads(sys.stdin.readline());"
            "print(json.dumps({"
            f"'protocol':'{PROTOCOL}',"
            "'request_id':'wrong-id','ok':True,"
            "'result':{'protocol':"
            f"'{PROTOCOL}',"
            "'operations':['START','STEP','SEMANTIC_VERDICT','CLOSE','PING','SHUTDOWN'],"
            "'server_role':'fake','native_competence_claim':False}}),flush=True)"
        )
        host = SubprocessHostGateway(
            [sys.executable, "-c", script],
            request_timeout=1.0,
        )
        try:
            with self.assertRaises(HostProtocolError) as caught:
                host.ping()
            self.assertIn("request_id mismatch", str(caught.exception))
            with self.assertRaises(HostProtocolClosed):
                host.ping()
        finally:
            host.shutdown()

    def test_host_timeout_terminates_transport_and_does_not_retry(self):
        script = "import sys,time; sys.stdin.readline(); time.sleep(30)"
        host = SubprocessHostGateway(
            [sys.executable, "-c", script],
            request_timeout=0.1,
        )
        try:
            with self.assertRaises(HostProtocolTimeout):
                host.ping()
            with self.assertRaises(HostProtocolClosed):
                host.ping()
        finally:
            host.shutdown()

    def test_host_stderr_is_not_reflected_to_client_error(self):
        script = (
            "import sys;"
            "sys.stdin.readline();"
            "print('HIDDEN_ORACLE_SECRET',file=sys.stderr,flush=True);"
            "sys.exit(7)"
        )
        host = SubprocessHostGateway(
            [sys.executable, "-c", script],
            request_timeout=1.0,
        )
        try:
            with self.assertRaises(HostProtocolClosed) as caught:
                host.ping()
            self.assertNotIn("HIDDEN_ORACLE_SECRET", str(caught.exception))
        finally:
            host.shutdown()

    def test_cleanup_failure_does_not_mask_primary_transport_failure(self):
        task = load_task(GYM_ROOT / "tasks" / "train" / "tg_a7.json")

        class BrokenHost:
            def start(self, task):
                env = create_environment(task)
                self.env = env
                return HostSessionStart("broken-session", task.agent_view(), env.reset())

            def step(self, session_id, action, *, candidate=None):
                raise HostProtocolTimeout("primary transport failure")

            def semantic_verdict(self, session_id):
                return False

            def close(self, session_id):
                raise HostProtocolError("cleanup failure")

        with self.assertRaises(HostProtocolTimeout) as caught:
            ReferenceGym(host_gateway=BrokenHost()).run(task, ReferencePolicy())
        self.assertIn("primary transport failure", str(caught.exception))

    def test_close_failure_after_success_remains_visible(self):
        task = load_task(GYM_ROOT / "tasks" / "validation" / "vg_d2.json")

        class CloseFailHost(LocalReferenceHost):
            def close(self, session_id):
                super().close(session_id)
                raise HostProtocolError("close failure")

        with self.assertRaises(HostProtocolError) as caught:
            ReferenceGym(host_gateway=CloseFailHost(create_environment)).run(
                task,
                ReferencePolicy(),
            )
        self.assertIn("close failure", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
