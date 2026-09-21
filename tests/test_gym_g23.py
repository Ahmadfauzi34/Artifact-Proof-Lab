from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path
import sys
import unittest

from gym.agent_endpoint import (
    AgentDecisionReceipt,
    AgentSessionStart,
    LocalReferenceAgentEndpoint,
    SubprocessAgentEndpoint,
)
from gym.agent_transcript import (
    AGENT_TRANSCRIPT_FORMAT,
    AgentBoundaryTranscript,
    AgentTranscriptError,
    verify_agent_transcript_against_episode,
    verify_agent_transcript_document,
)
from gym.agent_wire import (
    AGENT_PROTOCOL,
    AgentProtocolClosed,
    AgentProtocolError,
    AgentProtocolTimeout,
    decode_message,
    encode_message,
)
from gym.agent_worker import ReferenceAgentWorker
from gym.core import (
    AgentTaskView,
    MemoryLearningSink,
    PolicyDecision,
    ReferenceGym,
)
from gym.host import create_environment
from gym.io import load_task
from gym.isolated_agent import (
    AgentEndpointPolicyAdapter,
    run_isolated_episode,
)
from gym.ledger import sha256_json
from gym.private_pack import PrivateHoldoutPack
from gym.reference_policy import ReferenceGenerator, ReferencePolicy
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


def _agent_command():
    return [
        sys.executable,
        "-B",
        "-m",
        "gym.agent_worker",
    ]


def _host_command(pack: Path):
    return [
        sys.executable,
        "-B",
        "-m",
        "gym.private_host_server",
        "--pack",
        str(pack),
    ]


def _transcript_document(isolated):
    return {
        "format": AGENT_TRANSCRIPT_FORMAT,
        "root_sha256": isolated.agent_transcript_root_sha256,
        "entry_count": len(isolated.agent_transcript_entries),
        "entries": [
            deepcopy(dict(entry))
            for entry in isolated.agent_transcript_entries
        ],
    }


class GymG23Tests(unittest.TestCase):
    def test_agent_wire_is_strict_and_versioned(self):
        frame = encode_message(
            {
                "protocol": AGENT_PROTOCOL,
                "request_id": "a1",
                "op": "PING",
                "payload": {},
            }
        )
        self.assertEqual(decode_message(frame)["request_id"], "a1")

        with self.assertRaises(AgentProtocolError):
            decode_message(
                b'{"protocol":"proof-gym-agent-jsonl-v1","request_id":"a2","request_id":"a2","op":"PING","payload":{}}'
            )
        with self.assertRaises(AgentProtocolError):
            decode_message(
                b'{"protocol":"proof-gym-agent-jsonl-v1","request_id":"a3","op":"PING","payload":{"cost":1e999}}'
            )
        with self.assertRaises(AgentProtocolError):
            decode_message(
                b'{"protocol":"wrong","request_id":"a4","op":"PING","payload":{}}'
            )

    def test_local_endpoint_policy_receives_only_agent_task_view(self):
        task = load_task(GYM_ROOT / "tasks" / "train" / "tg_a7.json")
        seen = []

        class SpyPolicy:
            def decide(self, task_view, observations, trajectory):
                seen.append(task_view)
                return ReferencePolicy().decide(
                    task_view,
                    observations,
                    trajectory,
                )

        endpoint = LocalReferenceAgentEndpoint(SpyPolicy())
        learning = MemoryLearningSink()
        isolated = run_isolated_episode(
            ReferenceGym(create_environment),
            task,
            endpoint,
            learning_sink=learning,
        )

        self.assertTrue(isolated.accepted)
        self.assertTrue(isolated.agent_transcript_valid)
        self.assertEqual(learning.updates, [task.task_id])
        self.assertTrue(isolated.episode.learning_updated)
        self.assertGreater(len(seen), 0)
        self.assertIsInstance(seen[0], AgentTaskView)
        self.assertFalse(hasattr(seen[0], "split"))
        self.assertFalse(hasattr(seen[0], "skill_targets"))
        self.assertFalse(hasattr(seen[0], "environment"))
        self.assertFalse(hasattr(seen[0], "metadata"))

    def test_agent_transcript_binds_decisions_and_excludes_rationale(self):
        task = load_task(GYM_ROOT / "tasks" / "train" / "tg_a7.json")
        endpoint = LocalReferenceAgentEndpoint(ReferencePolicy())
        isolated = run_isolated_episode(
            ReferenceGym(create_environment),
            task,
            endpoint,
        )
        document = _transcript_document(isolated)
        valid, reason = verify_agent_transcript_document(document)
        self.assertTrue(valid, reason)

        decision_entries = [
            entry
            for entry in document["entries"]
            if entry["kind"] == "AGENT_DECISION"
        ]
        self.assertGreater(len(decision_entries), 0)
        for entry in decision_entries:
            self.assertEqual(
                set(entry["payload"]["decision"]),
                {"kind", "action"},
            )
            self.assertNotIn(
                "rationale",
                entry["payload"]["decision"],
            )

        bound, reason = verify_agent_transcript_against_episode(
            document,
            task=task.agent_view(),
            episode=isolated.episode,
        )
        self.assertTrue(bound, reason)

    def test_transcript_tamper_is_rejected(self):
        task = load_task(GYM_ROOT / "tasks" / "train" / "tg_a7.json")
        isolated = run_isolated_episode(
            ReferenceGym(create_environment),
            task,
            LocalReferenceAgentEndpoint(ReferencePolicy()),
        )
        document = _transcript_document(isolated)
        tampered = deepcopy(document)
        decision = next(
            entry
            for entry in tampered["entries"]
            if entry["kind"] == "AGENT_DECISION"
        )
        decision["payload"]["decision"]["action"] = "RESTART_PROCESS"

        valid, reason = verify_agent_transcript_document(tampered)
        self.assertFalse(valid)
        self.assertIn("payload hash", reason)

    def test_structurally_valid_wrong_decision_fails_episode_cross_binding(self):
        task = load_task(GYM_ROOT / "tasks" / "train" / "tg_a7.json")
        isolated = run_isolated_episode(
            ReferenceGym(create_environment),
            task,
            LocalReferenceAgentEndpoint(ReferencePolicy()),
        )
        episode = isolated.episode
        original = _transcript_document(isolated)
        start_session = original["entries"][0]["payload"]["session_id"]

        rebuilt = AgentBoundaryTranscript(
            task.agent_view(),
            start_session,
        )
        source_decisions = [
            entry["payload"]
            for entry in original["entries"]
            if entry["kind"] == "AGENT_DECISION"
        ]
        observations = tuple(episode.observations[:2])
        first = source_decisions[0]
        rebuilt.record_decision(
            turn=first["turn"],
            observations=observations,
            trajectory=(),
            decision=PolicyDecision.select("RESTART_PROCESS"),
        )
        rebuilt.record_close(session_id=start_session)

        valid, reason = verify_agent_transcript_against_episode(
            rebuilt.to_document(),
            task=task.agent_view(),
            episode=episode,
        )
        self.assertFalse(valid)
        self.assertIn("decision mismatch", reason)

    def test_generation_must_follow_call_generative_and_attempts_are_contiguous(self):
        task = load_task(
            GYM_ROOT / "tasks" / "validation" / "vg_e4.json"
        )
        transcript = AgentBoundaryTranscript(
            task.agent_view(),
            "agent-session-test",
        )
        transcript.record_generation(
            turn=1,
            observations=(),
            attempt=1,
            candidate="key=guess",
        )
        transcript.record_close(session_id="agent-session-test")
        valid, reason = verify_agent_transcript_document(
            transcript.to_document()
        )
        self.assertFalse(valid)
        self.assertIn("must follow a decision", reason)

        transcript = AgentBoundaryTranscript(
            task.agent_view(),
            "agent-session-test-2",
        )
        transcript.record_decision(
            turn=1,
            observations=(),
            trajectory=(),
            decision=PolicyDecision.call_generative(),
        )
        transcript.record_generation(
            turn=2,
            observations=(),
            attempt=2,
            candidate="key=guess",
        )
        transcript.record_close(session_id="agent-session-test-2")
        valid, reason = verify_agent_transcript_document(
            transcript.to_document()
        )
        self.assertFalse(valid)
        self.assertIn("attempt is not contiguous", reason)

    def test_local_endpoint_rejects_invalid_typed_decision(self):
        task = load_task(GYM_ROOT / "tasks" / "train" / "tg_a7.json")

        class InvalidPolicy:
            def decide(self, task_view, observations, trajectory):
                return PolicyDecision(
                    kind=PolicyDecision.call_generative().kind,
                    action="SHOULD_BE_NULL",
                )

        endpoint = LocalReferenceAgentEndpoint(InvalidPolicy())
        start = endpoint.start(task.agent_view())
        try:
            with self.assertRaises(AgentProtocolError):
                endpoint.decide(start.session_id, (), ())
        finally:
            endpoint.close(start.session_id)

    def test_generation_transcript_contains_digest_not_raw_candidate(self):
        task = load_task(
            GYM_ROOT / "tasks" / "validation" / "vg_e4.json"
        )
        endpoint = LocalReferenceAgentEndpoint(
            ReferencePolicy(),
            generator=ReferenceGenerator(),
        )
        isolated = run_isolated_episode(
            ReferenceGym(create_environment),
            task,
            endpoint,
        )
        self.assertTrue(isolated.accepted)
        document = _transcript_document(isolated)
        generations = [
            entry["payload"]
            for entry in document["entries"]
            if entry["kind"] == "AGENT_GENERATION"
        ]
        self.assertEqual(
            len(generations),
            isolated.episode.generative_calls,
        )
        self.assertGreater(len(generations), 0)
        for payload in generations:
            self.assertIn("candidate_sha256", payload)
            self.assertNotIn("candidate", payload)
        valid, reason = verify_agent_transcript_against_episode(
            document,
            task=task.agent_view(),
            episode=isolated.episode,
        )
        self.assertTrue(valid, reason)

    def test_successful_episode_with_invalid_transcript_cannot_update_learning(self):
        task = load_task(GYM_ROOT / "tasks" / "train" / "tg_a7.json")
        learning = MemoryLearningSink()

        class WrongTurnEndpoint(LocalReferenceAgentEndpoint):
            def decide(self, session_id, observations, trajectory):
                receipt = super().decide(
                    session_id,
                    observations,
                    trajectory,
                )
                return AgentDecisionReceipt(
                    turn=receipt.turn + 1,
                    decision=receipt.decision,
                )

        with self.assertRaises(AgentTranscriptError) as caught:
            run_isolated_episode(
                ReferenceGym(create_environment),
                task,
                WrongTurnEndpoint(ReferencePolicy()),
                learning_sink=learning,
            )
        self.assertIn("turn is not contiguous", str(caught.exception))
        self.assertEqual(learning.updates, [])

    def test_learning_does_not_update_when_endpoint_fails(self):
        task = load_task(GYM_ROOT / "tasks" / "train" / "tg_a7.json")
        learning = MemoryLearningSink()

        class FailingEndpoint:
            def __init__(self):
                self.task = None

            def start(self, task_view):
                self.task = task_view
                return AgentSessionStart(
                    session_id="fail-session",
                    agent_task_view_sha256=sha256_json(
                        asdict(task_view)
                    ),
                )

            def decide(self, session_id, observations, trajectory):
                raise AgentProtocolError("agent failed before decision")

            def generate(self, session_id, observations, attempt):
                raise AssertionError("generate must not be reached")

            def close(self, session_id):
                pass

        with self.assertRaises(AgentProtocolError):
            run_isolated_episode(
                ReferenceGym(create_environment),
                task,
                FailingEndpoint(),
                learning_sink=learning,
            )
        self.assertEqual(learning.updates, [])

    def test_close_failure_after_episode_prevents_learning(self):
        task = load_task(GYM_ROOT / "tasks" / "train" / "tg_a7.json")
        learning = MemoryLearningSink()

        class CloseFailEndpoint(LocalReferenceAgentEndpoint):
            def close(self, session_id):
                super().close(session_id)
                raise AgentProtocolError("agent close failed")

        with self.assertRaises(AgentProtocolError) as caught:
            run_isolated_episode(
                ReferenceGym(create_environment),
                task,
                CloseFailEndpoint(ReferencePolicy()),
                learning_sink=learning,
            )
        self.assertIn("agent close failed", str(caught.exception))
        self.assertEqual(learning.updates, [])

    def test_adapter_rejects_mid_session_task_switch(self):
        first = load_task(GYM_ROOT / "tasks" / "train" / "tg_a7.json")
        second = load_task(
            GYM_ROOT / "tasks" / "validation" / "vg_d2.json"
        )
        adapter = AgentEndpointPolicyAdapter(
            LocalReferenceAgentEndpoint(ReferencePolicy()),
            first.agent_view(),
        )
        try:
            with self.assertRaises(AgentTranscriptError):
                adapter.decide(
                    second.agent_view(),
                    (),
                    (),
                )
        finally:
            adapter.close()

    def test_worker_rejects_host_only_task_fields(self):
        worker = ReferenceAgentWorker()
        task = load_task(
            GYM_ROOT / "tasks" / "train" / "tg_a7.json"
        ).agent_view()
        task_wire = {
            "task_id": task.task_id,
            "domain": task.domain,
            "description": task.description,
            "allowed_actions": list(task.allowed_actions),
            "budget": dict(task.budget),
            "public_tests": list(task.public_tests),
            "public_inputs": list(task.public_inputs),
            "split": "train",
        }
        with self.assertRaises(AgentProtocolError) as caught:
            worker.dispatch("START", {"task": task_wire})
        self.assertIn("split", str(caught.exception))

    def test_reference_worker_turn_is_monotonic_and_rationale_free(self):
        worker = ReferenceAgentWorker()
        task = load_task(
            GYM_ROOT / "tasks" / "validation" / "vg_e4.json"
        ).agent_view()
        start = worker.dispatch(
            "START",
            {
                "task": {
                    "task_id": task.task_id,
                    "domain": task.domain,
                    "description": task.description,
                    "allowed_actions": list(task.allowed_actions),
                    "budget": dict(task.budget),
                    "public_tests": list(task.public_tests),
                    "public_inputs": list(task.public_inputs),
                }
            },
        )
        session_id = start["session_id"]

        with self.assertRaises(AgentProtocolError):
            worker.dispatch(
                "DECIDE",
                {
                    "session_id": session_id,
                    "turn": 2,
                    "observations": [],
                    "trajectory": [],
                },
            )

        decision = worker.dispatch(
            "DECIDE",
            {
                "session_id": session_id,
                "turn": 1,
                "observations": [],
                "trajectory": [],
            },
        )
        self.assertEqual(
            set(decision["decision"]),
            {"kind", "action"},
        )
        self.assertNotIn("rationale", decision["decision"])
        generation = worker.dispatch(
            "GENERATE",
            {
                "session_id": session_id,
                "turn": 2,
                "observations": [],
                "attempt": 1,
            },
        )
        self.assertEqual(generation["turn"], 2)

    def test_wrong_response_request_id_poison_transport(self):
        script = (
            "import json,sys;"
            "r=json.loads(sys.stdin.readline());"
            "print(json.dumps({"
            f"'protocol':'{AGENT_PROTOCOL}',"
            "'request_id':'wrong-id','ok':True,"
            "'result':{'protocol':"
            f"'{AGENT_PROTOCOL}',"
            "'operations':['START','DECIDE','GENERATE','CLOSE','PING','SHUTDOWN'],"
            "'endpoint_role':'fake',"
            "'reasoning_surface':'typed_decision_and_candidate_only'}}),flush=True)"
        )
        endpoint = SubprocessAgentEndpoint(
            [sys.executable, "-c", script],
            request_timeout=1.0,
        )
        try:
            with self.assertRaises(AgentProtocolError) as caught:
                endpoint.ping()
            self.assertIn(
                "request_id mismatch",
                str(caught.exception),
            )
            with self.assertRaises(AgentProtocolClosed):
                endpoint.ping()
        finally:
            endpoint.shutdown()

    def test_agent_timeout_terminates_transport_without_retry(self):
        script = "import sys,time; sys.stdin.readline(); time.sleep(30)"
        endpoint = SubprocessAgentEndpoint(
            [sys.executable, "-c", script],
            request_timeout=0.1,
        )
        try:
            with self.assertRaises(AgentProtocolTimeout):
                endpoint.ping()
            with self.assertRaises(AgentProtocolClosed):
                endpoint.ping()
        finally:
            endpoint.shutdown()

    def test_agent_stderr_is_not_reflected_to_controller(self):
        script = (
            "import sys;"
            "sys.stdin.readline();"
            "print('HIDDEN_AGENT_REASONING',file=sys.stderr,flush=True);"
            "sys.exit(7)"
        )
        endpoint = SubprocessAgentEndpoint(
            [sys.executable, "-c", script],
            request_timeout=1.0,
        )
        try:
            with self.assertRaises(AgentProtocolClosed) as caught:
                endpoint.ping()
            self.assertNotIn(
                "HIDDEN_AGENT_REASONING",
                str(caught.exception),
            )
        finally:
            endpoint.shutdown()

    def test_subprocess_agent_and_private_host_run_end_to_end(self):
        pack = PrivateHoldoutPack.load(
            REFERENCE_PACK,
            expected_runtime_id="reference-v1",
        )
        task = pack.task("hg_f8")
        with (
            SubprocessHostGateway(
                _host_command(REFERENCE_PACK),
                cwd=ROOT,
                env=_python_env(),
            ) as host,
            SubprocessAgentEndpoint(
                _agent_command(),
                cwd=ROOT,
                env=_python_env(),
            ) as agent,
        ):
            gym = ReferenceGym(host_gateway=host)
            isolated = run_isolated_episode(
                gym,
                task,
                agent,
            )
            second = run_isolated_episode(
                gym,
                task,
                agent,
            )
            self.assertTrue(isolated.accepted)
            self.assertTrue(second.accepted)
            self.assertTrue(isolated.agent_transcript_valid)
            self.assertTrue(second.agent_transcript_valid)
            self.assertNotEqual(
                isolated.agent_transcript_root_sha256,
                second.agent_transcript_root_sha256,
            )
            self.assertFalse(isolated.episode.learning_updated)
            self.assertFalse(second.episode.learning_updated)
            self.assertTrue(host.ping())
            self.assertTrue(agent.ping())


if __name__ == "__main__":
    unittest.main()
