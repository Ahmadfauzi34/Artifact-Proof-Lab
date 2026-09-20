from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import unittest

from gym.core import (
    AgentTaskView,
    ProofBundleVerifier,
    ReferenceGym,
    TerminalReason,
)
from gym.host import create_environment
from gym.host_boundary import LocalReferenceHost
from gym.io import load_task
from gym.ledger import LEDGER_FORMAT, verify_ledger_document
from gym.reference_policy import ReferenceGenerator, ReferencePolicy


ROOT = Path(__file__).resolve().parents[1]
GYM_ROOT = ROOT / "gym"


def ledger_document(result):
    return {
        "format": LEDGER_FORMAT,
        "root_sha256": result.ledger_root_sha256,
        "entry_count": result.ledger_entry_count,
        "entries": [entry.to_dict() for entry in result.ledger_entries],
    }


class GymG2Tests(unittest.TestCase):
    def setUp(self):
        self.policy = ReferencePolicy()
        self.generator = ReferenceGenerator()

    def test_local_host_returns_only_sanitized_agent_surface(self):
        task = load_task(GYM_ROOT / "tasks" / "train" / "tg_a7.json")
        host = LocalReferenceHost(create_environment)
        start = host.start(task)
        try:
            self.assertIsInstance(start.agent_task, AgentTaskView)
            self.assertFalse(hasattr(start.agent_task, "split"))
            self.assertFalse(hasattr(start.agent_task, "skill_targets"))
            self.assertFalse(hasattr(start.agent_task, "environment"))
            self.assertFalse(hasattr(start.agent_task, "metadata"))
            self.assertTrue(start.session_id.startswith("session-"))
            self.assertGreater(len(start.observations), 0)
        finally:
            host.close(start.session_id)
        with self.assertRaises(KeyError):
            host.semantic_verdict(start.session_id)

    def test_explicit_host_gateway_replaces_environment_factory(self):
        task = load_task(GYM_ROOT / "tasks" / "train" / "tg_b3.json")
        host = LocalReferenceHost(create_environment)
        gym = ReferenceGym(host_gateway=host)
        result = gym.run(task, self.policy, generator=self.generator)
        self.assertTrue(result.accepted)
        self.assertTrue(result.admission.trajectory_valid)

    def test_policy_receives_agent_task_view_through_host_boundary(self):
        task = load_task(GYM_ROOT / "tasks" / "validation" / "vg_d2.json")
        seen = []

        class SpyPolicy:
            def decide(self, task_view, observations, trajectory):
                seen.append(task_view)
                return ReferencePolicy().decide(task_view, observations, trajectory)

        result = ReferenceGym(create_environment).run(task, SpyPolicy())
        self.assertTrue(result.accepted)
        self.assertEqual(len(seen), 1)
        self.assertIsInstance(seen[0], AgentTaskView)
        self.assertFalse(hasattr(seen[0], "split"))

    def test_episode_ledger_binds_observation_decision_result_and_terminal(self):
        task = load_task(GYM_ROOT / "tasks" / "train" / "tg_a7.json")
        result = ReferenceGym(create_environment).run(task, self.policy)
        document = ledger_document(result)
        valid, reason = verify_ledger_document(document)
        self.assertTrue(valid, reason)

        kinds = [entry["kind"] for entry in document["entries"]]
        self.assertEqual(kinds[0], "EPISODE_START")
        self.assertEqual(kinds[1], "OBSERVATION_BATCH")
        self.assertIn("DECISION", kinds)
        self.assertIn("ACTION_RESULT", kinds)
        self.assertEqual(kinds[-1], "TERMINAL")

        decision = next(entry for entry in document["entries"] if entry["kind"] == "DECISION")
        action_result = next(entry for entry in document["entries"] if entry["kind"] == "ACTION_RESULT")
        self.assertEqual(action_result["payload"]["decision_entry_hash"], decision["entry_hash"])
        self.assertEqual(document["root_sha256"], document["entries"][-1]["entry_hash"])
        self.assertEqual(
            document["entries"][-1]["payload"]["terminal_reason"],
            TerminalReason.ENVIRONMENT_DONE.value,
        )

    def test_ledger_tamper_fails_closed(self):
        task = load_task(GYM_ROOT / "tasks" / "train" / "tg_a7.json")
        result = ReferenceGym(create_environment).run(task, self.policy)
        document = ledger_document(result)
        tampered = deepcopy(document)
        decision = next(entry for entry in tampered["entries"] if entry["kind"] == "DECISION")
        decision["payload"]["decision"]["rationale"] = "rewritten-after-the-fact"

        valid, reason = verify_ledger_document(tampered)
        self.assertFalse(valid)
        self.assertIn("payload hash", reason)

    def test_proof_bundle_rejects_tampered_ledger_before_integrity_pass(self):
        task = load_task(GYM_ROOT / "tasks" / "validation" / "vg_d2.json")
        result = ReferenceGym(create_environment).run(task, self.policy)
        tampered = deepcopy(ledger_document(result))
        tampered["root_sha256"] = "0" * 64

        valid, report = ProofBundleVerifier().verify(
            task,
            result.observations,
            semantic_valid=True,
            ledger_document=tampered,
        )
        self.assertFalse(valid)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("trajectory ledger rejected", report["reason"])

    def test_step_budget_exhaustion_has_explicit_terminal_reason(self):
        task = load_task(GYM_ROOT / "tasks" / "train" / "tg_b3.json")
        task = replace(task, budget={**task.budget, "max_steps": 1})
        result = ReferenceGym(create_environment).run(task, self.policy)
        self.assertFalse(result.accepted)
        self.assertTrue(result.budget_exhausted)
        self.assertEqual(result.terminal_reason, TerminalReason.STEP_BUDGET_EXHAUSTED.value)

    def test_generative_budget_exhaustion_has_explicit_terminal_reason(self):
        task = load_task(GYM_ROOT / "tasks" / "validation" / "vg_e4.json")
        task = replace(task, budget={**task.budget, "max_generative_calls": 1})
        result = ReferenceGym(create_environment).run(task, self.policy, generator=self.generator)
        self.assertFalse(result.accepted)
        self.assertTrue(result.budget_exhausted)
        self.assertEqual(result.generative_calls, 1)
        self.assertEqual(
            result.terminal_reason,
            TerminalReason.GENERATIVE_BUDGET_EXHAUSTED.value,
        )

    def test_tool_budget_exhaustion_has_explicit_terminal_reason(self):
        task = load_task(GYM_ROOT / "tasks" / "train" / "tg_a7.json")
        task = replace(task, budget={**task.budget, "max_tool_cost": 0.1})
        result = ReferenceGym(create_environment).run(task, self.policy)
        self.assertFalse(result.accepted)
        self.assertEqual(
            result.terminal_reason,
            TerminalReason.TOOL_COST_BUDGET_EXHAUSTED.value,
        )


if __name__ == "__main__":
    unittest.main()
