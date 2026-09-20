from dataclasses import replace
from pathlib import Path
import builtins
import unittest
from unittest.mock import patch

from gym.core import (
    EvidenceAdmission,
    MemoryLearningSink,
    Observation,
    ProofBundleVerifier,
    ReferenceGym,
)
from gym.host import create_environment
from gym.io import load_task, load_tasks
from gym.reference_policy import ReferenceGenerator, ReferencePolicy
from gym.validate_baseline import validate


ROOT = Path(__file__).resolve().parents[1]
GYM_ROOT = ROOT / "gym"


class GymBaselineTests(unittest.TestCase):
    def setUp(self):
        self.gym = ReferenceGym(create_environment)
        self.policy = ReferencePolicy()
        self.generator = ReferenceGenerator()

    def test_baseline_contracts_and_host_tasks_validate(self):
        self.assertEqual(validate(GYM_ROOT), [])

    def test_agent_view_hides_host_only_benchmark_fields(self):
        task = load_task(GYM_ROOT / "tasks" / "train" / "tg_a7.json")
        view = task.agent_view()
        self.assertFalse(hasattr(view, "split"))
        self.assertFalse(hasattr(view, "skill_targets"))
        self.assertFalse(hasattr(view, "environment"))
        self.assertFalse(hasattr(view, "metadata"))
        self.assertEqual(view.task_id, task.task_id)

    def test_three_train_reference_tasks_execute_end_to_end(self):
        for name in ("tg_a7.json", "tg_b3.json", "tg_c9.json"):
            with self.subTest(name=name):
                task = load_task(GYM_ROOT / "tasks" / "train" / name)
                result = self.gym.run(task, self.policy, generator=self.generator)
                self.assertTrue(result.accepted, result.admission.reason)
                self.assertTrue(result.admission.provenance_valid)
                self.assertTrue(result.admission.integrity_valid)

    def test_holdout_uses_unseen_reference_domain_and_never_updates_learning_sink(self):
        train_domains = {task.domain for task in load_tasks(GYM_ROOT / "tasks") if task.split == "train"}
        task = load_task(GYM_ROOT / "tasks" / "holdout" / "hg_f8.json")
        self.assertNotIn(task.domain, train_domains)
        sink = MemoryLearningSink()
        result = self.gym.run(task, self.policy, generator=self.generator, learning_sink=sink)
        self.assertTrue(result.accepted)
        self.assertFalse(result.learning_updated)
        self.assertEqual(sink.updates, [])

    def test_admitted_train_task_updates_learning_sink(self):
        task = load_task(GYM_ROOT / "tasks" / "train" / "tg_b3.json")
        sink = MemoryLearningSink()
        result = self.gym.run(task, self.policy, generator=self.generator, learning_sink=sink)
        self.assertTrue(result.accepted)
        self.assertTrue(result.learning_updated)
        self.assertEqual(sink.updates, [task.task_id])

    def test_tool_cost_budget_is_enforced(self):
        task = load_task(GYM_ROOT / "tasks" / "train" / "tg_a7.json")
        task = replace(task, budget={**task.budget, "max_tool_cost": 0.1})
        sink = MemoryLearningSink()
        result = self.gym.run(task, self.policy, learning_sink=sink)
        self.assertFalse(result.accepted)
        self.assertTrue(result.budget_exhausted)
        self.assertIn("BUDGET_EXHAUSTED", result.trajectory)
        self.assertEqual(sink.updates, [])

    def test_safe_defer_is_typed_and_can_be_correct(self):
        task = load_task(GYM_ROOT / "tasks" / "validation" / "vg_d2.json")
        result = self.gym.run(task, self.policy)
        self.assertTrue(result.accepted)
        self.assertEqual(result.trajectory, ("DEFER_NO_ADMISSIBLE",))

    def test_generated_candidate_is_rejected_before_valid_candidate(self):
        task = load_task(GYM_ROOT / "tasks" / "validation" / "vg_e4.json")
        result = self.gym.run(task, self.policy, generator=self.generator)
        self.assertTrue(result.accepted)
        self.assertEqual(result.generative_calls, 2)
        self.assertEqual(result.generated_candidates, ("key=guess", "key=stable-42"))
        candidate_evidence = [o for o in result.observations if o.check_id == "generated-candidate"]
        self.assertEqual([o.status for o in candidate_evidence], ["FAIL", "PASS"])

    def test_environment_reset_is_deterministic(self):
        task = load_task(GYM_ROOT / "tasks" / "train" / "tg_a7.json")
        first = create_environment(task).reset()
        second = create_environment(task).reset()
        self.assertEqual(first, second)

    def test_wrong_action_can_recover_without_provenance_aliasing(self):
        task = load_task(GYM_ROOT / "tasks" / "train" / "tg_c9.json")

        class WrongFirstPolicy:
            def decide(self, task, observations, trajectory):
                if not trajectory:
                    from gym.core import PolicyDecision
                    return PolicyDecision.select("APPLY_CANDIDATE_FIX")
                return ReferencePolicy().decide(task, observations, trajectory)

        result = self.gym.run(task, WrongFirstPolicy(), generator=self.generator)
        self.assertTrue(result.accepted, result.admission.reason)
        patch_attempts = [o.attempt_id for o in result.observations if o.check_id == "patch"]
        self.assertEqual(patch_attempts, [1, 2])

    def test_same_check_id_cannot_alias_different_probe_or_source(self):
        observations = (
            Observation("o1", "root-a", "same", "probe-a", 1, 0, 0, "x", "FAIL", 1),
            Observation("o2", "root-a", "same", "probe-b", 1, 1, 1, "x", "PASS", 2),
        )
        valid, reason = EvidenceAdmission.provenance_valid(observations)
        self.assertFalse(valid)
        self.assertIn("aliases", reason)

    def test_reward_channels_keep_semantic_success_and_proof_closure_distinct(self):
        task = load_task(GYM_ROOT / "tasks" / "validation" / "vg_d2.json")
        result = self.gym.run(task, self.policy)
        self.assertEqual(result.reward_channels["task_acceptance"], 1.0)
        self.assertEqual(result.reward_channels["proof_closure"], 1.0)
        self.assertEqual(result.reward_channels["admission_credit"], 1.0)

    def test_missing_artifact_proof_fails_closed(self):
        task = load_task(GYM_ROOT / "tasks" / "validation" / "vg_d2.json")
        observations = create_environment(task).reset()
        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "artifact_proof":
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=blocked_import):
            valid, report = ProofBundleVerifier().verify(task, observations, semantic_valid=True)
        self.assertFalse(valid)
        self.assertEqual(report["status"], "FAIL")

    def test_all_reference_tasks_keep_split_metrics_separable(self):
        rows = [self.gym.run(task, self.policy, generator=self.generator) for task in load_tasks(GYM_ROOT / "tasks")]
        self.assertEqual({row.split for row in rows}, {"train", "validation", "holdout"})
        self.assertTrue(all(row.accepted for row in rows))


if __name__ == "__main__":
    unittest.main()
