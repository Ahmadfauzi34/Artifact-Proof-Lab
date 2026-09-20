from pathlib import Path
import unittest

from gym.core import MemoryLearningSink, ReferenceGym
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

    def test_baseline_contracts_and_public_tasks_validate(self):
        self.assertEqual(validate(GYM_ROOT), [])

    def test_three_domain_reference_tasks_execute_end_to_end(self):
        for name in ("tg_a7.json", "tg_b3.json", "tg_c9.json"):
            with self.subTest(name=name):
                task = load_task(GYM_ROOT / "tasks" / "train" / name)
                result = self.gym.run(task, self.policy, generator=self.generator)
                self.assertTrue(result.accepted, result.admission.reason)
                self.assertTrue(result.admission.provenance_valid)
                self.assertTrue(result.admission.integrity_valid)

    def test_holdout_never_updates_learning_sink(self):
        task = load_task(GYM_ROOT / "tasks" / "holdout" / "hg_f8.json")
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
        self.assertEqual(
            result.trajectory,
            ("APPLY_CANDIDATE_FIX", "RUN_TARGETED_TEST", "APPLY_CANDIDATE_FIX", "RUN_REGRESSION"),
        )

    def test_all_reference_tasks_keep_split_metrics_separable(self):
        rows = []
        for task in load_tasks(GYM_ROOT / "tasks"):
            rows.append(self.gym.run(task, self.policy, generator=self.generator))
        self.assertEqual({row.split for row in rows}, {"train", "validation", "holdout"})
        self.assertTrue(all(row.accepted for row in rows))


if __name__ == "__main__":
    unittest.main()
