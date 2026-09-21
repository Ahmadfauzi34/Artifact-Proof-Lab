from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from gym.adaptive_policy import ProofGatedAdaptivePolicy
from gym.core import ReferenceGym
from gym.host import create_environment
from gym.io import load_task, load_tasks
from gym.on_policy import BoundedExplorationPolicy, OnPolicyTrainer
from gym.reference_policy import ReferenceGenerator, ReferencePolicy


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "gym" / "tasks"


class GymG41Tests(unittest.TestCase):
    def setUp(self):
        self.gym = ReferenceGym(create_environment)
        self.generator = ReferenceGenerator()

    def test_failed_search_changes_cursor_not_q(self):
        task = load_task(TASK_ROOT / "train" / "tg_b3.json")
        learner = ProofGatedAdaptivePolicy()
        policy = BoundedExplorationPolicy(learner)
        q0 = learner.policy_sha256
        s0 = policy.search_sha256

        first = self.gym.run(
            task,
            policy,
            generator=self.generator,
            learning_sink=learner,
        )
        self.assertFalse(first.accepted)
        self.assertEqual(learner.policy_sha256, q0)
        self.assertEqual(len(learner.learning_receipts), 0)

        self.assertTrue(policy.observe_failed_attempt(first))
        self.assertEqual(learner.policy_sha256, q0)
        self.assertNotEqual(policy.search_sha256, s0)

        second = self.gym.run(
            task,
            policy,
            generator=self.generator,
            learning_sink=learner,
        )
        self.assertTrue(second.accepted)
        self.assertTrue(second.learning_updated)
        self.assertNotEqual(learner.policy_sha256, q0)

    def test_invalid_failure_cannot_move_search_cursor(self):
        task = load_task(TASK_ROOT / "train" / "tg_b3.json")
        learner = ProofGatedAdaptivePolicy()
        policy = BoundedExplorationPolicy(learner)
        result = self.gym.run(
            task,
            policy,
            generator=self.generator,
            learning_sink=learner,
        )
        self.assertFalse(result.accepted)

        bad = replace(
            result,
            admission=replace(result.admission, integrity_valid=False),
        )
        before = policy.search_sha256
        self.assertFalse(policy.observe_failed_attempt(bad))
        self.assertEqual(policy.search_sha256, before)

    def test_reference_policy_is_not_training_actor(self):
        learner = ProofGatedAdaptivePolicy()
        policy = BoundedExplorationPolicy(learner)
        trainer = OnPolicyTrainer(
            self.gym,
            policy,
            generator=self.generator,
            max_attempts_per_task=6,
        )
        train = [task for task in load_tasks(TASK_ROOT) if task.split == "train"]

        with patch.object(
            ReferencePolicy,
            "decide",
            side_effect=AssertionError("teacher used"),
        ):
            results = trainer.train(train)

        self.assertTrue(all(item.accepted for item in results))
        self.assertEqual(len(learner.learning_receipts), len(train))
        self.assertGreater(len(policy.search_receipts), 0)

    def test_reference_attempt_profile(self):
        learner = ProofGatedAdaptivePolicy()
        policy = BoundedExplorationPolicy(learner)
        trainer = OnPolicyTrainer(
            self.gym,
            policy,
            generator=self.generator,
            max_attempts_per_task=6,
        )
        train = [task for task in load_tasks(TASK_ROOT) if task.split == "train"]
        results = trainer.train(train)

        observed = {item.task_id: item.attempts for item in results}
        self.assertEqual(
            observed,
            {
                "tg_a7": 1,
                "tg_b3": 2,
                "tg_c9": 3,
                "tg_d4": 3,
                "tg_e6": 2,
                "tg_f2": 1,
                "tg_g5": 2,
                "tg_h8": 1,
                "tg_i1": 1,
            },
        )
        self.assertEqual(sum(item.attempts for item in results), 16)
        self.assertEqual(len(policy.search_receipts), 7)
        self.assertEqual(learner.learned_pair_count, 11)

    def test_post_training_validation_is_read_only(self):
        learner = ProofGatedAdaptivePolicy()
        policy = BoundedExplorationPolicy(learner)
        trainer = OnPolicyTrainer(
            self.gym,
            policy,
            generator=self.generator,
            max_attempts_per_task=6,
        )
        train = [task for task in load_tasks(TASK_ROOT) if task.split == "train"]
        validation = [
            task for task in load_tasks(TASK_ROOT)
            if task.split == "validation"
        ]
        trainer.train(train)

        q_hash = learner.policy_sha256
        search_hash = policy.search_sha256
        results = [
            self.gym.run(task, learner, generator=self.generator)
            for task in validation
        ]

        self.assertTrue(all(result.accepted for result in results))
        self.assertTrue(all(not result.learning_updated for result in results))
        self.assertEqual(learner.policy_sha256, q_hash)
        self.assertEqual(policy.search_sha256, search_hash)


if __name__ == "__main__":
    unittest.main()
