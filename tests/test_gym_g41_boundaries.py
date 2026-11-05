from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

from gym.adaptive_policy import ProofGatedAdaptivePolicy
from gym.core import ReferenceGym
from gym.host import create_environment
from gym.io import load_task
from gym.on_policy import BoundedExplorationPolicy
from gym.reference_policy import ReferenceGenerator

ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "gym" / "tasks"


class GymG41BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.gym = ReferenceGym(create_environment)
        self.generator = ReferenceGenerator()

    def _failed_first_attempt(self):
        learner = ProofGatedAdaptivePolicy()
        policy = BoundedExplorationPolicy(learner)
        result = self.gym.run(
            load_task(TASK_ROOT / "train" / "tg_b3.json"),
            policy,
            generator=self.generator,
            learning_sink=learner,
        )
        self.assertFalse(result.accepted)
        return learner, policy, result

    def test_trajectory_invalid_failure_cannot_move_search_cursor(self):
        _, policy, result = self._failed_first_attempt()
        bad = replace(result, admission=replace(result.admission, trajectory_valid=False))
        before = policy.search_sha256
        self.assertFalse(policy.observe_failed_attempt(bad))
        self.assertEqual(policy.search_sha256, before)

    def test_provenance_invalid_failure_cannot_move_search_cursor(self):
        _, policy, result = self._failed_first_attempt()
        bad = replace(result, admission=replace(result.admission, provenance_valid=False))
        before = policy.search_sha256
        self.assertFalse(policy.observe_failed_attempt(bad))
        self.assertEqual(policy.search_sha256, before)


if __name__ == "__main__":
    unittest.main()
