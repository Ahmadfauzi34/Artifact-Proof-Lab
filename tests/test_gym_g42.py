from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from gym.adaptive_policy import ProofGatedAdaptivePolicy
from gym.core import ReferenceGym, TerminalReason
from gym.host import create_environment
from gym.io import load_task, load_tasks
from gym.negative_experience import (
    NegativeAwareExplorationPolicy,
    NegativeExperienceTrainer,
)
from gym.reference_policy import ReferenceGenerator, ReferencePolicy


ROOT = Path(__file__).resolve().parents[1]
GYM_ROOT = ROOT / "gym"
TASK_ROOT = GYM_ROOT / "tasks"


class GymG42Tests(unittest.TestCase):
    def setUp(self):
        self.gym = ReferenceGym(create_environment)
        self.generator = ReferenceGenerator()

    def _first_failure(self):
        learner = ProofGatedAdaptivePolicy()
        policy = NegativeAwareExplorationPolicy(learner)
        task = load_task(TASK_ROOT / "train" / "tg_b3.json")
        result = self.gym.run(
            task,
            policy,
            generator=self.generator,
            learning_sink=learner,
        )
        self.assertFalse(result.accepted)
        return learner, policy, result

    def test_semantic_rejection_updates_negative_not_q(self):
        learner, policy, result = self._first_failure()
        q0 = learner.policy_sha256
        n0 = policy.negative_sha256

        receipt = policy.negative.observe(
            result,
            learner=learner,
        )

        self.assertIsNotNone(receipt)
        self.assertEqual(learner.policy_sha256, q0)
        self.assertNotEqual(policy.negative_sha256, n0)
        self.assertEqual(receipt.state, "AUTHORITY_SLOT_1")
        self.assertEqual(
            receipt.intent,
            "SELECT_AUTHORITY_SLOT_2",
        )
        self.assertEqual(
            policy.negative.rejection_count(
                "AUTHORITY_UNKNOWN",
                "INSPECT_AUTHORITY",
            ),
            0,
        )

    def test_invalid_or_budget_failure_cannot_train_negative(self):
        learner, policy, result = self._first_failure()
        n0 = policy.negative_sha256

        invalid = replace(
            result,
            admission=replace(
                result.admission,
                integrity_valid=False,
            ),
        )
        self.assertIsNone(
            policy.negative.observe(
                invalid,
                learner=learner,
            )
        )

        budget = replace(
            result,
            budget_exhausted=True,
            terminal_reason=
                TerminalReason.TOOL_COST_BUDGET_EXHAUSTED.value,
        )
        self.assertIsNone(
            policy.negative.observe(
                budget,
                learner=learner,
            )
        )
        self.assertEqual(policy.negative_sha256, n0)

    def test_tampered_ledger_is_rejected_independently(self):
        learner, policy, result = self._first_failure()
        entries = list(result.ledger_entries)
        index = next(
            index
            for index, entry in enumerate(entries)
            if entry.kind == "DECISION"
        )
        entries[index] = replace(
            entries[index],
            payload={
                **entries[index].payload,
                "decision": {
                    "kind": "SELECT",
                    "action": "SELECT_SOURCE_A",
                },
            },
        )
        tampered = replace(
            result,
            ledger_entries=tuple(entries),
        )

        self.assertIsNone(
            policy.negative.observe(
                tampered,
                learner=learner,
            )
        )
        self.assertEqual(len(policy.negative.receipts), 0)

    def test_negative_memory_transfers_without_positive_q(self):
        learner, policy, failed = self._first_failure()
        receipt = policy.negative.observe(
            failed,
            learner=learner,
        )
        self.assertIsNotNone(receipt)
        self.assertEqual(len(learner.learning_receipts), 0)

        fresh = ProofGatedAdaptivePolicy()
        transfer = NegativeAwareExplorationPolicy(
            fresh,
            policy.negative,
        )
        holdout = load_task(
            TASK_ROOT / "holdout" / "hg_f8.json"
        )
        result = self.gym.run(
            holdout,
            transfer,
            generator=self.generator,
        )

        self.assertTrue(result.accepted, result.admission.reason)
        self.assertEqual(
            result.trajectory,
            (
                "INSPECT_CONFIG_ORIGIN",
                "SELECT_PROJECT_CONFIG",
            ),
        )
        self.assertEqual(fresh.learned_pair_count, 0)

    def test_negative_state_contains_no_task_identity(self):
        learner, policy, failed = self._first_failure()
        policy.negative.observe(
            failed,
            learner=learner,
        )
        encoded = json.dumps(
            policy.negative.document(),
            sort_keys=True,
        )
        self.assertNotIn("tg_b3", encoded)
        self.assertNotIn("task_id", encoded)
        self.assertNotIn("snapshot_id", encoded)

    def test_reference_policy_is_not_training_actor(self):
        learner = ProofGatedAdaptivePolicy()
        policy = NegativeAwareExplorationPolicy(learner)
        trainer = NegativeExperienceTrainer(
            self.gym,
            policy,
            generator=self.generator,
            max_attempts_per_task=6,
        )
        train = [
            task
            for task in load_tasks(TASK_ROOT)
            if task.split == "train"
        ]

        with patch.object(
            ReferencePolicy,
            "decide",
            side_effect=AssertionError("teacher used"),
        ):
            results = trainer.train(train)

        self.assertTrue(all(item.accepted for item in results))
        self.assertEqual(len(learner.learning_receipts), 9)
        self.assertEqual(len(policy.negative.receipts), 7)

    def test_reference_profile_and_schema(self):
        learner = ProofGatedAdaptivePolicy()
        policy = NegativeAwareExplorationPolicy(learner)
        trainer = NegativeExperienceTrainer(
            self.gym,
            policy,
            generator=self.generator,
            max_attempts_per_task=6,
        )
        train = [
            task
            for task in load_tasks(TASK_ROOT)
            if task.split == "train"
        ]
        results = trainer.train(train)

        self.assertTrue(all(item.accepted for item in results))
        self.assertEqual(sum(item.attempts for item in results), 16)
        self.assertEqual(len(policy.negative.receipts), 7)
        self.assertEqual(len(learner.learning_receipts), 9)
        self.assertEqual(learner.learned_pair_count, 11)

        schema = json.loads(
            (
                GYM_ROOT
                / "contracts"
                / "negative_experience.schema.json"
            ).read_text()
        )
        self.assertEqual(
            schema["title"],
            "Proof-Gated Typed Negative Experience State",
        )


if __name__ == "__main__":
    unittest.main()
