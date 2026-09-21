from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

from gym.adaptive_policy import (
    ProofGatedAdaptivePolicy,
    SemanticStateEncoder,
)
from gym.core import AgentTaskView, ReferenceGym
from gym.host import create_environment
from gym.io import load_task, load_tasks
from gym.reference_policy import ReferenceGenerator, ReferencePolicy


ROOT = Path(__file__).resolve().parents[1]
GYM_ROOT = ROOT / "gym"
TASK_ROOT = GYM_ROOT / "tasks"


class GymG4Tests(unittest.TestCase):
    def setUp(self):
        self.gym = ReferenceGym(create_environment)
        self.generator = ReferenceGenerator()

    def _train(self):
        learner = ProofGatedAdaptivePolicy()
        teacher = ReferencePolicy()
        train_tasks = [
            task
            for task in load_tasks(TASK_ROOT)
            if task.split == "train"
        ]
        results = [
            self.gym.run(
                task,
                teacher,
                generator=self.generator,
                learning_sink=learner,
            )
            for task in train_tasks
        ]
        return learner, train_tasks, results

    def test_untrained_policy_is_conservative(self):
        learner = ProofGatedAdaptivePolicy()
        task = load_task(
            TASK_ROOT / "validation" / "vg_g9.json"
        )
        result = self.gym.run(
            task,
            learner,
            generator=self.generator,
        )
        self.assertFalse(result.accepted)
        self.assertEqual(
            result.trajectory,
            ("DEFER_NO_SUPPORT",),
        )
        self.assertFalse(result.learning_updated)

    def test_train_episodes_change_policy_state_and_emit_receipts(self):
        learner = ProofGatedAdaptivePolicy()
        initial = learner.policy_sha256
        learner, train_tasks, results = self._train()

        self.assertNotEqual(learner.policy_sha256, initial)
        self.assertEqual(
            len(learner.learning_receipts),
            len(train_tasks),
        )
        self.assertTrue(
            all(result.accepted for result in results)
        )
        self.assertTrue(
            all(result.learning_updated for result in results)
        )
        self.assertTrue(
            all(
                receipt.ledger_root_sha256
                == result.ledger_root_sha256
                for receipt, result in zip(
                    learner.learning_receipts,
                    results,
                )
            )
        )

    def test_policy_document_contains_no_task_identity_or_description(self):
        learner, _, _ = self._train()
        encoded = json.dumps(
            learner.policy_document(),
            sort_keys=True,
        )
        self.assertNotIn("tg_", encoded)
        self.assertNotIn("task_id", encoded)
        self.assertNotIn("description", encoded)
        self.assertNotIn("snapshot_id", encoded)

    def test_validation_improves_after_training_without_learning(self):
        validation = [
            task
            for task in load_tasks(TASK_ROOT)
            if task.split == "validation"
        ]

        learner = ProofGatedAdaptivePolicy()
        before = [
            self.gym.run(
                task,
                learner,
                generator=self.generator,
            )
            for task in validation
        ]
        before_rate = (
            sum(result.accepted for result in before)
            / len(before)
        )

        teacher = ReferencePolicy()
        for task in load_tasks(TASK_ROOT):
            if task.split == "train":
                self.gym.run(
                    task,
                    teacher,
                    generator=self.generator,
                    learning_sink=learner,
                )

        trained_hash = learner.policy_sha256
        after = [
            self.gym.run(
                task,
                learner,
                generator=self.generator,
            )
            for task in validation
        ]
        after_rate = (
            sum(result.accepted for result in after)
            / len(after)
        )

        self.assertGreater(after_rate, before_rate)
        self.assertTrue(
            all(result.accepted for result in after)
        )
        self.assertTrue(
            all(not result.learning_updated for result in after)
        )
        self.assertEqual(
            learner.policy_sha256,
            trained_hash,
        )

    def test_cross_domain_authority_transfer_reaches_holdout(self):
        holdout = load_task(
            TASK_ROOT / "holdout" / "hg_f8.json"
        )

        learner = ProofGatedAdaptivePolicy()
        before = self.gym.run(
            holdout,
            learner,
            generator=self.generator,
        )
        self.assertFalse(before.accepted)

        teacher = ReferencePolicy()
        for name in ("tg_b3.json", "tg_e6.json"):
            task = load_task(
                TASK_ROOT / "train" / name
            )
            self.gym.run(
                task,
                teacher,
                learning_sink=learner,
            )

        trained_hash = learner.policy_sha256
        after = self.gym.run(
            holdout,
            learner,
            generator=self.generator,
        )

        self.assertTrue(after.accepted, after.admission.reason)
        self.assertEqual(
            after.trajectory,
            (
                "INSPECT_CONFIG_ORIGIN",
                "SELECT_PROJECT_CONFIG",
            ),
        )
        self.assertFalse(after.learning_updated)
        self.assertEqual(
            learner.policy_sha256,
            trained_hash,
        )

    def test_source_b_validation_is_not_position_memorization(self):
        learner = ProofGatedAdaptivePolicy()
        teacher = ReferencePolicy()
        for name in ("tg_b3.json", "tg_e6.json"):
            task = load_task(
                TASK_ROOT / "train" / name
            )
            self.gym.run(
                task,
                teacher,
                learning_sink=learner,
            )

        validation = load_task(
            TASK_ROOT / "validation" / "vg_g9.json"
        )
        result = self.gym.run(
            validation,
            learner,
        )
        self.assertTrue(result.accepted)
        self.assertEqual(
            result.trajectory,
            (
                "CHECK_PROVENANCE",
                "SELECT_SOURCE_B",
            ),
        )

    def test_learning_sink_rejects_non_train_and_unadmitted_evidence(self):
        learner = ProofGatedAdaptivePolicy()
        teacher = ReferencePolicy()
        train = load_task(
            TASK_ROOT / "train" / "tg_b3.json"
        )
        result = self.gym.run(
            train,
            teacher,
        )
        self.assertTrue(result.accepted)

        with self.assertRaises(ValueError):
            learner.update(
                replace(result, split="validation")
            )
        with self.assertRaises(ValueError):
            learner.update(
                replace(
                    result,
                    admission=replace(
                        result.admission,
                        admitted=False,
                    ),
                )
            )

    def test_learning_sink_reverifies_trajectory_ledger(self):
        learner = ProofGatedAdaptivePolicy()
        task = load_task(
            TASK_ROOT / "train" / "tg_b3.json"
        )
        result = self.gym.run(
            task,
            ReferencePolicy(),
        )
        self.assertTrue(result.accepted)

        entries = list(result.ledger_entries)
        decision_index = next(
            index
            for index, entry in enumerate(entries)
            if entry.kind == "DECISION"
        )
        entries[decision_index] = replace(
            entries[decision_index],
            payload={
                **entries[decision_index].payload,
                "decision": {
                    "kind": "SELECT",
                    "action": "SELECT_SOURCE_B",
                },
            },
        )
        tampered = replace(
            result,
            ledger_entries=tuple(entries),
        )

        with self.assertRaises(ValueError) as caught:
            learner.update(tampered)
        self.assertIn(
            "trajectory ledger",
            str(caught.exception),
        )

    def test_state_encoder_ignores_task_identity_and_wording(self):
        encoder = SemanticStateEncoder()
        task_a = load_task(
            TASK_ROOT / "train" / "tg_b3.json"
        )
        env = create_environment(task_a)
        try:
            observations = env.reset()
        finally:
            env.close()

        view_a = task_a.agent_view()
        view_b = AgentTaskView(
            task_id="unseen-task",
            domain=view_a.domain,
            description="completely different wording",
            allowed_actions=view_a.allowed_actions,
            budget=view_a.budget,
        )
        self.assertEqual(
            encoder.encode(
                view_a.domain,
                observations,
            ),
            encoder.encode(
                view_b.domain,
                observations,
            ),
        )

    def test_adaptive_policy_schema_parses(self):
        schema = json.loads(
            (
                GYM_ROOT
                / "contracts"
                / "adaptive_policy.schema.json"
            ).read_text()
        )
        self.assertEqual(
            schema["title"],
            "Proof-Gated Adaptive Policy State",
        )


if __name__ == "__main__":
    unittest.main()
