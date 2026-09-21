from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from gym.core import MemoryLearningSink, ReferenceGym
from gym.curriculum import (
    CurriculumError,
    load_curriculum,
    evaluate_stage,
    validate_curriculum_tasks,
    verify_promotion_chain,
)
from gym.host import create_environment
from gym.io import load_task, load_tasks
from gym.ledger import ZERO_HASH
from gym.reference_policy import ReferenceGenerator, ReferencePolicy


ROOT = Path(__file__).resolve().parents[1]
GYM_ROOT = ROOT / "gym"
TASK_ROOT = GYM_ROOT / "tasks"
CURRICULUM_PATH = GYM_ROOT / "curriculum" / "g3_reference.json"


class GymG3Tests(unittest.TestCase):
    def test_curriculum_loads_and_task_splits_are_valid(self):
        curriculum = load_curriculum(CURRICULUM_PATH)
        tasks = load_tasks(TASK_ROOT)
        validate_curriculum_tasks(curriculum, tasks)

        self.assertEqual(len(curriculum.stages), 5)
        train = [task for task in tasks if task.split == "train"]
        validation = [task for task in tasks if task.split == "validation"]
        self.assertEqual(len(train), 9)
        self.assertEqual(len(validation), 5)

    def test_hidden_scenario_is_not_in_agent_task_view(self):
        task = load_task(TASK_ROOT / "train" / "tg_d4.json")
        self.assertIn("scenario", task.environment)
        view = task.agent_view()
        self.assertFalse(hasattr(view, "environment"))
        self.assertFalse(hasattr(view, "scenario"))
        self.assertNotIn("stopped", repr(view))

    def test_stopped_process_requires_inspection_before_restart(self):
        task = load_task(TASK_ROOT / "train" / "tg_d4.json")
        env = create_environment(task)
        try:
            env.reset()
            premature = env.step("RESTART_PROCESS")
            self.assertEqual(premature.observations[0].status, "WARN")
            self.assertFalse(env.semantic_verdict())

            inspected = env.step("INSPECT_PROCESS_STATE")
            self.assertFalse(inspected.done)
            self.assertEqual(inspected.observations[0].value, "stopped")

            restarted = env.step("RESTART_PROCESS")
            self.assertEqual(restarted.observations[0].status, "PASS")

            verified = env.step("RETRY_HEALTHCHECK")
            self.assertTrue(verified.done)
            self.assertTrue(env.semantic_verdict())
        finally:
            env.close()

    def test_reference_policy_learns_no_source_a_position_bias(self):
        task = load_task(TASK_ROOT / "train" / "tg_e6.json")
        learning = MemoryLearningSink()
        result = ReferenceGym(create_environment).run(
            task,
            ReferencePolicy(),
            learning_sink=learning,
        )

        self.assertTrue(result.accepted, result.admission.reason)
        self.assertEqual(
            result.trajectory,
            ("CHECK_PROVENANCE", "SELECT_SOURCE_B"),
        )
        self.assertTrue(result.learning_updated)
        self.assertEqual(learning.updates, [task.task_id])

    def test_g3_stopped_process_reference_path_is_evidence_gated(self):
        task = load_task(TASK_ROOT / "train" / "tg_d4.json")
        result = ReferenceGym(create_environment).run(
            task,
            ReferencePolicy(),
            learning_sink=MemoryLearningSink(),
        )
        self.assertTrue(result.accepted, result.admission.reason)
        self.assertEqual(
            result.trajectory,
            (
                "INSPECT_PROCESS_STATE",
                "RESTART_PROCESS",
                "RETRY_HEALTHCHECK",
            ),
        )
        self.assertFalse(result.budget_exhausted)

    def test_validation_episode_never_updates_learning(self):
        task = load_task(TASK_ROOT / "validation" / "vg_f7.json")
        learning = MemoryLearningSink()
        result = ReferenceGym(create_environment).run(
            task,
            ReferencePolicy(),
            learning_sink=learning,
        )
        self.assertTrue(result.accepted, result.admission.reason)
        self.assertFalse(result.learning_updated)
        self.assertEqual(learning.updates, [])

    def test_curriculum_rejects_validation_learning_update(self):
        curriculum = load_curriculum(CURRICULUM_PATH)
        stage = curriculum.stages[0]
        tasks = {task.task_id: task for task in load_tasks(TASK_ROOT)}
        gym = ReferenceGym(create_environment)
        policy = ReferencePolicy()
        learning = MemoryLearningSink()

        train_results = [
            gym.run(
                tasks[task_id],
                policy,
                learning_sink=learning,
            )
            for task_id in stage.train_task_ids
        ]
        validation_results = [
            gym.run(
                tasks[task_id],
                policy,
                learning_sink=learning,
            )
            for task_id in stage.validation_task_ids
        ]
        invalid_validation = [
            replace(validation_results[0], learning_updated=True)
        ]

        with self.assertRaises(CurriculumError) as caught:
            evaluate_stage(
                curriculum,
                stage,
                train_results,
                invalid_validation,
                previous_checkpoint_sha256=ZERO_HASH,
            )
        self.assertIn("validation", str(caught.exception))

    def test_failed_stage_stops_promotion_chain(self):
        curriculum = load_curriculum(CURRICULUM_PATH)
        tasks = {task.task_id: task for task in load_tasks(TASK_ROOT)}
        stage = curriculum.stages[0]
        gym = ReferenceGym(create_environment)
        learning = MemoryLearningSink()

        train_results = [
            gym.run(
                tasks[task_id],
                ReferencePolicy(),
                learning_sink=learning,
            )
            for task_id in stage.train_task_ids
        ]
        validation_results = [
            gym.run(
                tasks[task_id],
                ReferencePolicy(),
                learning_sink=learning,
            )
            for task_id in stage.validation_task_ids
        ]
        broken_train = list(train_results)
        broken_train[0] = replace(
            broken_train[0],
            admission=replace(
                broken_train[0].admission,
                admitted=False,
            ),
            learning_updated=False,
        )
        receipt = evaluate_stage(
            curriculum,
            stage,
            broken_train,
            validation_results,
            previous_checkpoint_sha256=ZERO_HASH,
        )
        self.assertFalse(receipt.promoted)
        self.assertIn(
            "train_admission_below_threshold",
            receipt.reasons,
        )

    def test_promotion_checkpoint_tamper_is_rejected(self):
        curriculum = load_curriculum(CURRICULUM_PATH)
        tasks = {task.task_id: task for task in load_tasks(TASK_ROOT)}
        stage = curriculum.stages[0]
        gym = ReferenceGym(create_environment)
        learning = MemoryLearningSink()

        train_results = [
            gym.run(
                tasks[task_id],
                ReferencePolicy(),
                learning_sink=learning,
            )
            for task_id in stage.train_task_ids
        ]
        validation_results = [
            gym.run(
                tasks[task_id],
                ReferencePolicy(),
                learning_sink=learning,
            )
            for task_id in stage.validation_task_ids
        ]
        receipt = evaluate_stage(
            curriculum,
            stage,
            train_results,
            validation_results,
            previous_checkpoint_sha256=ZERO_HASH,
        )
        tampered = replace(
            receipt,
            train_admission_rate=0.5,
        )
        valid, reason = verify_promotion_chain(
            curriculum,
            [tampered],
        )
        self.assertFalse(valid)
        self.assertIn("hash mismatch", reason)

    def test_strict_curriculum_json_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "curriculum.json"
            path.write_text(
                '{"format":"proof-gym-curriculum-v1",'
                '"curriculum_id":"x","curriculum_id":"y",'
                '"stages":[]}'
            )
            with self.assertRaises(CurriculumError) as caught:
                load_curriculum(path)
            self.assertIn("duplicate JSON key", str(caught.exception))

    def test_full_reference_curriculum_promotes_in_order(self):
        curriculum = load_curriculum(CURRICULUM_PATH)
        tasks = load_tasks(TASK_ROOT)
        validate_curriculum_tasks(curriculum, tasks)
        by_id = {task.task_id: task for task in tasks}

        gym = ReferenceGym(create_environment)
        policy = ReferencePolicy()
        generator = ReferenceGenerator()
        learning = MemoryLearningSink()

        receipts = []
        previous = ZERO_HASH
        for stage in curriculum.stages:
            train_results = [
                gym.run(
                    by_id[task_id],
                    policy,
                    generator=generator,
                    learning_sink=learning,
                )
                for task_id in stage.train_task_ids
            ]
            validation_results = [
                gym.run(
                    by_id[task_id],
                    policy,
                    generator=generator,
                    learning_sink=learning,
                )
                for task_id in stage.validation_task_ids
            ]
            receipt = evaluate_stage(
                curriculum,
                stage,
                train_results,
                validation_results,
                previous_checkpoint_sha256=previous,
            )
            self.assertTrue(
                receipt.promoted,
                f"{stage.name}: {receipt.reasons}",
            )
            receipts.append(receipt)
            previous = receipt.checkpoint_sha256

        valid, reason = verify_promotion_chain(
            curriculum,
            receipts,
        )
        self.assertTrue(valid, reason)
        self.assertEqual(len(learning.updates), 9)
        self.assertEqual(
            set(learning.updates),
            {
                "tg_a7",
                "tg_b3",
                "tg_c9",
                "tg_d4",
                "tg_e6",
                "tg_f2",
                "tg_g5",
                "tg_h8",
                "tg_i1",
            },
        )


if __name__ == "__main__":
    unittest.main()
