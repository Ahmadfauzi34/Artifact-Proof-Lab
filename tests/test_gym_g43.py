from __future__ import annotations

from dataclasses import asdict
import unittest

from gym.core import (
    AdmissionReceipt,
    EpisodeResult,
    PublicTask,
    TerminalReason,
)
from gym.external_training import (
    NO_LEARNING_EXPERIENCE,
    POSITIVE_EXPERIENCE,
    TYPED_NEGATIVE_EXPERIENCE,
    build_external_training_receipt,
    verify_external_training_receipt,
)
from gym.isolated_agent import IsolatedEpisodeResult
from gym.ledger import sha256_json


class ExternalTrainingReceiptTests(unittest.TestCase):
    def _task(self) -> PublicTask:
        return PublicTask(
            task_id="train-x",
            domain="repository_coding",
            skill_targets=("repair",),
            description="repair task",
            environment={},
            allowed_actions=("RUN_TARGETED_TEST",),
            budget={"max_steps": 3, "max_generative_calls": 0},
            split="train",
        )

    def _isolated(
        self,
        *,
        semantic: bool,
        provenance: bool = True,
        trajectory: bool = True,
        integrity: bool = True,
        budget_exhausted: bool = False,
        terminal_reason: str = TerminalReason.ENVIRONMENT_DONE.value,
    ) -> IsolatedEpisodeResult:
        task = self._task()
        task_sha = sha256_json(asdict(task.agent_view()))
        transcript_entries = (
            {
                "index": 0,
                "kind": "AGENT_START",
                "previous_hash": "0" * 64,
                "payload_sha256": "1" * 64,
                "payload": {
                    "session_id": "s",
                    "agent_task_view_sha256": task_sha,
                },
                "entry_hash": "2" * 64,
            },
            {
                "index": 1,
                "kind": "AGENT_CLOSE",
                "previous_hash": "2" * 64,
                "payload_sha256": "3" * 64,
                "payload": {"session_id": "s", "closed": True},
                "entry_hash": "4" * 64,
            },
        )
        admission = AdmissionReceipt(
            semantic_valid=semantic,
            provenance_valid=provenance,
            trajectory_valid=trajectory,
            integrity_valid=integrity,
            admitted=bool(semantic and provenance and trajectory and integrity),
            reason="test",
        )
        episode = EpisodeResult(
            task_id=task.task_id,
            split="train",
            domain=task.domain,
            trajectory=("RUN_TARGETED_TEST",),
            observations=(),
            admission=admission,
            reward_channels={"task_acceptance": 1.0 if semantic else 0.0},
            learning_updated=False,
            generative_calls=0,
            terminal_reason=terminal_reason,
            budget_exhausted=budget_exhausted,
            ledger_root_sha256="a" * 64,
            ledger_entry_count=2,
            ledger_entries=(),
        )
        return IsolatedEpisodeResult(
            episode=episode,
            agent_transcript_valid=True,
            agent_transcript_reason="bound",
            agent_transcript_root_sha256="b" * 64,
            agent_transcript_entries=transcript_entries,
        )

    def test_positive_receipt(self):
        doc = build_external_training_receipt(
            self._task(),
            self._isolated(semantic=True),
        ).document()
        self.assertEqual(doc["experience_kind"], POSITIVE_EXPERIENCE)
        self.assertTrue(doc["learning_admissible"])
        self.assertFalse(doc["gym_applied_learning"])
        self.assertEqual(verify_external_training_receipt(doc), (
            True,
            "external training receipt valid",
        ))

    def test_typed_negative_receipt(self):
        doc = build_external_training_receipt(
            self._task(),
            self._isolated(semantic=False),
        ).document()
        self.assertEqual(doc["experience_kind"], TYPED_NEGATIVE_EXPERIENCE)
        self.assertTrue(doc["learning_admissible"])
        self.assertEqual(verify_external_training_receipt(doc)[0], True)

    def test_budget_failure_is_not_learning(self):
        doc = build_external_training_receipt(
            self._task(),
            self._isolated(
                semantic=False,
                budget_exhausted=True,
                terminal_reason=TerminalReason.STEP_BUDGET_EXHAUSTED.value,
            ),
        ).document()
        self.assertEqual(doc["experience_kind"], NO_LEARNING_EXPERIENCE)
        self.assertFalse(doc["learning_admissible"])
        self.assertEqual(verify_external_training_receipt(doc)[0], True)

    def test_tampered_receipt_is_rejected(self):
        doc = build_external_training_receipt(
            self._task(),
            self._isolated(semantic=True),
        ).document()
        doc["accepted"] = False
        self.assertEqual(
            verify_external_training_receipt(doc),
            (False, "external training receipt digest mismatch"),
        )


if __name__ == "__main__":
    unittest.main()
