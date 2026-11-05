from __future__ import annotations

import unittest
from types import SimpleNamespace

from gym.external_training import build_external_training_receipt

_SHA = "a" * 64


def _receipt(accepted, semantic, provenance, trajectory, integrity, budget, terminal):
    """Gate-logic unit test with a minimal fabricated episode.

    build_external_training_receipt is duck-typed: it reads attributes
    only, so a SimpleNamespace episode is enough to exercise the gate.
    Cases A and E pair accepted=True with a failed flag; that receipt is
    inconsistent with the pipeline invariant (admitted is the AND of the
    four flags), but the gate must still return "none" — it checks every
    flag, not just `accepted`.
    """
    task = SimpleNamespace(split="train", task_id="t1")
    admission = SimpleNamespace(
        semantic_valid=semantic,
        provenance_valid=provenance,
        trajectory_valid=trajectory,
        integrity_valid=integrity,
    )
    episode = SimpleNamespace(
        task_id="t1",
        split="train",
        domain="d",
        accepted=accepted,
        budget_exhausted=budget,
        terminal_reason=terminal,
        ledger_root_sha256=_SHA,
        reward_channels={},
        trajectory=(),
        admission=admission,
    )
    entries = ({"kind": "AGENT_START", "payload": {"agent_task_view_sha256": _SHA}},)
    isolated = SimpleNamespace(
        episode=episode,
        agent_transcript_valid=True,
        agent_transcript_root_sha256=_SHA,
        agent_transcript_entries=entries,
    )
    return build_external_training_receipt(task, isolated)


class GymG43GateTests(unittest.TestCase):
    def test_semantic_valid_but_provenance_invalid_is_not_positive(self):
        receipt = _receipt(True, True, False, True, True, False, "ENVIRONMENT_DONE")
        self.assertEqual(receipt.experience_kind, "none")
        self.assertFalse(receipt.learning_admissible)

    def test_budget_exhausted_is_not_typed_negative(self):
        receipt = _receipt(False, False, True, True, True, True, "ENVIRONMENT_DONE")
        self.assertEqual(receipt.experience_kind, "none")
        self.assertFalse(receipt.learning_admissible)

    def test_control_typed_negative(self):
        receipt = _receipt(False, False, True, True, True, False, "ENVIRONMENT_DONE")
        self.assertEqual(receipt.experience_kind, "typed_negative")
        self.assertTrue(receipt.learning_admissible)

    def test_control_positive(self):
        receipt = _receipt(True, True, True, True, True, False, "ENVIRONMENT_DONE")
        self.assertEqual(receipt.experience_kind, "positive")
        self.assertTrue(receipt.learning_admissible)

    def test_trajectory_invalid_is_not_positive(self):
        receipt = _receipt(True, True, True, False, True, False, "ENVIRONMENT_DONE")
        self.assertEqual(receipt.experience_kind, "none")
        self.assertFalse(receipt.learning_admissible)


if __name__ == "__main__":
    unittest.main()
