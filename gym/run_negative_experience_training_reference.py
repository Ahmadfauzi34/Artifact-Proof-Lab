from __future__ import annotations

import json
from pathlib import Path

from .adaptive_policy import ProofGatedAdaptivePolicy
from .core import ReferenceGym
from .host import create_environment
from .io import load_tasks
from .negative_experience import (
    NegativeAwareExplorationPolicy,
    NegativeExperienceTrainer,
)
from .reference_policy import ReferenceGenerator


def _rate(results) -> float:
    items = tuple(results)
    return 0.0 if not items else sum(r.accepted for r in items) / len(items)


def main() -> int:
    root = Path(__file__).parent
    tasks = load_tasks(root / "tasks")
    train = tuple(t for t in tasks if t.split == "train")
    validation = tuple(t for t in tasks if t.split == "validation")
    holdout = tuple(t for t in tasks if t.split == "holdout")

    gym = ReferenceGym(create_environment)
    learner = ProofGatedAdaptivePolicy()
    policy = NegativeAwareExplorationPolicy(learner)
    generator = ReferenceGenerator()

    initial_q = learner.policy_sha256
    initial_negative = policy.negative_sha256
    before_validation = [
        gym.run(t, learner, generator=generator)
        for t in validation
    ]
    before_holdout = [
        gym.run(t, learner, generator=generator)
        for t in holdout
    ]

    trainer = NegativeExperienceTrainer(
        gym,
        policy,
        generator=generator,
        max_attempts_per_task=6,
    )
    training = trainer.train(train)
    trained_q = learner.policy_sha256
    trained_negative = policy.negative_sha256

    after_validation = [
        gym.run(t, learner, generator=generator)
        for t in validation
    ]
    after_holdout = [
        gym.run(t, learner, generator=generator)
        for t in holdout
    ]
    final_q = learner.policy_sha256
    final_negative = policy.negative_sha256

    doc = {
        "format": "proof-gym-g42-typed-negative-experience-v1",
        "evaluation_scope":
            "reference_typed_negative_experience_conformance_only",
        "native_competence_claim": False,
        "reference_policy_used_for_training": False,
        "positive_q_credit": "fully_admitted_train_only",
        "negative_experience_credit":
            "semantic_rejection_with_provenance_trajectory_integrity_only",
        "negative_experience_updates_q": False,
        "negative_experience_is_truth": False,
        "negative_experience_is_proof": False,
        "initial_q_sha256": initial_q,
        "trained_q_sha256": trained_q,
        "final_q_sha256": final_q,
        "initial_negative_sha256": initial_negative,
        "trained_negative_sha256": trained_negative,
        "final_negative_sha256": final_negative,
        "learning_receipt_count": len(learner.learning_receipts),
        "negative_receipt_count": len(policy.negative.receipts),
        "learned_pair_count": learner.learned_pair_count,
        "total_training_attempts": sum(
            item.attempts for item in training
        ),
        "before_validation_admission_rate": _rate(before_validation),
        "after_validation_admission_rate": _rate(after_validation),
        "before_holdout_admission_rate": _rate(before_holdout),
        "after_holdout_admission_rate": _rate(after_holdout),
        "training": [
            {
                "task_id": item.task_id,
                "attempts": item.attempts,
                "accepted": item.accepted,
                "typed_negative_attempts":
                    item.typed_negative_attempts,
                "final_trajectory":
                    list(item.final_result.trajectory),
            }
            for item in training
        ],
        "negative_receipts": [
            {
                "task_id": receipt.task_id,
                "state": receipt.state,
                "intent": receipt.intent,
                "previous_rejections":
                    receipt.previous_rejections,
                "next_rejections": receipt.next_rejections,
                "ledger_root_sha256":
                    receipt.ledger_root_sha256,
            }
            for receipt in policy.negative.receipts
        ],
        "negative_state": policy.negative.document(),
        "policy": learner.policy_document(),
    }
    print(json.dumps(doc, indent=2, sort_keys=True))

    passed = (
        trained_q != initial_q
        and trained_negative != initial_negative
        and final_q == trained_q
        and final_negative == trained_negative
        and len(learner.learning_receipts) == len(train)
        and len(policy.negative.receipts) == 7
        and all(item.accepted for item in training)
        and all(
            not result.learning_updated
            for result in after_validation
        )
        and all(
            not result.learning_updated
            for result in after_holdout
        )
        and _rate(after_validation) > _rate(before_validation)
        and _rate(after_holdout) > _rate(before_holdout)
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
