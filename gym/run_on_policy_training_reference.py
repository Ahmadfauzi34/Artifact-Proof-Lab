from __future__ import annotations

import json
from pathlib import Path

from .adaptive_policy import ProofGatedAdaptivePolicy
from .core import ReferenceGym
from .host import create_environment
from .io import load_tasks
from .on_policy import BoundedExplorationPolicy, OnPolicyTrainer
from .reference_policy import ReferenceGenerator


def _rate(results) -> float:
    items = tuple(results)
    return 0.0 if not items else sum(r.accepted for r in items) / len(items)


def _row(result):
    return {
        "task_id": result.task_id,
        "accepted": result.accepted,
        "trajectory": list(result.trajectory),
        "learning_updated": result.learning_updated,
        "ledger_root_sha256": result.ledger_root_sha256,
    }


def main() -> int:
    root = Path(__file__).parent
    tasks = load_tasks(root / "tasks")
    train = tuple(t for t in tasks if t.split == "train")
    validation = tuple(t for t in tasks if t.split == "validation")
    holdout = tuple(t for t in tasks if t.split == "holdout")

    gym = ReferenceGym(create_environment)
    learner = ProofGatedAdaptivePolicy()
    explorer = BoundedExplorationPolicy(learner)
    generator = ReferenceGenerator()

    initial_policy = learner.policy_sha256
    before_validation = [gym.run(t, learner, generator=generator) for t in validation]
    before_holdout = [gym.run(t, learner, generator=generator) for t in holdout]
    pretrain_policy = learner.policy_sha256

    trainer = OnPolicyTrainer(gym, explorer, generator=generator, max_attempts_per_task=6)
    training = trainer.train(train)
    trained_policy = learner.policy_sha256
    trained_search = explorer.search_sha256

    after_validation = [gym.run(t, learner, generator=generator) for t in validation]
    after_holdout = [gym.run(t, learner, generator=generator) for t in holdout]
    final_policy = learner.policy_sha256
    final_search = explorer.search_sha256

    doc = {
        "format": "proof-gym-g41-on-policy-search-v1",
        "evaluation_scope": "reference_on_policy_search_conformance_only",
        "native_competence_claim": False,
        "reference_policy_used_for_training": False,
        "training_actor": "BoundedExplorationPolicy",
        "q_learning_credit": "fully_admitted_train_only",
        "failed_episode_effect": "validated_search_cursor_only",
        "negative_q_learning_claim": False,
        "initial_policy_sha256": initial_policy,
        "pretrain_policy_sha256": pretrain_policy,
        "trained_policy_sha256": trained_policy,
        "final_policy_sha256": final_policy,
        "trained_search_sha256": trained_search,
        "final_search_sha256": final_search,
        "learning_receipt_count": len(learner.learning_receipts),
        "search_receipt_count": len(explorer.search_receipts),
        "learned_pair_count": learner.learned_pair_count,
        "total_training_attempts": sum(item.attempts for item in training),
        "before_validation_admission_rate": _rate(before_validation),
        "after_validation_admission_rate": _rate(after_validation),
        "before_holdout_admission_rate": _rate(before_holdout),
        "after_holdout_admission_rate": _rate(after_holdout),
        "training": [
            {
                "task_id": item.task_id,
                "attempts": item.attempts,
                "accepted": item.accepted,
                "validated_failed_attempts": item.validated_failed_attempts,
                "final_trajectory": list(item.final_result.trajectory),
            }
            for item in training
        ],
        "search_receipts": [
            {
                "task_id": r.task_id,
                "state": r.state,
                "intent": r.intent,
                "previous_cursor": r.previous_cursor,
                "next_cursor": r.next_cursor,
                "ledger_root_sha256": r.ledger_root_sha256,
            }
            for r in explorer.search_receipts
        ],
        "before_validation": [_row(r) for r in before_validation],
        "after_validation": [_row(r) for r in after_validation],
        "before_holdout": [_row(r) for r in before_holdout],
        "after_holdout": [_row(r) for r in after_holdout],
        "policy": learner.policy_document(),
    }
    print(json.dumps(doc, indent=2, sort_keys=True))

    passed = (
        initial_policy == pretrain_policy
        and trained_policy != initial_policy
        and final_policy == trained_policy
        and final_search == trained_search
        and len(learner.learning_receipts) == len(train)
        and all(item.accepted for item in training)
        and all(not r.learning_updated for r in after_validation)
        and all(not r.learning_updated for r in after_holdout)
        and _rate(after_validation) > _rate(before_validation)
        and _rate(after_holdout) > _rate(before_holdout)
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
