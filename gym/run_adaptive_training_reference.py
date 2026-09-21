from __future__ import annotations

import json
from pathlib import Path

from .adaptive_policy import ProofGatedAdaptivePolicy
from .core import ReferenceGym
from .host import create_environment
from .io import load_tasks
from .reference_policy import ReferenceGenerator, ReferencePolicy


def _rate(results) -> float:
    items = tuple(results)
    if not items:
        return 0.0
    return sum(1 for result in items if result.accepted) / len(items)


def _result_row(result):
    return {
        "task_id": result.task_id,
        "split": result.split,
        "domain": result.domain,
        "accepted": result.accepted,
        "learning_updated": result.learning_updated,
        "terminal_reason": result.terminal_reason,
        "budget_exhausted": result.budget_exhausted,
        "trajectory": list(result.trajectory),
        "ledger_root_sha256": result.ledger_root_sha256,
    }


def main() -> int:
    root = Path(__file__).parent
    tasks = load_tasks(root / "tasks")
    train_tasks = tuple(
        task for task in tasks if task.split == "train"
    )
    validation_tasks = tuple(
        task for task in tasks if task.split == "validation"
    )
    holdout_tasks = tuple(
        task for task in tasks if task.split == "holdout"
    )

    gym = ReferenceGym(create_environment)
    learner = ProofGatedAdaptivePolicy()
    teacher = ReferencePolicy()
    generator = ReferenceGenerator()

    initial_policy_sha256 = learner.policy_sha256

    before_validation = [
        gym.run(
            task,
            learner,
            generator=generator,
        )
        for task in validation_tasks
    ]
    before_holdout = [
        gym.run(
            task,
            learner,
            generator=generator,
        )
        for task in holdout_tasks
    ]
    pre_eval_policy_sha256 = learner.policy_sha256

    train_results = [
        gym.run(
            task,
            teacher,
            generator=generator,
            learning_sink=learner,
        )
        for task in train_tasks
    ]
    trained_policy_sha256 = learner.policy_sha256

    after_validation = [
        gym.run(
            task,
            learner,
            generator=generator,
        )
        for task in validation_tasks
    ]
    after_holdout = [
        gym.run(
            task,
            learner,
            generator=generator,
        )
        for task in holdout_tasks
    ]
    final_policy_sha256 = learner.policy_sha256

    before_validation_rate = _rate(before_validation)
    after_validation_rate = _rate(after_validation)
    before_holdout_rate = _rate(before_holdout)
    after_holdout_rate = _rate(after_holdout)

    summary = {
        "format": "proof-gym-g4-adaptive-training-reference-v1",
        "evaluation_scope": "reference_adaptive_training_conformance_only",
        "native_competence_claim": False,
        "training_mode": "proof_gated_teacher_demonstration_q",
        "on_policy_exploration_claim": False,
        "negative_experience_learning_claim": False,
        "initial_policy_sha256": initial_policy_sha256,
        "pre_eval_policy_sha256": pre_eval_policy_sha256,
        "trained_policy_sha256": trained_policy_sha256,
        "final_policy_sha256": final_policy_sha256,
        "learned_pair_count": learner.learned_pair_count,
        "learning_receipt_count": len(learner.learning_receipts),
        "before_validation_admission_rate": before_validation_rate,
        "after_validation_admission_rate": after_validation_rate,
        "before_holdout_admission_rate": before_holdout_rate,
        "after_holdout_admission_rate": after_holdout_rate,
        "train": [_result_row(result) for result in train_results],
        "before_validation": [
            _result_row(result)
            for result in before_validation
        ],
        "after_validation": [
            _result_row(result)
            for result in after_validation
        ],
        "before_holdout": [
            _result_row(result)
            for result in before_holdout
        ],
        "after_holdout": [
            _result_row(result)
            for result in after_holdout
        ],
        "learning_receipts": [
            {
                "task_id": receipt.task_id,
                "ledger_root_sha256": receipt.ledger_root_sha256,
                "before_policy_sha256": receipt.before_policy_sha256,
                "after_policy_sha256": receipt.after_policy_sha256,
                "updated_pairs": receipt.updated_pairs,
            }
            for receipt in learner.learning_receipts
        ],
        "policy": learner.policy_document(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    passed = (
        initial_policy_sha256 == pre_eval_policy_sha256
        and trained_policy_sha256 != initial_policy_sha256
        and final_policy_sha256 == trained_policy_sha256
        and len(learner.learning_receipts) == len(train_tasks)
        and all(
            result.accepted and result.learning_updated
            for result in train_results
        )
        and all(
            result.accepted and not result.learning_updated
            for result in after_validation
        )
        and all(
            result.accepted and not result.learning_updated
            for result in after_holdout
        )
        and after_validation_rate > before_validation_rate
        and after_holdout_rate > before_holdout_rate
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
