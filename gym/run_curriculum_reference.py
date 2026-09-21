from __future__ import annotations

import json
from pathlib import Path

from .core import MemoryLearningSink, ReferenceGym
from .curriculum import (
    load_curriculum,
    evaluate_stage,
    validate_curriculum_tasks,
    verify_promotion_chain,
)
from .host import create_environment
from .io import load_tasks
from .ledger import ZERO_HASH
from .reference_policy import ReferenceGenerator, ReferencePolicy


def main() -> int:
    root = Path(__file__).parent
    tasks = load_tasks(root / "tasks")
    by_id = {task.task_id: task for task in tasks}
    curriculum = load_curriculum(
        root / "curriculum" / "g3_reference.json"
    )
    validate_curriculum_tasks(curriculum, tasks)

    gym = ReferenceGym(create_environment)
    policy = ReferencePolicy()
    generator = ReferenceGenerator()
    learning = MemoryLearningSink()

    receipts = []
    stages = []
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
        receipts.append(receipt)
        stages.append(
            {
                **receipt.to_dict(),
                "train_results": [
                    {
                        "task_id": result.task_id,
                        "accepted": result.accepted,
                        "learning_updated": result.learning_updated,
                        "terminal_reason": result.terminal_reason,
                        "budget_exhausted": result.budget_exhausted,
                        "ledger_root_sha256": result.ledger_root_sha256,
                    }
                    for result in train_results
                ],
                "validation_results": [
                    {
                        "task_id": result.task_id,
                        "accepted": result.accepted,
                        "learning_updated": result.learning_updated,
                        "terminal_reason": result.terminal_reason,
                        "budget_exhausted": result.budget_exhausted,
                        "ledger_root_sha256": result.ledger_root_sha256,
                    }
                    for result in validation_results
                ],
            }
        )
        previous = receipt.checkpoint_sha256
        if not receipt.promoted:
            break

    chain_valid, chain_reason = verify_promotion_chain(
        curriculum,
        receipts,
    )
    summary = {
        "format": "proof-gym-curriculum-reference-run-v1",
        "evaluation_scope": "reference_curriculum_conformance_only",
        "native_competence_claim": False,
        "curriculum_id": curriculum.curriculum_id,
        "stage_count": len(curriculum.stages),
        "completed_stage_count": len(receipts),
        "promotion_chain_valid": chain_valid,
        "promotion_chain_reason": chain_reason,
        "final_checkpoint_sha256": previous,
        "learning_updates": list(learning.updates),
        "stages": stages,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    passed = (
        len(receipts) == len(curriculum.stages)
        and chain_valid
        and all(receipt.promoted for receipt in receipts)
        and all(
            not row["learning_updated"]
            for stage in stages
            for row in stage["validation_results"]
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
