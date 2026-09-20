from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import MemoryLearningSink, ReferenceGym
from .host import create_environment
from .io import load_tasks
from .reference_policy import ReferenceGenerator, ReferencePolicy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Proof-Gated Adaptive Agent Gym reference policy")
    parser.add_argument("--tasks", type=Path, default=Path(__file__).parent / "tasks")
    args = parser.parse_args(argv)

    gym = ReferenceGym(create_environment)
    policy = ReferencePolicy()
    generator = ReferenceGenerator()
    learning = MemoryLearningSink()
    rows = []
    for task in load_tasks(args.tasks):
        result = gym.run(task, policy, generator=generator, learning_sink=learning)
        rows.append(
            {
                "task_id": result.task_id,
                "split": result.split,
                "domain": result.domain,
                "accepted": result.accepted,
                "learning_updated": result.learning_updated,
                "generative_calls": result.generative_calls,
                "trajectory": list(result.trajectory),
            }
        )
    summary = {
        "format": "proof-gym-reference-run-v1",
        "tasks": rows,
        "learning_updates": list(learning.updates),
        "split_metrics": {
            split: {
                "count": sum(1 for row in rows if row["split"] == split),
                "accepted": sum(1 for row in rows if row["split"] == split and row["accepted"]),
            }
            for split in ("train", "validation", "holdout", "external_real")
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if all(row["accepted"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
