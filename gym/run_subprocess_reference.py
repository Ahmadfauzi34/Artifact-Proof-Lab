from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .core import MemoryLearningSink, ReferenceGym
from .io import load_tasks
from .reference_policy import ReferenceGenerator, ReferencePolicy
from .subprocess_host import SubprocessHostGateway


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run reference gym through the G2.1 subprocess private-host transport"
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path(__file__).parent / "tasks",
    )
    parser.add_argument("--request-timeout", type=float, default=10.0)
    args = parser.parse_args(argv)

    tasks_root = args.tasks.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    pythonpath = os.pathsep.join(
        [str(repo_root), str(repo_root / "src"), os.environ.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    command = [
        sys.executable,
        "-B",
        "-m",
        "gym.private_host_server",
        "--tasks",
        str(tasks_root),
    ]

    policy = ReferencePolicy()
    generator = ReferenceGenerator()
    learning = MemoryLearningSink()
    rows = []

    with SubprocessHostGateway(
        command,
        cwd=repo_root,
        env={"PYTHONPATH": pythonpath},
        request_timeout=args.request_timeout,
    ) as host:
        gym = ReferenceGym(host_gateway=host)
        for task in load_tasks(tasks_root):
            result = gym.run(
                task,
                policy,
                generator=generator,
                learning_sink=learning,
            )
            rows.append(
                {
                    "task_id": result.task_id,
                    "split": result.split,
                    "domain": result.domain,
                    "accepted": result.accepted,
                    "terminal_reason": result.terminal_reason,
                    "ledger_root_sha256": result.ledger_root_sha256,
                    "learning_updated": result.learning_updated,
                }
            )
        transport_alive_after_episodes = host.ping()

    summary = {
        "format": "proof-gym-subprocess-reference-run-v1",
        "evaluation_scope": "reference_contract_conformance_only",
        "native_competence_claim": False,
        "transport": "proof-gym-host-jsonl-v1",
        "transport_alive_after_episodes": transport_alive_after_episodes,
        "tasks": rows,
        "learning_updates": list(learning.updates),
        "split_metrics": {
            split: {
                "count": sum(1 for row in rows if row["split"] == split),
                "accepted": sum(
                    1
                    for row in rows
                    if row["split"] == split and row["accepted"]
                ),
            }
            for split in ("train", "validation", "holdout", "external_real")
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if rows and all(row["accepted"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
