from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import ReferenceGym
from .external_agent import ExternalAgentEndpoint
from .external_training import (
    build_external_training_receipt,
    verify_external_training_receipt,
)
from .host import create_environment
from .io import load_tasks
from .isolated_agent import run_isolated_episode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run proof-gated train episodes against a pre-existing external "
            "AgentEndpoint and emit portable training receipts"
        )
    )
    parser.add_argument("--agent-host", required=True)
    parser.add_argument("--agent-port", required=True, type=int)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path(__file__).parent / "tasks",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="limit execution to one or more train task ids",
    )
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--request-timeout", type=float, default=10.0)
    args = parser.parse_args(argv)

    tasks = tuple(load_tasks(args.tasks.resolve()))
    by_id = {task.task_id: task for task in tasks}
    if args.task_id:
        missing = [task_id for task_id in args.task_id if task_id not in by_id]
        if missing:
            parser.error(f"unknown task ids: {', '.join(missing)}")
        selected = tuple(by_id[task_id] for task_id in args.task_id)
    else:
        selected = tuple(task for task in tasks if task.split == "train")

    if not selected:
        parser.error("no train tasks selected")
    non_train = [task.task_id for task in selected if task.split != "train"]
    if non_train:
        parser.error(
            "external-agent-train accepts train split only: "
            + ", ".join(non_train)
        )

    gym = ReferenceGym(create_environment)
    receipts = []

    with ExternalAgentEndpoint(
        args.agent_host,
        args.agent_port,
        connect_timeout=args.connect_timeout,
        request_timeout=args.request_timeout,
    ) as agent:
        for task in selected:
            isolated = run_isolated_episode(
                gym,
                task,
                agent,
                learning_sink=None,
            )
            receipt = build_external_training_receipt(task, isolated).document()
            valid, reason = verify_external_training_receipt(receipt)
            if not valid:
                raise RuntimeError(
                    f"external training receipt verification failed: {reason}"
                )
            receipts.append(receipt)
        agent_alive_after_training = agent.ping()

    summary = {
        "format": "proof-gym-external-agent-training-run-v1",
        "evaluation_scope": "external_agent_proof_gated_training_receipts",
        "native_competence_claim": False,
        "controller_launched_agent": False,
        "agent_launch_mode": "external_preexisting_endpoint",
        "gym_applied_learning": False,
        "consumer_must_apply_receipts": True,
        "agent_transport": "proof-gym-agent-jsonl-v1-over-tcp",
        "agent_alive_after_training": agent_alive_after_training,
        "task_count": len(receipts),
        "positive_receipt_count": sum(
            row["experience_kind"] == "positive" for row in receipts
        ),
        "typed_negative_receipt_count": sum(
            row["experience_kind"] == "typed_negative" for row in receipts
        ),
        "non_learning_receipt_count": sum(
            row["experience_kind"] == "none" for row in receipts
        ),
        "receipts": receipts,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
