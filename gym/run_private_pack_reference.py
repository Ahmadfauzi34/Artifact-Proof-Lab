from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .core import MemoryLearningSink, ReferenceGym
from .private_pack import PrivateHoldoutPack
from .reference_policy import ReferenceGenerator, ReferencePolicy
from .subprocess_host import SubprocessHostGateway


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run G2.2 sealed private holdout reference pack through subprocess host"
    )
    parser.add_argument(
        "--pack",
        type=Path,
        default=Path(__file__).parent / "private_holdout_reference",
    )
    parser.add_argument("--request-timeout", type=float, default=10.0)
    args = parser.parse_args(argv)

    pack_root = args.pack.resolve()
    pack = PrivateHoldoutPack.load(
        pack_root,
        expected_runtime_id="reference-v1",
    )

    repo_root = Path(__file__).resolve().parents[1]
    pythonpath = os.pathsep.join(
        [str(repo_root), str(repo_root / "src"), os.environ.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    command = [
        sys.executable,
        "-B",
        "-m",
        "gym.private_host_server",
        "--pack",
        str(pack_root),
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
        for task_id in pack.task_ids:
            task = pack.task(task_id)
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
                    "sealed_task_descriptor_sha256": pack.task_commitment(task_id),
                }
            )
        transport_alive_after_episodes = host.ping()

    summary = {
        "format": "proof-gym-private-pack-reference-run-v1",
        "evaluation_scope": "reference_private_pack_conformance_only",
        "native_competence_claim": False,
        "pack_id": pack.pack_id,
        "pack_manifest_sha256": pack.manifest_sha256,
        "pack_commitment_sha256": pack.commitment_sha256,
        "runtime_id": pack.runtime_id,
        "runtime_descriptor_sha256": pack.runtime_descriptor_sha256,
        "transport": "proof-gym-host-jsonl-v1",
        "transport_alive_after_episodes": transport_alive_after_episodes,
        "tasks": rows,
        "learning_updates": list(learning.updates),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    passed = (
        bool(rows)
        and all(row["accepted"] for row in rows)
        and all(not row["learning_updated"] for row in rows)
        and not learning.updates
        and pack.native_competence_claim is False
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
