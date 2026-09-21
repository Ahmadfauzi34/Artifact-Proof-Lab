from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .core import MemoryLearningSink, ReferenceGym
from .external_agent import ExternalAgentEndpoint
from .isolated_agent import run_isolated_episode
from .private_pack import PrivateHoldoutPack
from .subprocess_host import SubprocessHostGateway


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the G2.4 trusted controller against a pre-existing external "
            "AgentEndpoint service"
        )
    )
    parser.add_argument("--agent-host", required=True)
    parser.add_argument("--agent-port", required=True, type=int)
    parser.add_argument(
        "--pack",
        type=Path,
        default=Path(__file__).parent / "private_holdout_reference",
    )
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--request-timeout", type=float, default=10.0)
    args = parser.parse_args(argv)

    pack_root = args.pack.resolve()
    pack = PrivateHoldoutPack.load(
        pack_root,
        expected_runtime_id="reference-v1",
    )

    repo_root = Path(__file__).resolve().parents[1]
    pythonpath = os.pathsep.join(
        [
            str(repo_root),
            str(repo_root / "src"),
            os.environ.get("PYTHONPATH", ""),
        ]
    ).rstrip(os.pathsep)

    host_command = [
        sys.executable,
        "-B",
        "-m",
        "gym.private_host_server",
        "--pack",
        str(pack_root),
    ]

    learning = MemoryLearningSink()
    rows = []

    with (
        SubprocessHostGateway(
            host_command,
            cwd=repo_root,
            env={"PYTHONPATH": pythonpath},
            request_timeout=args.request_timeout,
        ) as host,
        ExternalAgentEndpoint(
            args.agent_host,
            args.agent_port,
            connect_timeout=args.connect_timeout,
            request_timeout=args.request_timeout,
        ) as agent,
    ):
        gym = ReferenceGym(host_gateway=host)
        for task_id in pack.task_ids:
            task = pack.task(task_id)
            isolated = run_isolated_episode(
                gym,
                task,
                agent,
                learning_sink=learning,
            )
            episode = isolated.episode
            rows.append(
                {
                    "task_id": episode.task_id,
                    "split": episode.split,
                    "domain": episode.domain,
                    "accepted": isolated.accepted,
                    "terminal_reason": episode.terminal_reason,
                    "ledger_root_sha256": episode.ledger_root_sha256,
                    "agent_transcript_valid": isolated.agent_transcript_valid,
                    "agent_transcript_root_sha256": (
                        isolated.agent_transcript_root_sha256
                    ),
                    "learning_updated": episode.learning_updated,
                    "sealed_task_descriptor_sha256": (
                        pack.task_commitment(task_id)
                    ),
                }
            )
        host_alive_after_episodes = host.ping()
        agent_alive_after_episodes = agent.ping()

    summary = {
        "format": "proof-gym-external-agent-reference-run-v1",
        "evaluation_scope": "external_agent_launch_decoupling_conformance_only",
        "native_competence_claim": False,
        "controller_launched_agent": False,
        "agent_launch_mode": "external_preexisting_endpoint",
        "authority_attestation": "not_present",
        "filesystem_authority_isolated": "unverified",
        "kernel_authority_isolated": "unverified",
        "credential_authority_isolated": "unverified",
        "transport_confidentiality": "not_provided_by_reference_tcp",
        "pack_id": pack.pack_id,
        "pack_manifest_sha256": pack.manifest_sha256,
        "pack_commitment_sha256": pack.commitment_sha256,
        "host_transport": "proof-gym-host-jsonl-v1",
        "agent_transport": "proof-gym-agent-jsonl-v1-over-tcp",
        "host_alive_after_episodes": host_alive_after_episodes,
        "agent_alive_after_episodes": agent_alive_after_episodes,
        "tasks": rows,
        "learning_updates": list(learning.updates),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    passed = (
        bool(rows)
        and all(row["accepted"] for row in rows)
        and all(row["agent_transcript_valid"] for row in rows)
        and all(not row["learning_updated"] for row in rows)
        and not learning.updates
        and pack.native_competence_claim is False
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
