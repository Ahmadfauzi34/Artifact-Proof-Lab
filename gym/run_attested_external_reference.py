from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import sys

from .attested_external import run_attested_external_episode
from .authority_attestation import (
    AuthorityClaims,
    REFERENCE_ASSURANCE,
    ReferenceHMACAuthorityAttestor,
    ReferenceHMACAuthorityVerifier,
)
from .core import MemoryLearningSink, ReferenceGym
from .external_agent import ExternalAgentEndpoint
from .private_pack import PrivateHoldoutPack
from .subprocess_host import SubprocessHostGateway


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run G2.5 reference authority-attestation mechanics against a "
            "pre-existing external AgentEndpoint"
        )
    )
    parser.add_argument("--agent-host", required=True)
    parser.add_argument("--agent-port", required=True, type=int)
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

    # Reference-only shared key created for this process. This deliberately
    # exercises receipt authenticity/binding/replay mechanics without claiming
    # independent hardware/cloud/VM authority.
    key = secrets.token_bytes(32)
    claims = AuthorityClaims(
        filesystem_isolated=True,
        kernel_isolated=True,
        credential_isolated=True,
        network_isolated=True,
        machine_identity_verified=False,
        model_identity_verified=False,
    )
    attestor = ReferenceHMACAuthorityAttestor(
        key,
        claims=claims,
    )
    verifier = ReferenceHMACAuthorityVerifier(key)

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
            request_timeout=args.request_timeout,
        ) as agent,
    ):
        gym = ReferenceGym(host_gateway=host)
        for task_id in pack.task_ids:
            task = pack.task(task_id)
            result = run_attested_external_episode(
                gym,
                task,
                agent,
                evaluation_context_sha256=pack.commitment_sha256,
                attestor=attestor,
                verifier=verifier,
                learning_sink=learning,
            )
            rows.append(
                {
                    "task_id": task_id,
                    "accepted": result.accepted,
                    "authority_valid": result.authority.valid,
                    "authority_assurance": result.authority.assurance,
                    "authority_receipt_sha256": result.authority_receipt_sha256,
                    "authority_request_sha256": result.authority_request_sha256,
                    "agent_transcript_root_sha256": (
                        result.isolated.agent_transcript_root_sha256
                    ),
                    "ledger_root_sha256": result.episode.ledger_root_sha256,
                    "learning_updated": result.episode.learning_updated,
                }
            )

    summary = {
        "format": "proof-gym-attested-external-reference-run-v1",
        "evaluation_scope": "reference_authority_attestation_conformance_only",
        "native_competence_claim": False,
        "authority_assurance": REFERENCE_ASSURANCE,
        "authority_independence_verified": False,
        "authority_cryptography": "shared_hmac_reference_only",
        "controller_launched_agent": False,
        "pack_id": pack.pack_id,
        "pack_commitment_sha256": pack.commitment_sha256,
        "tasks": rows,
        "learning_updates": list(learning.updates),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    passed = (
        bool(rows)
        and all(row["accepted"] for row in rows)
        and all(row["authority_valid"] for row in rows)
        and all(
            row["authority_assurance"] == REFERENCE_ASSURANCE
            for row in rows
        )
        and all(not row["learning_updated"] for row in rows)
        and not learning.updates
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
