from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from .authority_attestation import (
    AuthorityAttestationError,
    AuthorityAttestationRequest,
    AuthorityAttestor,
    AuthorityVerification,
    AuthorityVerifier,
    authority_receipt_sha256,
)
from .core import EpisodeResult, LearningSink, PublicTask, ReferenceGym
from .external_agent import ExternalAgentEndpoint
from .isolated_agent import IsolatedEpisodeResult, run_isolated_episode
from .ledger import sha256_json


@dataclass(frozen=True)
class AttestedExternalEpisodeResult:
    isolated: IsolatedEpisodeResult
    authority: AuthorityVerification
    authority_receipt_sha256: str
    authority_request_sha256: str

    @property
    def episode(self) -> EpisodeResult:
        return self.isolated.episode

    @property
    def accepted(self) -> bool:
        return self.isolated.accepted and self.authority.valid


def run_attested_external_episode(
    gym: ReferenceGym,
    task: PublicTask,
    endpoint: ExternalAgentEndpoint,
    *,
    evaluation_context_sha256: str,
    attestor: AuthorityAttestor,
    verifier: AuthorityVerifier,
    learning_sink: LearningSink | None = None,
) -> AttestedExternalEpisodeResult:
    """Run an external-agent episode gated by independently supplied evidence.

    The reference HMAC attestor/verifier only proves contract mechanics. Real
    authority assurance depends on replacing them with a trusted independent
    issuer/verifier whose evidence semantics are appropriate to the deployment.
    """

    peer_host, peer_port = endpoint.connected_peer
    request = AuthorityAttestationRequest.create(
        peer_host=peer_host,
        peer_port=peer_port,
        host_task_commitment=sha256_json(asdict(task)),
        agent_task_view_sha256=sha256_json(asdict(task.agent_view())),
        evaluation_context_sha256=evaluation_context_sha256,
    )

    try:
        receipt = attestor.attest(request)
        verification = verifier.verify(request, receipt)
    except Exception:
        endpoint.disconnect()
        raise

    if not verification.valid:
        endpoint.disconnect()
        raise AuthorityAttestationError(
            f"authority attestation rejected: {verification.reason}"
        )

    isolated = run_isolated_episode(
        gym,
        task,
        endpoint,
        learning_sink=None,
    )
    try:
        _verify_episode_binding(
            request,
            isolated.episode,
        )
    except Exception:
        endpoint.disconnect()
        raise

    learning_updated = False
    episode = isolated.episode
    if (
        isolated.accepted
        and task.split == "train"
        and learning_sink is not None
    ):
        learning_sink.update(episode)
        learning_updated = True

    if learning_updated:
        episode = replace(
            episode,
            learning_updated=True,
        )
        isolated = replace(
            isolated,
            episode=episode,
        )

    return AttestedExternalEpisodeResult(
        isolated=isolated,
        authority=verification,
        authority_receipt_sha256=authority_receipt_sha256(receipt),
        authority_request_sha256=sha256_json(request.to_dict()),
    )


def _verify_episode_binding(
    request: AuthorityAttestationRequest,
    episode: EpisodeResult,
) -> None:
    if not episode.ledger_entries:
        raise AuthorityAttestationError(
            "episode ledger missing for authority cross-binding"
        )
    start = episode.ledger_entries[0]
    if start.kind != "EPISODE_START":
        raise AuthorityAttestationError(
            "episode ledger does not start with EPISODE_START"
        )
    if (
        start.payload.get("host_task_commitment")
        != request.host_task_commitment
    ):
        raise AuthorityAttestationError(
            "authority host-task commitment mismatch with episode ledger"
        )
    if (
        start.payload.get("agent_task_view_sha256")
        != request.agent_task_view_sha256
    ):
        raise AuthorityAttestationError(
            "authority agent-task commitment mismatch with episode ledger"
        )
