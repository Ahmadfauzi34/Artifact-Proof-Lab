from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import sys
from typing import Mapping

from .agent_endpoint import AgentEndpoint
from .agent_transcript import (
    AgentBoundaryTranscript,
    AgentTranscriptError,
    verify_agent_transcript_against_episode,
)
from .core import (
    AgentTaskView,
    EpisodeResult,
    LearningSink,
    Observation,
    PolicyDecision,
    PublicTask,
    ReferenceGym,
)
from .ledger import sha256_json


@dataclass(frozen=True)
class IsolatedEpisodeResult:
    episode: EpisodeResult
    agent_transcript_valid: bool
    agent_transcript_reason: str
    agent_transcript_root_sha256: str
    agent_transcript_entries: tuple[Mapping[str, object], ...]

    @property
    def accepted(self) -> bool:
        return self.episode.accepted and self.agent_transcript_valid


class AgentEndpointPolicyAdapter:
    """Policy+Generator proxy that exposes only AgentTaskView and observations."""

    def __init__(
        self,
        endpoint: AgentEndpoint,
        task: AgentTaskView,
    ) -> None:
        self._endpoint = endpoint
        self._task = task
        start = endpoint.start(task)
        expected_sha = sha256_json(asdict(task))
        if start.agent_task_view_sha256 != expected_sha:
            try:
                endpoint.close(start.session_id)
            finally:
                raise AgentTranscriptError(
                    "agent endpoint start task commitment mismatch"
                )
        self._session_id = start.session_id
        self._transcript = AgentBoundaryTranscript(
            task,
            start.session_id,
        )
        self._closed = False

    @property
    def transcript(self) -> AgentBoundaryTranscript:
        return self._transcript

    def decide(
        self,
        task: AgentTaskView,
        observations: tuple[Observation, ...],
        trajectory: tuple[str, ...],
    ) -> PolicyDecision:
        self._require_task(task)
        receipt = self._endpoint.decide(
            self._session_id,
            observations,
            trajectory,
        )
        self._transcript.record_decision(
            turn=receipt.turn,
            observations=observations,
            trajectory=trajectory,
            decision=receipt.decision,
        )
        return receipt.decision

    def generate(
        self,
        task: AgentTaskView,
        observations: tuple[Observation, ...],
        attempt: int,
    ) -> str:
        self._require_task(task)
        receipt = self._endpoint.generate(
            self._session_id,
            observations,
            attempt,
        )
        self._transcript.record_generation(
            turn=receipt.turn,
            observations=observations,
            attempt=attempt,
            candidate=receipt.candidate,
        )
        return receipt.candidate

    def close(self) -> None:
        if self._closed:
            return
        self._endpoint.close(self._session_id)
        self._transcript.record_close(
            session_id=self._session_id,
        )
        self._closed = True

    def _require_task(self, task: AgentTaskView) -> None:
        if task != self._task:
            raise AgentTranscriptError(
                "ReferenceGym attempted to switch AgentTaskView mid-session"
            )


def run_isolated_episode(
    gym: ReferenceGym,
    task: PublicTask,
    endpoint: AgentEndpoint,
    *,
    learning_sink: LearningSink | None = None,
) -> IsolatedEpisodeResult:
    """Run one episode with an isolated agent boundary.

    Learning is intentionally withheld from ReferenceGym and applied only after
    the agent-boundary transcript is structurally valid and cross-bound to the
    episode trajectory ledger.
    """

    adapter = AgentEndpointPolicyAdapter(
        endpoint,
        task.agent_view(),
    )
    episode: EpisodeResult | None = None
    try:
        episode = gym.run(
            task,
            adapter,
            generator=adapter,
            learning_sink=None,
        )
    finally:
        primary_failure_active = sys.exc_info()[0] is not None
        try:
            adapter.close()
        except Exception:
            if not primary_failure_active:
                raise

    assert episode is not None
    transcript_document = adapter.transcript.to_document()
    transcript_valid, transcript_reason = (
        verify_agent_transcript_against_episode(
            transcript_document,
            task=task.agent_view(),
            episode=episode,
        )
    )
    if not transcript_valid:
        raise AgentTranscriptError(
            f"agent transcript rejected: {transcript_reason}"
        )

    learning_updated = False
    if (
        episode.accepted
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

    return IsolatedEpisodeResult(
        episode=episode,
        agent_transcript_valid=True,
        agent_transcript_reason=transcript_reason,
        agent_transcript_root_sha256=adapter.transcript.root_sha256,
        agent_transcript_entries=adapter.transcript.entries,
    )
