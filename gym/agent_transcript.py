from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
from typing import Any, Mapping

from .core import AgentTaskView, DecisionKind, EpisodeResult, PolicyDecision
from .ledger import ZERO_HASH, observation_frontier_sha256, sha256_json


AGENT_TRANSCRIPT_FORMAT = "proof-gym-agent-boundary-transcript-v1"


class AgentTranscriptError(RuntimeError):
    """Agent boundary transcript failed structural or episode cross-binding."""


class AgentBoundaryTranscript:
    """Trusted-controller hash chain of sanitized agent I/O.

    Private reasoning is intentionally absent. Generated candidate text is stored
    only by SHA-256.
    """

    def __init__(self, task: AgentTaskView, session_id: str) -> None:
        self._entries: list[dict[str, Any]] = []
        self._append(
            "AGENT_START",
            {
                "session_id": session_id,
                "agent_task_view_sha256": sha256_json(asdict(task)),
            },
        )

    @property
    def entries(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(deepcopy(entry) for entry in self._entries)

    @property
    def root_sha256(self) -> str:
        if not self._entries:
            return ZERO_HASH
        return str(self._entries[-1]["entry_hash"])

    def record_decision(
        self,
        *,
        turn: int,
        observations,
        trajectory: tuple[str, ...],
        decision: PolicyDecision,
    ) -> None:
        self._append(
            "AGENT_DECISION",
            {
                "turn": turn,
                "observation_frontier_sha256": observation_frontier_sha256(
                    [item.to_dict() for item in observations]
                ),
                "trajectory_sha256": sha256_json(list(trajectory)),
                "decision": {
                    "kind": decision.kind.value,
                    "action": decision.action,
                },
            },
        )

    def record_generation(
        self,
        *,
        turn: int,
        observations,
        attempt: int,
        candidate: str,
    ) -> None:
        self._append(
            "AGENT_GENERATION",
            {
                "turn": turn,
                "observation_frontier_sha256": observation_frontier_sha256(
                    [item.to_dict() for item in observations]
                ),
                "attempt": attempt,
                "candidate_sha256": hashlib.sha256(
                    candidate.encode("utf-8")
                ).hexdigest(),
            },
        )

    def record_close(self, *, session_id: str) -> None:
        self._append(
            "AGENT_CLOSE",
            {
                "session_id": session_id,
                "closed": True,
            },
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "format": AGENT_TRANSCRIPT_FORMAT,
            "root_sha256": self.root_sha256,
            "entry_count": len(self._entries),
            "entries": deepcopy(self._entries),
        }

    def verify(self) -> tuple[bool, str]:
        return verify_agent_transcript_document(self.to_document())

    def _append(self, kind: str, payload: Mapping[str, Any]) -> None:
        index = len(self._entries)
        previous_hash = (
            ZERO_HASH if not self._entries else str(self._entries[-1]["entry_hash"])
        )
        payload_copy = deepcopy(dict(payload))
        payload_sha256 = sha256_json(payload_copy)
        entry_hash = sha256_json(
            {
                "index": index,
                "kind": kind,
                "previous_hash": previous_hash,
                "payload_sha256": payload_sha256,
            }
        )
        self._entries.append(
            {
                "index": index,
                "kind": kind,
                "previous_hash": previous_hash,
                "payload_sha256": payload_sha256,
                "payload": payload_copy,
                "entry_hash": entry_hash,
            }
        )


def verify_agent_transcript_document(
    document: Mapping[str, Any],
) -> tuple[bool, str]:
    if set(document) != {
        "format",
        "root_sha256",
        "entry_count",
        "entries",
    }:
        return False, "agent transcript document fields are invalid"
    if document.get("format") != AGENT_TRANSCRIPT_FORMAT:
        return False, "unsupported agent transcript format"
    if not _is_sha256(document.get("root_sha256")):
        return False, "agent transcript root is invalid"
    entries = document.get("entries")
    if not isinstance(entries, list) or len(entries) < 2:
        return False, "agent transcript requires start and close entries"
    if document.get("entry_count") != len(entries):
        return False, "agent transcript entry_count mismatch"

    previous_hash = ZERO_HASH
    expected_turn = 1
    expected_generation_attempt = 1
    close_seen = False

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            return False, "agent transcript entry must be an object"
        if set(entry) != {
            "index",
            "kind",
            "previous_hash",
            "payload_sha256",
            "payload",
            "entry_hash",
        }:
            return False, "agent transcript entry fields are invalid"
        if entry.get("index") != index:
            return False, "agent transcript index mismatch"
        kind = entry.get("kind")
        payload = entry.get("payload")
        if not isinstance(kind, str) or not isinstance(payload, dict):
            return False, "agent transcript entry shape is invalid"
        if entry.get("previous_hash") != previous_hash:
            return False, "agent transcript previous_hash mismatch"
        if not _is_sha256(entry.get("payload_sha256")):
            return False, "agent transcript payload digest is invalid"
        if not _is_sha256(entry.get("entry_hash")):
            return False, "agent transcript entry hash is invalid"

        payload_sha = sha256_json(payload)
        if entry.get("payload_sha256") != payload_sha:
            return False, "agent transcript payload hash mismatch"
        expected_hash = sha256_json(
            {
                "index": index,
                "kind": kind,
                "previous_hash": previous_hash,
                "payload_sha256": payload_sha,
            }
        )
        if entry.get("entry_hash") != expected_hash:
            return False, "agent transcript entry hash mismatch"

        if close_seen:
            return False, "agent transcript has entries after close"

        if index == 0:
            if kind != "AGENT_START":
                return False, "agent transcript must start with AGENT_START"
            if set(payload) != {
                "session_id",
                "agent_task_view_sha256",
            }:
                return False, "agent transcript start fields are invalid"
            session_id = payload.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                return False, "agent transcript start missing session_id"
            if not _is_sha256(payload.get("agent_task_view_sha256")):
                return False, "agent transcript start task commitment is invalid"
        elif kind == "AGENT_DECISION":
            if set(payload) != {
                "turn",
                "observation_frontier_sha256",
                "trajectory_sha256",
                "decision",
            }:
                return False, "agent decision transcript fields are invalid"
            if payload.get("turn") != expected_turn:
                return False, "agent transcript turn is not contiguous"
            if not _is_sha256(payload.get("observation_frontier_sha256")):
                return False, "agent decision observation frontier is invalid"
            if not _is_sha256(payload.get("trajectory_sha256")):
                return False, "agent decision trajectory digest is invalid"
            decision = payload.get("decision")
            if not isinstance(decision, dict):
                return False, "agent transcript decision payload is invalid"
            if set(decision) != {"kind", "action"}:
                return False, "agent transcript decision fields are invalid"
            kind_raw = decision.get("kind")
            if not isinstance(kind_raw, str):
                return False, "agent transcript decision kind is invalid"
            try:
                decision_kind = DecisionKind(kind_raw)
            except ValueError:
                return False, "agent transcript decision kind is unsupported"
            action = decision.get("action")
            if decision_kind is DecisionKind.SELECT:
                if not isinstance(action, str) or not action:
                    return False, "agent transcript SELECT action is invalid"
            elif action is not None:
                return False, "agent transcript non-SELECT action must be null"
            expected_turn += 1
        elif kind == "AGENT_GENERATION":
            if set(payload) != {
                "turn",
                "observation_frontier_sha256",
                "attempt",
                "candidate_sha256",
            }:
                return False, "agent generation transcript fields are invalid"
            if payload.get("turn") != expected_turn:
                return False, "agent transcript turn is not contiguous"
            if not _is_sha256(payload.get("observation_frontier_sha256")):
                return False, "agent generation observation frontier is invalid"
            if payload.get("attempt") != expected_generation_attempt:
                return False, "agent generation attempt is not contiguous"
            if not _is_sha256(payload.get("candidate_sha256")):
                return False, "agent generation candidate digest is invalid"
            previous_entry = entries[index - 1] if index > 0 else None
            if (
                not isinstance(previous_entry, dict)
                or previous_entry.get("kind") != "AGENT_DECISION"
            ):
                return False, "agent generation must follow a decision"
            previous_payload = previous_entry.get("payload")
            if not isinstance(previous_payload, dict):
                return False, "agent generation preceding decision is invalid"
            previous_decision = previous_payload.get("decision")
            if (
                not isinstance(previous_decision, dict)
                or previous_decision.get("kind") != DecisionKind.CALL_GENERATIVE.value
            ):
                return False, "agent generation must follow CALL_GENERATIVE"
            if (
                payload.get("observation_frontier_sha256")
                != previous_payload.get("observation_frontier_sha256")
            ):
                return False, "agent generation frontier differs from decision frontier"
            expected_generation_attempt += 1
            expected_turn += 1
        elif kind == "AGENT_CLOSE":
            if set(payload) != {"session_id", "closed"}:
                return False, "agent close transcript fields are invalid"
            if index != len(entries) - 1:
                return False, "agent close must be terminal transcript entry"
            if payload.get("closed") is not True:
                return False, "agent close acknowledgement is invalid"
            if payload.get("session_id") != entries[0]["payload"].get("session_id"):
                return False, "agent close session mismatch"
            close_seen = True
        else:
            return False, f"unsupported agent transcript event: {kind}"

        previous_hash = expected_hash

    if not close_seen:
        return False, "agent transcript has no close entry"
    if document.get("root_sha256") != previous_hash:
        return False, "agent transcript root mismatch"
    return True, "agent transcript valid"


def verify_agent_transcript_against_episode(
    document: Mapping[str, Any],
    *,
    task: AgentTaskView,
    episode: EpisodeResult,
) -> tuple[bool, str]:
    valid, reason = verify_agent_transcript_document(document)
    if not valid:
        return False, reason

    entries = document["entries"]
    start_payload = entries[0]["payload"]
    if start_payload.get("agent_task_view_sha256") != sha256_json(asdict(task)):
        return False, "agent transcript task commitment mismatch"

    transcript_decisions = [
        entry["payload"]
        for entry in entries
        if entry["kind"] == "AGENT_DECISION"
    ]
    ledger_decisions = []
    actions_so_far: list[str] = []
    for entry in episode.ledger_entries:
        if entry.kind == "DECISION":
            ledger_decisions.append(
                {
                    "payload": entry.payload,
                    "trajectory_sha256": sha256_json(actions_so_far),
                }
            )
        elif entry.kind == "ACTION_RESULT":
            action = entry.payload.get("action")
            if not isinstance(action, str):
                return False, "episode ledger action result is invalid"
            actions_so_far.append(action)

    if len(transcript_decisions) != len(ledger_decisions):
        return False, "agent transcript decision count mismatch"

    for agent_decision, ledger_record in zip(
        transcript_decisions,
        ledger_decisions,
    ):
        ledger_decision = ledger_record["payload"]
        if (
            agent_decision.get("observation_frontier_sha256")
            != ledger_decision.get("observation_frontier_sha256")
        ):
            return False, "agent transcript observation frontier mismatch"
        if agent_decision.get("decision") != ledger_decision.get("decision"):
            return False, "agent transcript decision mismatch"
        if (
            agent_decision.get("trajectory_sha256")
            != ledger_record["trajectory_sha256"]
        ):
            return False, "agent transcript trajectory digest mismatch"

    generation_payloads = [
        entry["payload"]
        for entry in entries
        if entry["kind"] == "AGENT_GENERATION"
    ]
    if len(generation_payloads) != episode.generative_calls:
        return False, "agent transcript generation count mismatch"
    if len(episode.generated_candidates) != episode.generative_calls:
        return False, "episode generated candidate count mismatch"

    candidate_hashes = [
        hashlib.sha256(candidate.encode("utf-8")).hexdigest()
        for candidate in episode.generated_candidates
    ]
    transcript_hashes = [
        payload["candidate_sha256"]
        for payload in generation_payloads
    ]
    if transcript_hashes != candidate_hashes:
        return False, "agent transcript candidate digest mismatch"

    ledger_candidate_hashes = [
        entry.payload["candidate_sha256"]
        for entry in episode.ledger_entries
        if entry.kind == "ACTION_RESULT"
        and "candidate_sha256" in entry.payload
    ]
    if ledger_candidate_hashes != candidate_hashes:
        return False, "agent transcript candidate ledger binding mismatch"

    return True, "agent transcript bound to episode"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )
