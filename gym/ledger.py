from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping


ZERO_HASH = "0" * 64
LEDGER_FORMAT = "proof-gym-trajectory-ledger-v1"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def observation_frontier_sha256(observations: Iterable[Mapping[str, Any]]) -> str:
    return sha256_json(list(observations))


@dataclass(frozen=True)
class LedgerEntry:
    index: int
    kind: str
    previous_hash: str
    payload_sha256: str
    payload: Mapping[str, Any]
    entry_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "previous_hash": self.previous_hash,
            "payload_sha256": self.payload_sha256,
            "payload": dict(self.payload),
            "entry_hash": self.entry_hash,
        }


class TrajectoryLedger:
    """Append-only hash chain binding evidence, decisions, results, and terminal state."""

    def __init__(
        self,
        *,
        agent_task_view: Mapping[str, Any],
        host_task_commitment: str,
    ) -> None:
        self._entries: list[LedgerEntry] = []
        self.append(
            "EPISODE_START",
            {
                "agent_task_view_sha256": sha256_json(agent_task_view),
                "host_task_commitment": host_task_commitment,
            },
        )

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        return tuple(self._entries)

    @property
    def root_sha256(self) -> str:
        return self._entries[-1].entry_hash if self._entries else ZERO_HASH

    def append(self, kind: str, payload: Mapping[str, Any]) -> LedgerEntry:
        previous_hash = self.root_sha256
        index = len(self._entries)
        payload_copy = dict(payload)
        payload_sha256 = sha256_json(payload_copy)
        entry_hash = sha256_json(
            {
                "index": index,
                "kind": kind,
                "previous_hash": previous_hash,
                "payload_sha256": payload_sha256,
            }
        )
        entry = LedgerEntry(
            index=index,
            kind=kind,
            previous_hash=previous_hash,
            payload_sha256=payload_sha256,
            payload=payload_copy,
            entry_hash=entry_hash,
        )
        self._entries.append(entry)
        return entry

    def record_observation_batch(
        self,
        observations: Iterable[Mapping[str, Any]],
        *,
        phase: str,
    ) -> LedgerEntry:
        items = [dict(item) for item in observations]
        return self.append(
            "OBSERVATION_BATCH",
            {
                "phase": phase,
                "observations": items,
                "observations_sha256": observation_frontier_sha256(items),
            },
        )

    def record_decision(
        self,
        *,
        decision: Mapping[str, Any],
        visible_observations: Iterable[Mapping[str, Any]],
    ) -> LedgerEntry:
        frontier = [dict(item) for item in visible_observations]
        return self.append(
            "DECISION",
            {
                "decision": dict(decision),
                "observation_frontier_sha256": observation_frontier_sha256(frontier),
            },
        )

    def record_action_result(
        self,
        *,
        decision_entry_hash: str,
        action: str,
        observations: Iterable[Mapping[str, Any]],
        done: bool,
        accepted_candidate: bool | None,
        candidate: str | None = None,
    ) -> LedgerEntry:
        items = [dict(item) for item in observations]
        payload: dict[str, Any] = {
            "decision_entry_hash": decision_entry_hash,
            "action": action,
            "done": bool(done),
            "accepted_candidate": accepted_candidate,
            "observations": items,
            "observations_sha256": observation_frontier_sha256(items),
        }
        if candidate is not None:
            payload["candidate_sha256"] = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
        return self.append("ACTION_RESULT", payload)

    def record_terminal(
        self,
        *,
        terminal_reason: str,
        semantic_valid: bool,
        provenance_valid: bool,
        budget_exhausted: bool,
        observations: Iterable[Mapping[str, Any]],
    ) -> LedgerEntry:
        frontier = [dict(item) for item in observations]
        return self.append(
            "TERMINAL",
            {
                "terminal_reason": terminal_reason,
                "semantic_valid": bool(semantic_valid),
                "provenance_valid": bool(provenance_valid),
                "budget_exhausted": bool(budget_exhausted),
                "observation_frontier_sha256": observation_frontier_sha256(frontier),
            },
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "format": LEDGER_FORMAT,
            "root_sha256": self.root_sha256,
            "entry_count": len(self._entries),
            "entries": [entry.to_dict() for entry in self._entries],
        }

    def verify(self) -> tuple[bool, str]:
        return verify_ledger_document(self.to_document())


def verify_ledger_document(document: Mapping[str, Any]) -> tuple[bool, str]:
    if document.get("format") != LEDGER_FORMAT:
        return False, "unsupported trajectory ledger format"
    entries_raw = document.get("entries")
    if not isinstance(entries_raw, list) or not entries_raw:
        return False, "trajectory ledger has no entries"
    if document.get("entry_count") != len(entries_raw):
        return False, "trajectory ledger entry_count mismatch"

    previous_hash = ZERO_HASH
    observed_frontier: list[Mapping[str, Any]] = []
    pending_decision_hash: str | None = None
    terminal_seen = False

    for expected_index, raw in enumerate(entries_raw):
        if not isinstance(raw, dict):
            return False, "trajectory ledger entry must be an object"
        if raw.get("index") != expected_index:
            return False, "trajectory ledger index is not contiguous"
        kind = raw.get("kind")
        payload = raw.get("payload")
        if not isinstance(kind, str) or not isinstance(payload, dict):
            return False, "trajectory ledger entry shape is invalid"
        if raw.get("previous_hash") != previous_hash:
            return False, "trajectory ledger previous_hash mismatch"

        payload_sha256 = sha256_json(payload)
        if raw.get("payload_sha256") != payload_sha256:
            return False, "trajectory ledger payload hash mismatch"
        expected_hash = sha256_json(
            {
                "index": expected_index,
                "kind": kind,
                "previous_hash": previous_hash,
                "payload_sha256": payload_sha256,
            }
        )
        if raw.get("entry_hash") != expected_hash:
            return False, "trajectory ledger entry hash mismatch"

        if terminal_seen:
            return False, "trajectory ledger contains entries after terminal"

        if expected_index == 0:
            if kind != "EPISODE_START":
                return False, "trajectory ledger must start with EPISODE_START"
            if not isinstance(payload.get("host_task_commitment"), str):
                return False, "trajectory ledger missing host task commitment"
            if not isinstance(payload.get("agent_task_view_sha256"), str):
                return False, "trajectory ledger missing agent task view commitment"
        elif kind == "OBSERVATION_BATCH":
            if pending_decision_hash is not None:
                return False, "observation batch cannot bypass pending decision"
            observations = payload.get("observations")
            if not isinstance(observations, list):
                return False, "observation batch payload is invalid"
            if payload.get("observations_sha256") != observation_frontier_sha256(observations):
                return False, "observation batch digest mismatch"
            observed_frontier.extend(observations)
        elif kind == "DECISION":
            if pending_decision_hash is not None:
                return False, "decision cannot replace unresolved decision"
            decision = payload.get("decision")
            if not isinstance(decision, dict):
                return False, "decision payload is invalid"
            if payload.get("observation_frontier_sha256") != observation_frontier_sha256(observed_frontier):
                return False, "decision is not bound to the visible observation frontier"
            pending_decision_hash = expected_hash
        elif kind == "ACTION_RESULT":
            if pending_decision_hash is None:
                return False, "action result has no preceding decision"
            if payload.get("decision_entry_hash") != pending_decision_hash:
                return False, "action result is bound to the wrong decision"
            observations = payload.get("observations")
            if not isinstance(observations, list):
                return False, "action result observations are invalid"
            if payload.get("observations_sha256") != observation_frontier_sha256(observations):
                return False, "action result observation digest mismatch"
            observed_frontier.extend(observations)
            pending_decision_hash = None
        elif kind == "TERMINAL":
            if payload.get("observation_frontier_sha256") != observation_frontier_sha256(observed_frontier):
                return False, "terminal receipt is not bound to final observations"
            terminal_seen = True
        else:
            return False, f"unsupported trajectory ledger entry kind: {kind}"

        previous_hash = expected_hash

    if not terminal_seen:
        return False, "trajectory ledger has no terminal receipt"
    if document.get("root_sha256") != previous_hash:
        return False, "trajectory ledger root mismatch"
    return True, "trajectory ledger verified"
