from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .agent_transcript import AGENT_TRANSCRIPT_FORMAT
from .core import PublicTask, TerminalReason
from .isolated_agent import IsolatedEpisodeResult
from .ledger import sha256_json


EXTERNAL_TRAINING_RECEIPT_FORMAT = "proof-gym-external-training-receipt-v1"
POSITIVE_EXPERIENCE = "positive"
TYPED_NEGATIVE_EXPERIENCE = "typed_negative"
NO_LEARNING_EXPERIENCE = "none"


@dataclass(frozen=True)
class ExternalTrainingReceipt:
    task_id: str
    split: str
    domain: str
    experience_kind: str
    learning_admissible: bool
    accepted: bool
    terminal_reason: str
    budget_exhausted: bool
    semantic_valid: bool
    provenance_valid: bool
    trajectory_valid: bool
    integrity_valid: bool
    ledger_root_sha256: str
    agent_transcript_root_sha256: str
    agent_task_view_sha256: str
    reward_channels: Mapping[str, float]
    trajectory: tuple[str, ...]
    decision_trace: tuple[Mapping[str, Any], ...]
    reason: str

    def document(self) -> dict[str, Any]:
        body = {
            "format": EXTERNAL_TRAINING_RECEIPT_FORMAT,
            "task_id": self.task_id,
            "split": self.split,
            "domain": self.domain,
            "experience_kind": self.experience_kind,
            "learning_admissible": self.learning_admissible,
            "accepted": self.accepted,
            "terminal_reason": self.terminal_reason,
            "budget_exhausted": self.budget_exhausted,
            "admission": {
                "semantic_valid": self.semantic_valid,
                "provenance_valid": self.provenance_valid,
                "trajectory_valid": self.trajectory_valid,
                "integrity_valid": self.integrity_valid,
            },
            "ledger_root_sha256": self.ledger_root_sha256,
            "agent_transcript_format": AGENT_TRANSCRIPT_FORMAT,
            "agent_transcript_root_sha256": self.agent_transcript_root_sha256,
            "agent_task_view_sha256": self.agent_task_view_sha256,
            "reward_channels": dict(sorted(self.reward_channels.items())),
            "trajectory": list(self.trajectory),
            "decision_trace": [dict(item) for item in self.decision_trace],
            "reason": self.reason,
            "gym_applied_learning": False,
        }
        body["receipt_sha256"] = sha256_json(body)
        return body


def build_external_training_receipt(
    task: PublicTask,
    isolated: IsolatedEpisodeResult,
) -> ExternalTrainingReceipt:
    if task.split != "train":
        raise ValueError("external training receipts accept train tasks only")
    episode = isolated.episode
    if episode.task_id != task.task_id or episode.split != "train":
        raise ValueError("episode/task identity mismatch")
    if not isolated.agent_transcript_valid:
        raise ValueError("agent transcript must validate before receipt creation")
    if not episode.ledger_root_sha256:
        raise ValueError("episode is missing ledger root")

    admission = episode.admission
    positive = bool(
        episode.accepted
        and admission.semantic_valid
        and admission.provenance_valid
        and admission.trajectory_valid
        and admission.integrity_valid
    )
    typed_negative = bool(
        not episode.accepted
        and not admission.semantic_valid
        and admission.provenance_valid
        and admission.trajectory_valid
        and admission.integrity_valid
        and not episode.budget_exhausted
        and episode.terminal_reason == TerminalReason.ENVIRONMENT_DONE.value
    )

    if positive:
        experience_kind = POSITIVE_EXPERIENCE
        learning_admissible = True
        reason = "fully admitted train episode"
    elif typed_negative:
        experience_kind = TYPED_NEGATIVE_EXPERIENCE
        learning_admissible = True
        reason = "proof/integrity-valid terminal semantic rejection"
    else:
        experience_kind = NO_LEARNING_EXPERIENCE
        learning_admissible = False
        reason = "episode is not admissible learning evidence"

    start_entries = [
        entry
        for entry in isolated.agent_transcript_entries
        if entry.get("kind") == "AGENT_START"
    ]
    if len(start_entries) != 1:
        raise ValueError("agent transcript start entry missing")
    start_payload = start_entries[0].get("payload")
    if not isinstance(start_payload, Mapping):
        raise ValueError("agent transcript start payload invalid")
    task_sha = start_payload.get("agent_task_view_sha256")
    if not isinstance(task_sha, str):
        raise ValueError("agent task view commitment missing")

    trace = tuple(
        dict(entry)
        for entry in isolated.agent_transcript_entries
        if entry.get("kind") in {"AGENT_DECISION", "AGENT_GENERATION"}
    )

    return ExternalTrainingReceipt(
        task_id=episode.task_id,
        split=episode.split,
        domain=episode.domain,
        experience_kind=experience_kind,
        learning_admissible=learning_admissible,
        accepted=episode.accepted,
        terminal_reason=episode.terminal_reason,
        budget_exhausted=episode.budget_exhausted,
        semantic_valid=admission.semantic_valid,
        provenance_valid=admission.provenance_valid,
        trajectory_valid=admission.trajectory_valid,
        integrity_valid=admission.integrity_valid,
        ledger_root_sha256=episode.ledger_root_sha256,
        agent_transcript_root_sha256=isolated.agent_transcript_root_sha256,
        agent_task_view_sha256=task_sha,
        reward_channels=episode.reward_channels,
        trajectory=episode.trajectory,
        decision_trace=trace,
        reason=reason,
    )


def verify_external_training_receipt(
    document: Mapping[str, Any],
) -> tuple[bool, str]:
    required = {
        "format",
        "task_id",
        "split",
        "domain",
        "experience_kind",
        "learning_admissible",
        "accepted",
        "terminal_reason",
        "budget_exhausted",
        "admission",
        "ledger_root_sha256",
        "agent_transcript_format",
        "agent_transcript_root_sha256",
        "agent_task_view_sha256",
        "reward_channels",
        "trajectory",
        "decision_trace",
        "reason",
        "gym_applied_learning",
        "receipt_sha256",
    }
    if set(document) != required:
        return False, "external training receipt fields are invalid"
    if document.get("format") != EXTERNAL_TRAINING_RECEIPT_FORMAT:
        return False, "unsupported external training receipt format"
    if document.get("split") != "train":
        return False, "external training receipt is not train split"
    if document.get("gym_applied_learning") is not False:
        return False, "gym must not claim to mutate the external agent"

    supplied_sha = document.get("receipt_sha256")
    if not _is_sha256(supplied_sha):
        return False, "external training receipt digest is invalid"
    payload = dict(document)
    payload.pop("receipt_sha256")
    if sha256_json(payload) != supplied_sha:
        return False, "external training receipt digest mismatch"

    for field in (
        "ledger_root_sha256",
        "agent_transcript_root_sha256",
        "agent_task_view_sha256",
    ):
        if not _is_sha256(document.get(field)):
            return False, f"{field} is invalid"

    admission = document.get("admission")
    if not isinstance(admission, Mapping) or set(admission) != {
        "semantic_valid",
        "provenance_valid",
        "trajectory_valid",
        "integrity_valid",
    }:
        return False, "external training admission dimensions are invalid"
    if any(not isinstance(admission[key], bool) for key in admission):
        return False, "external training admission dimensions must be booleans"

    kind = document.get("experience_kind")
    admissible = document.get("learning_admissible")
    accepted = document.get("accepted")
    budget_exhausted = document.get("budget_exhausted")
    terminal_reason = document.get("terminal_reason")
    if not isinstance(admissible, bool) or not isinstance(accepted, bool):
        return False, "external training receipt booleans are invalid"
    if not isinstance(budget_exhausted, bool) or not isinstance(terminal_reason, str):
        return False, "external training terminal fields are invalid"

    all_proof = bool(
        admission["provenance_valid"]
        and admission["trajectory_valid"]
        and admission["integrity_valid"]
    )
    if kind == POSITIVE_EXPERIENCE:
        if not (
            admissible
            and accepted
            and admission["semantic_valid"]
            and all_proof
        ):
            return False, "positive receipt violates admission gate"
    elif kind == TYPED_NEGATIVE_EXPERIENCE:
        if not (
            admissible
            and not accepted
            and not admission["semantic_valid"]
            and all_proof
            and not budget_exhausted
            and terminal_reason == TerminalReason.ENVIRONMENT_DONE.value
        ):
            return False, "typed-negative receipt violates negative gate"
    elif kind == NO_LEARNING_EXPERIENCE:
        if admissible:
            return False, "non-learning receipt cannot be learning-admissible"
    else:
        return False, "unsupported external training experience kind"

    if not isinstance(document.get("trajectory"), list):
        return False, "external training trajectory must be a list"
    if not isinstance(document.get("decision_trace"), list):
        return False, "external training decision trace must be a list"
    if not isinstance(document.get("reward_channels"), Mapping):
        return False, "external training reward channels must be an object"
    return True, "external training receipt valid"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )
