from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .adaptive_policy import ProofGatedAdaptivePolicy, UNKNOWN_STATE
from .core import (
    AgentTaskView,
    DecisionKind,
    EpisodeResult,
    Observation,
    PolicyDecision,
    TerminalReason,
)
from .ledger import LEDGER_FORMAT, sha256_json, verify_ledger_document
from .on_policy import BoundedExplorationPolicy


@dataclass(frozen=True)
class NegativeStat:
    rejections: int


@dataclass(frozen=True)
class NegativeExperienceReceipt:
    task_id: str
    ledger_root_sha256: str
    state: str
    intent: str
    terminal_reason: str
    previous_rejections: int
    next_rejections: int
    before_negative_sha256: str
    after_negative_sha256: str


class NegativeExperienceStore:
    """Typed memory of proof/integrity-valid semantic rejections.

    This store is not truth, proof, reward, or Q. It only records how many
    times a canonical intent was terminally rejected in a semantic state after
    provenance, trajectory, and Artifact-Proof integrity all validated.
    """

    def __init__(self) -> None:
        self._negative: dict[str, dict[str, NegativeStat]] = {}
        self._receipts: list[NegativeExperienceReceipt] = []

    @property
    def receipts(self) -> tuple[NegativeExperienceReceipt, ...]:
        return tuple(self._receipts)

    def rejection_count(self, state: str, intent: str) -> int:
        return self._negative.get(state, {}).get(intent, NegativeStat(0)).rejections

    def document(self) -> dict[str, Any]:
        return {
            "format": "proof-gym-negative-experience-v1",
            "states": {
                state: {
                    intent: {"rejections": stat.rejections}
                    for intent, stat in sorted(row.items())
                }
                for state, row in sorted(self._negative.items())
            },
        }

    @property
    def sha256(self) -> str:
        return sha256_json(self.document())

    def observe(
        self,
        result: EpisodeResult,
        *,
        learner: ProofGatedAdaptivePolicy,
    ) -> NegativeExperienceReceipt | None:
        if result.split != "train":
            raise ValueError("negative experience accepts train attempts only")
        if result.accepted:
            raise ValueError("accepted episode belongs to positive learning")
        if result.admission.semantic_valid:
            return None
        if not (
            result.admission.provenance_valid
            and result.admission.trajectory_valid
            and result.admission.integrity_valid
        ):
            return None
        if result.budget_exhausted:
            return None
        if result.terminal_reason != TerminalReason.ENVIRONMENT_DONE.value:
            return None
        if result.ledger_root_sha256 is None:
            return None

        document = {
            "format": LEDGER_FORMAT,
            "root_sha256": result.ledger_root_sha256,
            "entry_count": result.ledger_entry_count,
            "entries": [entry.to_dict() for entry in result.ledger_entries],
        }
        valid, _reason = verify_ledger_document(document)
        if not valid:
            return None

        cause = _terminal_decision_experience(result, learner)
        if cause is None:
            return None
        state, intent = cause
        if state == UNKNOWN_STATE or intent is None:
            return None

        before = self.sha256
        previous = self.rejection_count(state, intent)
        next_value = previous + 1
        self._negative.setdefault(state, {})[intent] = NegativeStat(next_value)
        after = self.sha256
        receipt = NegativeExperienceReceipt(
            task_id=result.task_id,
            ledger_root_sha256=result.ledger_root_sha256,
            state=state,
            intent=intent,
            terminal_reason=result.terminal_reason,
            previous_rejections=previous,
            next_rejections=next_value,
            before_negative_sha256=before,
            after_negative_sha256=after,
        )
        self._receipts.append(receipt)
        return receipt


class NegativeAwareExplorationPolicy(BoundedExplorationPolicy):
    """G4.1 actor with a separate typed negative-experience preference."""

    def __init__(
        self,
        learner: ProofGatedAdaptivePolicy | None = None,
        negative: NegativeExperienceStore | None = None,
    ) -> None:
        super().__init__(learner)
        self.negative = negative or NegativeExperienceStore()

    @property
    def negative_sha256(self) -> str:
        return self.negative.sha256

    def decide(
        self,
        task: AgentTaskView,
        observations: tuple[Observation, ...],
        trajectory: tuple[str, ...],
    ) -> PolicyDecision:
        learned = self.learner.decide(task, observations, trajectory)
        if learned.kind not in {
            DecisionKind.DEFER_NO_SUPPORT,
            DecisionKind.DEFER_TIE,
        }:
            # Admitted positive Q remains the learned preference. G4.2 negative
            # memory ranks only unresolved exploration and does not erase Q.
            return learned

        state = self.learner.encoder.encode(task.domain, observations)
        candidates = self._supported_candidates(state, task)
        if not candidates:
            return PolicyDecision(DecisionKind.DEFER_NO_SUPPORT)

        ranked = sorted(
            enumerate(candidates),
            key=lambda item: (
                self.negative.rejection_count(state, item[1]),
                item[0],
            ),
        )
        intent = ranked[0][1]
        decision = self.learner.adapter.intent_to_decision(intent, task)
        if decision is None:
            return PolicyDecision(DecisionKind.DEFER_NO_SUPPORT)
        return decision

    def observe_negative_attempt(self, result: EpisodeResult) -> bool:
        return self.negative.observe(
            result,
            learner=self.learner,
        ) is not None


@dataclass(frozen=True)
class NegativeTrainingResult:
    task_id: str
    attempts: int
    accepted: bool
    typed_negative_attempts: int
    final_result: EpisodeResult


class NegativeExperienceTrainer:
    def __init__(
        self,
        gym,
        policy: NegativeAwareExplorationPolicy,
        *,
        generator=None,
        max_attempts_per_task: int = 6,
    ) -> None:
        if max_attempts_per_task < 1:
            raise ValueError("max_attempts_per_task must be positive")
        self.gym = gym
        self.policy = policy
        self.generator = generator
        self.max_attempts_per_task = max_attempts_per_task

    def train_task(self, task) -> NegativeTrainingResult:
        typed_negative = 0
        final_result = None
        for attempt in range(1, self.max_attempts_per_task + 1):
            result = self.gym.run(
                task,
                self.policy,
                generator=self.generator,
                learning_sink=self.policy.learner,
            )
            final_result = result
            if result.accepted:
                return NegativeTrainingResult(
                    task_id=task.task_id,
                    attempts=attempt,
                    accepted=True,
                    typed_negative_attempts=typed_negative,
                    final_result=result,
                )
            if self.policy.observe_negative_attempt(result):
                typed_negative += 1
        assert final_result is not None
        return NegativeTrainingResult(
            task_id=task.task_id,
            attempts=self.max_attempts_per_task,
            accepted=False,
            typed_negative_attempts=typed_negative,
            final_result=final_result,
        )

    def train(self, tasks) -> tuple[NegativeTrainingResult, ...]:
        return tuple(self.train_task(task) for task in tasks)


def _terminal_decision_experience(
    result: EpisodeResult,
    learner: ProofGatedAdaptivePolicy,
) -> tuple[str, str | None] | None:
    frontier: list[Mapping[str, Any]] = []
    decisions: dict[str, tuple[str, str | None]] = {}
    terminal_cause: str | None = None

    for entry in result.ledger_entries:
        if entry.kind == "OBSERVATION_BATCH":
            frontier.extend(entry.payload.get("observations", ()))
        elif entry.kind == "DECISION":
            decision = entry.payload.get("decision")
            intent = None
            if isinstance(decision, Mapping):
                intent = learner.adapter.decision_to_intent(decision)
            state = learner.encoder.encode(result.domain, frontier)
            decisions[entry.entry_hash] = (state, intent)
        elif entry.kind == "ACTION_RESULT":
            if entry.payload.get("done") is True:
                raw = entry.payload.get("decision_entry_hash")
                if isinstance(raw, str):
                    terminal_cause = raw
            frontier.extend(entry.payload.get("observations", ()))
        elif entry.kind == "TERMINAL":
            raw = entry.payload.get("caused_by_decision_entry_hash")
            if isinstance(raw, str):
                terminal_cause = raw

    if terminal_cause is None:
        return None
    return decisions.get(terminal_cause)
