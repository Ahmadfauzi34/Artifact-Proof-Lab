from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .adaptive_policy import ProofGatedAdaptivePolicy, UNKNOWN_STATE
from .core import AgentTaskView, DecisionKind, EpisodeResult, Observation, PolicyDecision
from .ledger import sha256_json


@dataclass(frozen=True)
class SearchReceipt:
    task_id: str
    ledger_root_sha256: str
    state: str
    intent: str
    previous_cursor: int
    next_cursor: int


class BoundedExplorationPolicy:
    """On-policy actor with proof-gated search bookkeeping and Q exploitation.

    Q state changes only through ProofGatedAdaptivePolicy.update() on fully
    admitted train episodes. Rejected episodes may advance a separate search
    cursor only when provenance, trajectory, and integrity all validate.
    """

    _CANDIDATES = {
        "PROCESS_STATE_AMBIGUOUS": ("INSPECT_PROCESS_STATE",),
        "PROCESS_STOPPED_CONFIRMED": ("SAFE_DEFER", "RESTART_PROCESS"),
        "PROCESS_RESTARTED_NEEDS_VERIFY": ("SAFE_DEFER", "RETRY_HEALTHCHECK"),
        "AUTHORITY_UNKNOWN": ("INSPECT_AUTHORITY",),
        "AUTHORITY_SLOT_1": ("SELECT_AUTHORITY_SLOT_2", "SELECT_AUTHORITY_SLOT_1"),
        "AUTHORITY_SLOT_2": ("SELECT_AUTHORITY_SLOT_1", "SELECT_AUTHORITY_SLOT_2"),
        "REPO_NEEDS_EXECUTION": ("RUN_TARGETED_TEST",),
        "REPO_NEEDS_REPAIR": ("RUN_REGRESSION", "APPLY_CANDIDATE_FIX"),
        "REPO_NEEDS_REGRESSION": ("SAFE_DEFER", "RUN_REGRESSION"),
        "SYNTHESIS_REQUIRED": ("SAFE_DEFER", "CALL_GENERATIVE"),
        "NO_SUPPORTED_ACTUATOR": ("SAFE_DEFER",),
    }

    def __init__(self, learner: ProofGatedAdaptivePolicy | None = None) -> None:
        self.learner = learner or ProofGatedAdaptivePolicy()
        self._cursor: dict[str, int] = {}
        self._search_receipts: list[SearchReceipt] = []

    @property
    def search_receipts(self) -> tuple[SearchReceipt, ...]:
        return tuple(self._search_receipts)

    @property
    def search_sha256(self) -> str:
        return sha256_json(
            {
                "format": "proof-gym-bounded-search-v1",
                "cursor": dict(sorted(self._cursor.items())),
            }
        )

    def cursor(self, state: str) -> int:
        return self._cursor.get(state, 0)

    def decide(
        self,
        task: AgentTaskView,
        observations: tuple[Observation, ...],
        trajectory: tuple[str, ...],
    ) -> PolicyDecision:
        learned = self.learner.decide(task, observations, trajectory)
        if learned.kind not in {DecisionKind.DEFER_NO_SUPPORT, DecisionKind.DEFER_TIE}:
            return learned

        state = self.learner.encoder.encode(task.domain, observations)
        candidates = self._supported_candidates(state, task)
        if not candidates:
            return PolicyDecision(DecisionKind.DEFER_NO_SUPPORT)
        index = min(self.cursor(state), len(candidates) - 1)
        decision = self.learner.adapter.intent_to_decision(candidates[index], task)
        if decision is None:
            return PolicyDecision(DecisionKind.DEFER_NO_SUPPORT)
        return decision

    def observe_failed_attempt(self, result: EpisodeResult) -> bool:
        if result.split != "train":
            raise ValueError("search bookkeeping accepts train attempts only")
        if result.accepted:
            raise ValueError("accepted episode must teach Q rather than advance search")
        if not (
            result.admission.provenance_valid
            and result.admission.trajectory_valid
            and result.admission.integrity_valid
        ):
            return False
        if result.ledger_root_sha256 is None:
            return False

        trace = self._decision_trace(result)
        if not trace:
            return False
        state, intent = trace[-1]
        if state == UNKNOWN_STATE or intent is None:
            return False
        if self.learner.q_value(state, intent) > 0.0:
            # A rejected exploit is not silently converted into negative Q credit.
            return False

        previous = self.cursor(state)
        self._cursor[state] = previous + 1
        self._search_receipts.append(
            SearchReceipt(
                task_id=result.task_id,
                ledger_root_sha256=result.ledger_root_sha256,
                state=state,
                intent=intent,
                previous_cursor=previous,
                next_cursor=previous + 1,
            )
        )
        return True

    def _supported_candidates(self, state: str, task: AgentTaskView) -> tuple[str, ...]:
        supported: list[str] = []
        for intent in self._CANDIDATES.get(state, ()):
            if intent == "SAFE_DEFER" and "DEFER" not in task.allowed_actions:
                continue
            if self.learner.adapter.intent_to_decision(intent, task) is not None:
                supported.append(intent)
        return tuple(supported)

    def _decision_trace(self, result: EpisodeResult) -> tuple[tuple[str, str | None], ...]:
        frontier: list[Mapping] = []
        trace: list[tuple[str, str | None]] = []
        for entry in result.ledger_entries:
            if entry.kind == "OBSERVATION_BATCH":
                frontier.extend(entry.payload.get("observations", ()))
            elif entry.kind == "DECISION":
                state = self.learner.encoder.encode(result.domain, frontier)
                decision = entry.payload.get("decision")
                intent = None
                if isinstance(decision, Mapping):
                    intent = self.learner.adapter.decision_to_intent(decision)
                trace.append((state, intent))
            elif entry.kind == "ACTION_RESULT":
                frontier.extend(entry.payload.get("observations", ()))
        return tuple(trace)


@dataclass(frozen=True)
class TaskSearchResult:
    task_id: str
    attempts: int
    accepted: bool
    validated_failed_attempts: int
    final_result: EpisodeResult


class OnPolicyTrainer:
    def __init__(
        self,
        gym,
        policy: BoundedExplorationPolicy,
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

    def train_task(self, task) -> TaskSearchResult:
        validated_failed = 0
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
                return TaskSearchResult(
                    task_id=task.task_id,
                    attempts=attempt,
                    accepted=True,
                    validated_failed_attempts=validated_failed,
                    final_result=result,
                )
            if self.policy.observe_failed_attempt(result):
                validated_failed += 1
        assert final_result is not None
        return TaskSearchResult(
            task_id=task.task_id,
            attempts=self.max_attempts_per_task,
            accepted=False,
            validated_failed_attempts=validated_failed,
            final_result=final_result,
        )

    def train(self, tasks: Iterable) -> tuple[TaskSearchResult, ...]:
        return tuple(self.train_task(task) for task in tasks)
