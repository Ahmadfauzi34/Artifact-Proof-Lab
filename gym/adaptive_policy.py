from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .core import (
    AgentTaskView,
    DecisionKind,
    EpisodeResult,
    Observation,
    PolicyDecision,
)
from .ledger import LEDGER_FORMAT, sha256_json, verify_ledger_document


UNKNOWN_STATE = "UNKNOWN"


@dataclass(frozen=True)
class QStat:
    visits: int
    value: float


@dataclass(frozen=True)
class LearningReceipt:
    task_id: str
    ledger_root_sha256: str
    before_policy_sha256: str
    after_policy_sha256: str
    updated_pairs: int


class SemanticStateEncoder:
    """Fixed semantic abstraction; deliberately ignores task_id/description/split."""

    def encode(
        self,
        domain: str,
        observations: Iterable[Observation | Mapping[str, Any]],
    ) -> str:
        items = tuple(_obs_dict(item) for item in observations)
        latest: dict[str, Mapping[str, Any]] = {}
        for item in items:
            latest[str(item.get("check_id"))] = item

        generated = latest.get("generated-candidate")
        if "bounded-actions" in latest or (
            generated is not None and generated.get("status") == "FAIL"
        ):
            return "SYNTHESIS_REQUIRED"

        capability = latest.get("capability")
        if capability is not None and capability.get("status") == "FAIL":
            return "NO_SUPPORTED_ACTUATOR"

        if domain == "repository_coding" or any(
            key in latest
            for key in (
                "static-review",
                "historical-regression",
                "targeted-test",
                "patch",
                "regression",
            )
        ):
            patch = latest.get("patch")
            targeted = latest.get("targeted-test")
            if (
                patch is not None
                and patch.get("status") == "PASS"
                and "regression" not in latest
            ):
                return "REPO_NEEDS_REGRESSION"
            if (
                targeted is not None
                and targeted.get("status") == "FAIL"
                and not (
                    patch is not None
                    and patch.get("status") == "PASS"
                )
            ):
                return "REPO_NEEDS_REPAIR"
            if targeted is None:
                return "REPO_NEEDS_EXECUTION"

        process = latest.get("process-state")
        restart = latest.get("restart")
        if process is not None:
            if restart is not None and restart.get("status") == "PASS":
                return "PROCESS_RESTARTED_NEEDS_VERIFY"
            if process.get("value") == "stopped":
                return "PROCESS_STOPPED_CONFIRMED"
            if process.get("value") == "running":
                return "PROCESS_RUNNING_CONFIRMED"
        if "health" in latest:
            return "PROCESS_STATE_AMBIGUOUS"

        provenance = latest.get("provenance") or latest.get("config-origin")
        if provenance is not None and isinstance(provenance.get("value"), dict):
            value = provenance["value"]
            if (
                value.get("source_a") == "signed"
                or value.get("project") == "signed"
            ):
                return "AUTHORITY_SLOT_1"
            if (
                value.get("source_b") == "signed"
                or value.get("cache") == "signed"
            ):
                return "AUTHORITY_SLOT_2"
        if (
            {"source-a", "source-b"} <= set(latest)
            or {"project-config", "cache-config"} <= set(latest)
        ):
            return "AUTHORITY_UNKNOWN"

        return UNKNOWN_STATE


class CanonicalIntentAdapter:
    _RAW_TO_INTENT = {
        "INSPECT_PROCESS_STATE": "INSPECT_PROCESS_STATE",
        "RESTART_PROCESS": "RESTART_PROCESS",
        "RETRY_HEALTHCHECK": "RETRY_HEALTHCHECK",
        "CHECK_PROVENANCE": "INSPECT_AUTHORITY",
        "INSPECT_CONFIG_ORIGIN": "INSPECT_AUTHORITY",
        "SELECT_SOURCE_A": "SELECT_AUTHORITY_SLOT_1",
        "SELECT_PROJECT_CONFIG": "SELECT_AUTHORITY_SLOT_1",
        "SELECT_SOURCE_B": "SELECT_AUTHORITY_SLOT_2",
        "SELECT_CACHE_CONFIG": "SELECT_AUTHORITY_SLOT_2",
        "RUN_TARGETED_TEST": "RUN_TARGETED_TEST",
        "APPLY_CANDIDATE_FIX": "APPLY_CANDIDATE_FIX",
        "RUN_REGRESSION": "RUN_REGRESSION",
    }

    def decision_to_intent(self, decision: Mapping[str, Any]) -> str | None:
        kind = decision.get("kind")
        action = decision.get("action")
        if kind == DecisionKind.CALL_GENERATIVE.value:
            return "CALL_GENERATIVE"
        if kind in {
            DecisionKind.DEFER_TIE.value,
            DecisionKind.DEFER_NO_SUPPORT.value,
            DecisionKind.DEFER_NO_ADMISSIBLE.value,
        }:
            return "SAFE_DEFER"
        if kind == DecisionKind.SELECT.value and isinstance(action, str):
            return self._RAW_TO_INTENT.get(action)
        return None

    def intent_to_decision(
        self,
        intent: str,
        task: AgentTaskView,
    ) -> PolicyDecision | None:
        actions = set(task.allowed_actions)
        if intent == "INSPECT_AUTHORITY":
            for action in ("CHECK_PROVENANCE", "INSPECT_CONFIG_ORIGIN"):
                if action in actions:
                    return PolicyDecision.select(action)
            return None
        if intent == "SELECT_AUTHORITY_SLOT_1":
            for action in ("SELECT_SOURCE_A", "SELECT_PROJECT_CONFIG"):
                if action in actions:
                    return PolicyDecision.select(action)
            return None
        if intent == "SELECT_AUTHORITY_SLOT_2":
            for action in ("SELECT_SOURCE_B", "SELECT_CACHE_CONFIG"):
                if action in actions:
                    return PolicyDecision.select(action)
            return None
        if intent == "CALL_GENERATIVE":
            if "CALL_GENERATIVE" in actions:
                return PolicyDecision.call_generative()
            return None
        if intent == "SAFE_DEFER":
            return PolicyDecision(DecisionKind.DEFER_NO_ADMISSIBLE)
        if intent in actions:
            return PolicyDecision.select(intent)
        return None


class ProofGatedAdaptivePolicy:
    """Deterministic Q learner over proof-admitted canonical trajectories.

    The state abstraction and local action adapter are fixed. Only state/intent
    Q values and visit counts are learned from admitted train EpisodeResults.
    """

    def __init__(self, *, gamma: float = 0.9) -> None:
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be within (0, 1]")
        self.gamma = float(gamma)
        self.encoder = SemanticStateEncoder()
        self.adapter = CanonicalIntentAdapter()
        self._q: dict[str, dict[str, QStat]] = {}
        self._receipts: list[LearningReceipt] = []

    @property
    def learning_receipts(self) -> tuple[LearningReceipt, ...]:
        return tuple(self._receipts)

    def policy_document(self) -> dict[str, Any]:
        return {
            "format": "proof-gym-adaptive-policy-v1",
            "gamma": self.gamma,
            "states": {
                state: {
                    intent: {
                        "visits": stat.visits,
                        "value": stat.value,
                    }
                    for intent, stat in sorted(row.items())
                }
                for state, row in sorted(self._q.items())
            },
        }

    @property
    def policy_sha256(self) -> str:
        return sha256_json(self.policy_document())

    @property
    def learned_pair_count(self) -> int:
        return sum(len(row) for row in self._q.values())

    def q_value(self, state: str, intent: str) -> float:
        return self._q.get(state, {}).get(intent, QStat(0, 0.0)).value

    def decide(
        self,
        task: AgentTaskView,
        observations: tuple[Observation, ...],
        trajectory: tuple[str, ...],
    ) -> PolicyDecision:
        state = self.encoder.encode(task.domain, observations)
        candidates: list[tuple[float, str, PolicyDecision]] = []
        for intent, stat in self._q.get(state, {}).items():
            decision = self.adapter.intent_to_decision(intent, task)
            if decision is not None and stat.visits > 0 and stat.value > 0.0:
                candidates.append((stat.value, intent, decision))

        if not candidates:
            return PolicyDecision(DecisionKind.DEFER_NO_SUPPORT)

        best_value = max(item[0] for item in candidates)
        best = [
            item
            for item in candidates
            if abs(item[0] - best_value) <= 1e-12
        ]
        if len(best) != 1:
            return PolicyDecision(DecisionKind.DEFER_TIE)
        return best[0][2]

    def update(self, result: EpisodeResult) -> None:
        if result.split != "train":
            raise ValueError("adaptive learner accepts train evidence only")
        if not result.accepted:
            raise ValueError("adaptive learner requires admitted episode evidence")
        if not (
            result.admission.semantic_valid
            and result.admission.provenance_valid
            and result.admission.trajectory_valid
            and result.admission.integrity_valid
        ):
            raise ValueError("adaptive learner requires all admission boundaries")

        document = {
            "format": LEDGER_FORMAT,
            "root_sha256": result.ledger_root_sha256,
            "entry_count": result.ledger_entry_count,
            "entries": [
                entry.to_dict()
                for entry in result.ledger_entries
            ],
        }
        valid, reason = verify_ledger_document(document)
        if not valid:
            raise ValueError(
                f"adaptive learner rejected trajectory ledger: {reason}"
            )

        decisions = self._decision_experiences(result)
        if not decisions:
            raise ValueError("admitted episode contains no learnable decisions")

        before = self.policy_sha256
        updated = 0
        count = len(decisions)
        for index, (state, intent) in enumerate(decisions):
            target = self.gamma ** (count - index - 1)
            row = self._q.setdefault(state, {})
            old = row.get(intent, QStat(0, 0.0))
            visits = old.visits + 1
            value = old.value + (target - old.value) / visits
            row[intent] = QStat(
                visits=visits,
                value=value,
            )
            updated += 1

        after = self.policy_sha256
        if result.ledger_root_sha256 is None:
            raise ValueError("admitted episode is missing ledger root")
        self._receipts.append(
            LearningReceipt(
                task_id=result.task_id,
                ledger_root_sha256=result.ledger_root_sha256,
                before_policy_sha256=before,
                after_policy_sha256=after,
                updated_pairs=updated,
            )
        )

    def _decision_experiences(
        self,
        result: EpisodeResult,
    ) -> tuple[tuple[str, str], ...]:
        frontier: list[Mapping[str, Any]] = []
        learned: list[tuple[str, str]] = []

        for entry in result.ledger_entries:
            if entry.kind == "OBSERVATION_BATCH":
                frontier.extend(
                    entry.payload.get("observations", ())
                )
            elif entry.kind == "DECISION":
                decision = entry.payload.get("decision")
                if not isinstance(decision, Mapping):
                    continue
                intent = self.adapter.decision_to_intent(decision)
                if intent is None:
                    continue
                state = self.encoder.encode(
                    result.domain,
                    frontier,
                )
                if state != UNKNOWN_STATE:
                    learned.append((state, intent))
            elif entry.kind == "ACTION_RESULT":
                frontier.extend(
                    entry.payload.get("observations", ())
                )

        return tuple(learned)


def _obs_dict(
    item: Observation | Mapping[str, Any],
) -> Mapping[str, Any]:
    if isinstance(item, Mapping):
        return item
    return item.to_dict()
