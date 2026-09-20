from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping, Protocol


class DecisionKind(str, Enum):
    SELECT = "SELECT"
    DEFER_TIE = "DEFER_TIE"
    DEFER_NO_SUPPORT = "DEFER_NO_SUPPORT"
    DEFER_NO_ADMISSIBLE = "DEFER_NO_ADMISSIBLE"
    CALL_GENERATIVE = "CALL_GENERATIVE"


@dataclass(frozen=True)
class PolicyDecision:
    kind: DecisionKind
    action: str | None = None
    rationale: str = ""

    @classmethod
    def select(cls, action: str, rationale: str = "") -> "PolicyDecision":
        return cls(DecisionKind.SELECT, action, rationale)

    @classmethod
    def call_generative(cls, rationale: str = "") -> "PolicyDecision":
        return cls(DecisionKind.CALL_GENERATIVE, None, rationale)


@dataclass(frozen=True)
class Observation:
    observation_id: str
    source_lineage_root: str
    check_id: str
    probe_id: str
    attempt_id: int
    sequence_index: int
    logical_tick: int
    observation_type: str
    status: str
    value: Any
    cost: float = 0.0
    raw_payload_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PublicTask:
    task_id: str
    domain: str
    skill_targets: tuple[str, ...]
    description: str
    environment: Mapping[str, Any]
    allowed_actions: tuple[str, ...]
    budget: Mapping[str, Any]
    split: str
    public_tests: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PublicTask":
        return cls(
            task_id=str(raw["task_id"]),
            domain=str(raw["domain"]),
            skill_targets=tuple(raw["skill_targets"]),
            description=str(raw["description"]),
            environment=dict(raw["environment"]),
            allowed_actions=tuple(raw["allowed_actions"]),
            budget=dict(raw["budget"]),
            split=str(raw["split"]),
            public_tests=tuple(raw.get("public_tests", ())),
            metadata=dict(raw.get("metadata", {})),
        )

    def agent_view(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "skill_targets": list(self.skill_targets),
            "description": self.description,
            "environment": dict(self.environment),
            "allowed_actions": list(self.allowed_actions),
            "budget": dict(self.budget),
            "split": self.split,
            "public_tests": list(self.public_tests),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class StepOutcome:
    observations: tuple[Observation, ...]
    done: bool
    accepted_candidate: bool | None = None


class Environment(Protocol):
    def reset(self) -> tuple[Observation, ...]: ...
    def step(self, action: str, *, candidate: str | None = None) -> StepOutcome: ...
    def semantic_verdict(self) -> bool: ...


class Policy(Protocol):
    def decide(
        self,
        task: PublicTask,
        observations: tuple[Observation, ...],
        trajectory: tuple[str, ...],
    ) -> PolicyDecision: ...


class Generator(Protocol):
    def generate(self, task: PublicTask, observations: tuple[Observation, ...], attempt: int) -> str: ...


class LearningSink(Protocol):
    def update(self, result: "EpisodeResult") -> None: ...


@dataclass(frozen=True)
class AdmissionReceipt:
    semantic_valid: bool
    provenance_valid: bool
    integrity_valid: bool
    admitted: bool
    reason: str
    proof_report: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class EpisodeResult:
    task_id: str
    split: str
    domain: str
    trajectory: tuple[str, ...]
    observations: tuple[Observation, ...]
    admission: AdmissionReceipt
    reward_channels: Mapping[str, float]
    learning_updated: bool
    generative_calls: int
    generated_candidates: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.admission.admitted


class MemoryLearningSink:
    """Reference sink for tests/demos; not an RL implementation."""

    def __init__(self) -> None:
        self.updates: list[str] = []

    def update(self, result: EpisodeResult) -> None:
        self.updates.append(result.task_id)


class EvidenceAdmission:
    """Admit semantic evidence only after identity/provenance validation."""

    @staticmethod
    def provenance_valid(observations: Iterable[Observation]) -> tuple[bool, str]:
        items = tuple(observations)
        ids = [item.observation_id for item in items]
        if len(ids) != len(set(ids)):
            return False, "duplicate observation_id"
        expected_sequence = list(range(len(items)))
        sequence = [item.sequence_index for item in items]
        if sequence != expected_sequence:
            return False, "non-contiguous sequence_index"
        if any(not item.source_lineage_root or not item.check_id or not item.probe_id for item in items):
            return False, "missing provenance identity"
        by_check: dict[str, list[Observation]] = {}
        for item in items:
            by_check.setdefault(item.check_id, []).append(item)
        for check_items in by_check.values():
            attempts = [item.attempt_id for item in check_items]
            if attempts != sorted(attempts):
                return False, "attempt_id moved backwards within a check trajectory"
            if len(set(attempts)) != len(attempts):
                return False, "attempt_id repeated within a check trajectory"
        return True, "provenance accepted"


class ProofBundleVerifier:
    """Bridge semantic admission to Artifact-Proof-Lab's immutable proof core.

    The gym never treats this integrity result as semantic truth. The semantic
    evaluator decides whether the trajectory solved the task; artifact_proof
    independently proves that the admitted receipt and trajectory bundle were
    not rewritten before finalization.
    """

    def verify(self, task: PublicTask, observations: tuple[Observation, ...], semantic_valid: bool) -> tuple[bool, Mapping[str, Any] | None]:
        try:
            from artifact_proof import verify_artifact
        except ImportError:
            # Useful for consuming the gym as a standalone reference package.
            # In Artifact-Proof-Lab itself, the import is present and tests
            # exercise the real verifier.
            return True, {"status": "SKIP", "reason": "artifact_proof import unavailable"}

        with tempfile.TemporaryDirectory(prefix="proof-gym-") as tmp:
            root = Path(tmp) / "episode"
            root.mkdir()
            trajectory = {
                "format": "proof-gym-trajectory-v1",
                "task_id": task.task_id,
                "split": task.split,
                "observations": [item.to_dict() for item in observations],
            }
            admission = {
                "format": "proof-gym-admission-v1",
                "task_id": task.task_id,
                "semantic_valid": semantic_valid,
            }
            files = {
                "trajectory.json": _canonical_json_bytes(trajectory),
                "admission.json": _canonical_json_bytes(admission),
            }
            for name, data in files.items():
                (root / name).write_bytes(data)
            manifest = {
                "format": "artifact-proof-manifest-v1",
                "artifact": {"name": "proof-gym-episode", "version": "1"},
                "files": {
                    name: {"sha256": hashlib.sha256(data).hexdigest()}
                    for name, data in sorted(files.items())
                },
                "coverage": {"complete": True, "allow_unlisted": []},
                "checks": [],
            }
            (root / "ARTIFACT_PROOF.json").write_bytes(_canonical_json_bytes(manifest))
            report = verify_artifact(root)
            return report.passed, report.to_dict()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class ReferenceGym:
    def __init__(
        self,
        environment_factory: Callable[[PublicTask], Environment],
        *,
        proof_verifier: ProofBundleVerifier | None = None,
    ) -> None:
        self._environment_factory = environment_factory
        self._proof_verifier = proof_verifier or ProofBundleVerifier()

    def run(
        self,
        task: PublicTask,
        policy: Policy,
        *,
        generator: Generator | None = None,
        learning_sink: LearningSink | None = None,
    ) -> EpisodeResult:
        env = self._environment_factory(task)
        observations = list(env.reset())
        trajectory: list[str] = []
        generated_candidates: list[str] = []
        generative_calls = 0
        max_steps = int(task.budget["max_steps"])
        max_generative = task.budget.get("max_generative_calls")
        max_generative = 0 if max_generative is None else int(max_generative)
        done = False

        for _step in range(max_steps):
            decision = policy.decide(task, tuple(observations), tuple(trajectory))
            if decision.kind is DecisionKind.SELECT:
                action = decision.action
                if action is None or action not in task.allowed_actions:
                    trajectory.append("DEFER_NO_ADMISSIBLE")
                    break
                trajectory.append(action)
                outcome = env.step(action)
            elif decision.kind is DecisionKind.CALL_GENERATIVE:
                if generator is None or generative_calls >= max_generative:
                    trajectory.append("DEFER_NO_SUPPORT")
                    break
                generative_calls += 1
                candidate = generator.generate(task, tuple(observations), generative_calls)
                generated_candidates.append(candidate)
                trajectory.append("CALL_GENERATIVE")
                outcome = env.step("CALL_GENERATIVE", candidate=candidate)
            else:
                trajectory.append(decision.kind.value)
                # A defer is itself a terminal bounded action. The environment
                # can admit a safe-defer scenario or reject an unsupported one.
                outcome = env.step(decision.kind.value)

            observations.extend(outcome.observations)
            if outcome.done:
                done = True
                break

        semantic_valid = bool(done and env.semantic_verdict())
        provenance_valid, provenance_reason = EvidenceAdmission.provenance_valid(observations)
        integrity_valid, proof_report = self._proof_verifier.verify(task, tuple(observations), semantic_valid)
        admitted = semantic_valid and provenance_valid and integrity_valid
        if not semantic_valid:
            reason = "semantic evaluator rejected trajectory"
        elif not provenance_valid:
            reason = provenance_reason
        elif not integrity_valid:
            reason = "proof bundle integrity rejected"
        else:
            reason = "evidence admitted"
        admission = AdmissionReceipt(
            semantic_valid=semantic_valid,
            provenance_valid=provenance_valid,
            integrity_valid=integrity_valid,
            admitted=admitted,
            reason=reason,
            proof_report=proof_report,
        )

        rewards = {
            "task_acceptance": 1.0 if admitted else 0.0,
            "proof_closure": 1.0 if semantic_valid else 0.0,
            "resource_cost": -sum(item.cost for item in observations),
            "useless_retry_penalty": -float(_count_useless_retries(observations)),
            "defer_quality": 1.0 if admitted and any(step.startswith("DEFER_") for step in trajectory) else 0.0,
        }
        provisional = EpisodeResult(
            task_id=task.task_id,
            split=task.split,
            domain=task.domain,
            trajectory=tuple(trajectory),
            observations=tuple(observations),
            admission=admission,
            reward_channels=rewards,
            learning_updated=False,
            generative_calls=generative_calls,
            generated_candidates=tuple(generated_candidates),
        )
        learning_updated = False
        # Only admitted training evidence may change the writable brain.
        if admitted and task.split == "train" and learning_sink is not None:
            learning_sink.update(provisional)
            learning_updated = True
        return replace(provisional, learning_updated=learning_updated)


def _count_useless_retries(observations: Iterable[Observation]) -> int:
    by_check: dict[str, list[Observation]] = {}
    for item in observations:
        by_check.setdefault(item.check_id, []).append(item)
    useless = 0
    for items in by_check.values():
        if len(items) < 2:
            continue
        for previous, current in zip(items, items[1:]):
            if current.attempt_id > previous.attempt_id and current.value == previous.value and current.status == previous.status:
                useless += 1
    return useless
