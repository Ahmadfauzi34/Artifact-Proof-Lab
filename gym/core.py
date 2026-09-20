from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import hashlib
import math
import sys
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Protocol

from .ledger import (
    LedgerEntry,
    TrajectoryLedger,
    canonical_json_bytes,
    observation_frontier_sha256,
    sha256_json,
    verify_ledger_document,
)

if TYPE_CHECKING:
    from .host_boundary import HostGateway


class DecisionKind(str, Enum):
    SELECT = "SELECT"
    DEFER_TIE = "DEFER_TIE"
    DEFER_NO_SUPPORT = "DEFER_NO_SUPPORT"
    DEFER_NO_ADMISSIBLE = "DEFER_NO_ADMISSIBLE"
    CALL_GENERATIVE = "CALL_GENERATIVE"


class TerminalReason(str, Enum):
    ENVIRONMENT_DONE = "ENVIRONMENT_DONE"
    TOOL_COST_BUDGET_EXHAUSTED = "TOOL_COST_BUDGET_EXHAUSTED"
    STEP_BUDGET_EXHAUSTED = "STEP_BUDGET_EXHAUSTED"
    GENERATIVE_BUDGET_EXHAUSTED = "GENERATIVE_BUDGET_EXHAUSTED"
    GENERATOR_UNAVAILABLE = "GENERATOR_UNAVAILABLE"
    ACTION_UNSUPPORTED = "ACTION_UNSUPPORTED"
    INVALID_POLICY_ACTION = "INVALID_POLICY_ACTION"


class HostActionRejected(RuntimeError):
    """Typed host rejection for an action outside the exposed contract."""


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
class AgentTaskView:
    """Only the task surface a policy/generator is allowed to observe."""

    task_id: str
    domain: str
    description: str
    allowed_actions: tuple[str, ...]
    budget: Mapping[str, Any]
    public_tests: tuple[str, ...] = ()
    public_inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class PublicTask:
    """Host-side task descriptor.

    The historical class name is retained for compatibility. Policies and
    generators must receive AgentTaskView, never this descriptor.
    """

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

    def agent_view(self) -> AgentTaskView:
        return AgentTaskView(
            task_id=self.task_id,
            domain=self.domain,
            description=self.description,
            allowed_actions=self.allowed_actions,
            budget=dict(self.budget),
            public_tests=self.public_tests,
            public_inputs=tuple(self.environment.get("public_inputs", ())),
        )


HostTaskDescriptor = PublicTask


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
        task: AgentTaskView,
        observations: tuple[Observation, ...],
        trajectory: tuple[str, ...],
    ) -> PolicyDecision: ...


class Generator(Protocol):
    def generate(self, task: AgentTaskView, observations: tuple[Observation, ...], attempt: int) -> str: ...


class LearningSink(Protocol):
    def update(self, result: "EpisodeResult") -> None: ...


@dataclass(frozen=True)
class AdmissionReceipt:
    semantic_valid: bool
    provenance_valid: bool
    trajectory_valid: bool
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
    terminal_reason: str
    budget_exhausted: bool = False
    generated_candidates: tuple[str, ...] = ()
    ledger_root_sha256: str | None = None
    ledger_entry_count: int = 0
    ledger_entries: tuple[LedgerEntry, ...] = ()

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
        logical_ticks = [item.logical_tick for item in items]
        if logical_ticks != sorted(logical_ticks) or len(logical_ticks) != len(set(logical_ticks)):
            return False, "logical_tick must be strictly monotonic"
        if any(not item.source_lineage_root or not item.check_id or not item.probe_id for item in items):
            return False, "missing provenance identity"
        if any(not math.isfinite(float(item.cost)) or item.cost < 0 for item in items):
            return False, "invalid observation cost"

        check_identity: dict[str, tuple[str, str]] = {}
        by_identity: dict[tuple[str, str, str], list[Observation]] = {}
        for item in items:
            identity = (item.source_lineage_root, item.probe_id)
            previous = check_identity.setdefault(item.check_id, identity)
            if previous != identity:
                return False, "check_id aliases multiple source/probe identities"
            by_identity.setdefault((item.source_lineage_root, item.check_id, item.probe_id), []).append(item)

        for check_items in by_identity.values():
            attempts = [item.attempt_id for item in check_items]
            if attempts != sorted(attempts):
                return False, "attempt_id moved backwards within an evidence identity"
            if len(set(attempts)) != len(attempts):
                return False, "attempt_id repeated within an evidence identity"
        return True, "provenance accepted"


class ProofBundleVerifier:
    """Bridge semantic/trajectory admission to Artifact-Proof-Lab integrity proof."""

    def verify(
        self,
        task: PublicTask,
        observations: tuple[Observation, ...],
        semantic_valid: bool,
        *,
        provenance_valid: bool | None = None,
        ledger_document: Mapping[str, Any] | None = None,
    ) -> tuple[bool, Mapping[str, Any] | None]:
        try:
            from artifact_proof import verify_artifact
        except ImportError as exc:
            return False, {"status": "FAIL", "reason": f"artifact_proof import unavailable: {exc}"}

        ledger_valid_for_bundle = False
        if ledger_document is not None:
            ledger_valid, ledger_reason = verify_ledger_document(ledger_document)
            if not ledger_valid:
                return False, {"status": "FAIL", "reason": f"trajectory ledger rejected: {ledger_reason}"}

            entries = ledger_document.get("entries")
            start_payload = entries[0].get("payload") if isinstance(entries, list) and entries else None
            terminal_payload = entries[-1].get("payload") if isinstance(entries, list) and entries else None
            if not isinstance(start_payload, dict):
                return False, {"status": "FAIL", "reason": "trajectory ledger start payload missing"}
            if not isinstance(terminal_payload, dict):
                return False, {"status": "FAIL", "reason": "trajectory ledger terminal payload missing"}

            expected_agent_view = sha256_json(asdict(task.agent_view()))
            expected_host_task = _host_task_commitment(task)
            if start_payload.get("agent_task_view_sha256") != expected_agent_view:
                return False, {"status": "FAIL", "reason": "trajectory ledger agent task commitment mismatch"}
            if start_payload.get("host_task_commitment") != expected_host_task:
                return False, {"status": "FAIL", "reason": "trajectory ledger host task commitment mismatch"}

            expected_frontier = observation_frontier_sha256(
                [item.to_dict() for item in observations]
            )
            if terminal_payload.get("observation_frontier_sha256") != expected_frontier:
                return False, {"status": "FAIL", "reason": "trajectory ledger observation frontier mismatch"}
            if terminal_payload.get("semantic_valid") != bool(semantic_valid):
                return False, {"status": "FAIL", "reason": "trajectory ledger semantic receipt mismatch"}
            if (
                provenance_valid is not None
                and terminal_payload.get("provenance_valid") != bool(provenance_valid)
            ):
                return False, {"status": "FAIL", "reason": "trajectory ledger provenance receipt mismatch"}
            ledger_valid_for_bundle = True

        with tempfile.TemporaryDirectory(prefix="proof-gym-") as tmp:
            root = Path(tmp) / "episode"
            root.mkdir()
            agent_task_view = asdict(task.agent_view())
            trajectory = {
                "format": "proof-gym-trajectory-v2",
                "task_id": task.task_id,
                "split": task.split,
                "observations": [item.to_dict() for item in observations],
            }
            admission: dict[str, Any] = {
                "format": "proof-gym-admission-v2",
                "task_id": task.task_id,
                "semantic_valid": semantic_valid,
                "host_task_commitment": _host_task_commitment(task),
            }
            if provenance_valid is not None:
                admission["provenance_valid"] = provenance_valid
            files = {
                "agent_task_view.json": canonical_json_bytes(agent_task_view),
                "trajectory.json": canonical_json_bytes(trajectory),
            }
            if ledger_document is not None:
                ledger_bytes = canonical_json_bytes(ledger_document)
                files["ledger.json"] = ledger_bytes
                admission["trajectory_valid"] = ledger_valid_for_bundle
                admission["trajectory_ledger_sha256"] = hashlib.sha256(ledger_bytes).hexdigest()
                admission["trajectory_ledger_root_sha256"] = ledger_document.get("root_sha256")
            files["admission.json"] = canonical_json_bytes(admission)

            for name, data in files.items():
                (root / name).write_bytes(data)
            manifest = {
                "format": "artifact-proof-manifest-v1",
                "artifact": {"name": "proof-gym-episode", "version": "2"},
                "files": {
                    name: {"sha256": hashlib.sha256(data).hexdigest()}
                    for name, data in sorted(files.items())
                },
                "coverage": {"complete": True, "allow_unlisted": []},
                "checks": [],
            }
            (root / "ARTIFACT_PROOF.json").write_bytes(canonical_json_bytes(manifest))
            report = verify_artifact(root)
            return report.passed, report.to_dict()


class ReferenceGym:
    def __init__(
        self,
        environment_factory: Callable[[PublicTask], Environment] | None = None,
        *,
        host_gateway: "HostGateway | None" = None,
        proof_verifier: ProofBundleVerifier | None = None,
    ) -> None:
        if environment_factory is not None and host_gateway is not None:
            raise ValueError("provide environment_factory or host_gateway, not both")
        if host_gateway is None:
            if environment_factory is None:
                raise ValueError("a host gateway or environment factory is required")
            from .host_boundary import LocalReferenceHost

            host_gateway = LocalReferenceHost(environment_factory)
        self._host_gateway = host_gateway
        self._proof_verifier = proof_verifier or ProofBundleVerifier()

    def run(
        self,
        task: PublicTask,
        policy: Policy,
        *,
        generator: Generator | None = None,
        learning_sink: LearningSink | None = None,
    ) -> EpisodeResult:
        host = self._host_gateway
        start = host.start(task)
        try:
            expected_agent_task = task.agent_view()
            if start.agent_task != expected_agent_task:
                raise RuntimeError("host returned an agent task surface inconsistent with the host descriptor")

            agent_task = start.agent_task
            observations = list(start.observations)
            trajectory: list[str] = []
            generated_candidates: list[str] = []
            generative_calls = 0
            max_steps = int(task.budget["max_steps"])
            max_generative = task.budget.get("max_generative_calls")
            max_generative = 0 if max_generative is None else int(max_generative)
            max_tool_cost = task.budget.get("max_tool_cost")
            max_tool_cost = None if max_tool_cost is None else float(max_tool_cost)
            done = False
            budget_exhausted = False
            terminal_reason: TerminalReason | None = None
            terminal_cause_decision_hash: str | None = None
            steps_taken = 0

            ledger = TrajectoryLedger(
                agent_task_view=asdict(agent_task),
                host_task_commitment=_host_task_commitment(task),
            )
            ledger.record_observation_batch(
                [item.to_dict() for item in observations],
                phase="reset",
            )

            if max_tool_cost is not None and _tool_cost(observations) > max_tool_cost:
                budget_exhausted = True
                terminal_reason = TerminalReason.TOOL_COST_BUDGET_EXHAUSTED
                trajectory.append("BUDGET_EXHAUSTED")

            while terminal_reason is None:
                if steps_taken >= max_steps:
                    budget_exhausted = True
                    terminal_reason = TerminalReason.STEP_BUDGET_EXHAUSTED
                    trajectory.append("BUDGET_EXHAUSTED")
                    break

                decision = policy.decide(agent_task, tuple(observations), tuple(trajectory))
                steps_taken += 1
                decision_entry = ledger.record_decision(
                    decision={
                        "kind": decision.kind.value,
                        "action": decision.action,
                    },
                    visible_observations=[item.to_dict() for item in observations],
                )

                candidate: str | None = None
                if decision.kind is DecisionKind.SELECT:
                    action = decision.action
                    if action is None or action not in agent_task.allowed_actions:
                        trajectory.append("DEFER_NO_ADMISSIBLE")
                        terminal_reason = TerminalReason.INVALID_POLICY_ACTION
                        terminal_cause_decision_hash = decision_entry.entry_hash
                        break
                elif decision.kind is DecisionKind.CALL_GENERATIVE:
                    action = "CALL_GENERATIVE"
                    if action not in agent_task.allowed_actions:
                        trajectory.append("DEFER_NO_SUPPORT")
                        terminal_reason = TerminalReason.ACTION_UNSUPPORTED
                        terminal_cause_decision_hash = decision_entry.entry_hash
                        break
                    if generator is None:
                        trajectory.append("DEFER_NO_SUPPORT")
                        terminal_reason = TerminalReason.GENERATOR_UNAVAILABLE
                        terminal_cause_decision_hash = decision_entry.entry_hash
                        break
                    if generative_calls >= max_generative:
                        budget_exhausted = True
                        trajectory.append("BUDGET_EXHAUSTED")
                        terminal_reason = TerminalReason.GENERATIVE_BUDGET_EXHAUSTED
                        terminal_cause_decision_hash = decision_entry.entry_hash
                        break
                    generative_calls += 1
                    candidate = generator.generate(agent_task, tuple(observations), generative_calls)
                    generated_candidates.append(candidate)
                else:
                    action = decision.kind.value

                try:
                    outcome = host.step(start.session_id, action, candidate=candidate)
                except HostActionRejected:
                    trajectory.append("DEFER_NO_ADMISSIBLE")
                    terminal_reason = TerminalReason.INVALID_POLICY_ACTION
                    terminal_cause_decision_hash = decision_entry.entry_hash
                    break

                trajectory.append(action)
                ledger.record_action_result(
                    decision_entry_hash=decision_entry.entry_hash,
                    action=action,
                    observations=[item.to_dict() for item in outcome.observations],
                    done=outcome.done,
                    accepted_candidate=outcome.accepted_candidate,
                    candidate=candidate,
                )
                observations.extend(outcome.observations)

                if max_tool_cost is not None and _tool_cost(observations) > max_tool_cost:
                    budget_exhausted = True
                    trajectory.append("BUDGET_EXHAUSTED")
                    terminal_reason = TerminalReason.TOOL_COST_BUDGET_EXHAUSTED
                    break
                if outcome.done:
                    done = True
                    terminal_reason = TerminalReason.ENVIRONMENT_DONE
                    break

            assert terminal_reason is not None
            semantic_valid = bool(
                done
                and not budget_exhausted
                and terminal_reason is TerminalReason.ENVIRONMENT_DONE
                and host.semantic_verdict(start.session_id)
            )
            provenance_valid, provenance_reason = EvidenceAdmission.provenance_valid(observations)

            ledger.record_terminal(
                terminal_reason=terminal_reason.value,
                semantic_valid=semantic_valid,
                provenance_valid=provenance_valid,
                budget_exhausted=budget_exhausted,
                observations=[item.to_dict() for item in observations],
                caused_by_decision_entry_hash=terminal_cause_decision_hash,
            )
            trajectory_valid, trajectory_reason = ledger.verify()
            ledger_document = ledger.to_document()

            integrity_valid, proof_report = self._proof_verifier.verify(
                task,
                tuple(observations),
                semantic_valid,
                provenance_valid=provenance_valid,
                ledger_document=ledger_document,
            )
            admitted = semantic_valid and provenance_valid and trajectory_valid and integrity_valid

            if budget_exhausted:
                reason = terminal_reason.value
            elif not semantic_valid:
                reason = "semantic evaluator rejected trajectory"
            elif not provenance_valid:
                reason = provenance_reason
            elif not trajectory_valid:
                reason = trajectory_reason
            elif not integrity_valid:
                reason = "proof bundle integrity rejected"
            else:
                reason = "evidence admitted"

            admission = AdmissionReceipt(
                semantic_valid=semantic_valid,
                provenance_valid=provenance_valid,
                trajectory_valid=trajectory_valid,
                integrity_valid=integrity_valid,
                admitted=admitted,
                reason=reason,
                proof_report=proof_report,
            )

            rewards = {
                "task_acceptance": 1.0 if semantic_valid else 0.0,
                "proof_closure": 1.0 if provenance_valid and trajectory_valid and integrity_valid else 0.0,
                "admission_credit": 1.0 if admitted else 0.0,
                "resource_cost": -_tool_cost(observations),
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
                terminal_reason=terminal_reason.value,
                budget_exhausted=budget_exhausted,
                generated_candidates=tuple(generated_candidates),
                ledger_root_sha256=ledger.root_sha256,
                ledger_entry_count=len(ledger.entries),
                ledger_entries=ledger.entries,
            )
            learning_updated = False
            if admitted and task.split == "train" and learning_sink is not None:
                learning_sink.update(provisional)
                learning_updated = True
            return replace(provisional, learning_updated=learning_updated)
        finally:
            primary_failure_active = sys.exc_info()[0] is not None
            try:
                host.close(start.session_id)
            except Exception:
                if not primary_failure_active:
                    raise


def _host_task_commitment(task: PublicTask) -> str:
    return sha256_json(asdict(task))


def _tool_cost(observations: Iterable[Observation]) -> float:
    return sum(float(item.cost) for item in observations)


def _count_useless_retries(observations: Iterable[Observation]) -> int:
    by_identity: dict[tuple[str, str, str], list[Observation]] = {}
    for item in observations:
        by_identity.setdefault((item.source_lineage_root, item.check_id, item.probe_id), []).append(item)
    useless = 0
    for items in by_identity.values():
        if len(items) < 2:
            continue
        for previous, current in zip(items, items[1:]):
            if current.attempt_id > previous.attempt_id and current.value == previous.value and current.status == previous.status:
                useless += 1
    return useless
