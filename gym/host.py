from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from .core import Observation, PublicTask, StepOutcome


class _BaseEnvironment:
    def __init__(self, task: PublicTask) -> None:
        self.task = task
        self._sequence = 0
        self._tick = 0
        self._semantic_valid = False
        self._attempt_counters: dict[str, int] = {}
        self._tmp = tempfile.TemporaryDirectory(prefix=f"gym-{task.domain}-")
        self.root = Path(self._tmp.name)

    def __del__(self) -> None:
        try:
            self._tmp.cleanup()
        except Exception:
            pass

    def _obs(
        self,
        *,
        check_id: str,
        probe_id: str,
        attempt_id: int,
        observation_type: str,
        status: str,
        value: Any,
        cost: float = 0.0,
        raw_payload_ref: str | None = None,
    ) -> Observation:
        previous_attempt = self._attempt_counters.get(check_id, 0)
        if attempt_id <= previous_attempt:
            raise ValueError(f"attempt_id must increase for {check_id}: {attempt_id} <= {previous_attempt}")
        self._attempt_counters[check_id] = attempt_id
        item = Observation(
            observation_id=f"obs-{self.task.task_id}-{self._sequence:04d}",
            source_lineage_root=f"lineage-{self.task.environment['snapshot_id']}",
            check_id=check_id,
            probe_id=probe_id,
            attempt_id=attempt_id,
            sequence_index=self._sequence,
            logical_tick=self._tick,
            observation_type=observation_type,
            status=status,
            value=value,
            cost=cost,
            raw_payload_ref=raw_payload_ref,
        )
        self._sequence += 1
        self._tick += 1
        return item

    def _next_attempt(self, check_id: str) -> int:
        return self._attempt_counters.get(check_id, 0) + 1

    def semantic_verdict(self) -> bool:
        return self._semantic_valid


class CliProcessEnvironment(_BaseEnvironment):
    """Same-check retries conflict; inspect process state before destructive repair."""

    def reset(self) -> tuple[Observation, ...]:
        return (
            self._obs(check_id="health", probe_id="health", attempt_id=1, observation_type="healthcheck", status="FAIL", value="timeout"),
            self._obs(check_id="health", probe_id="health", attempt_id=2, observation_type="healthcheck", status="PASS", value="ok"),
        )

    def step(self, action: str, *, candidate: str | None = None) -> StepOutcome:
        if action == "INSPECT_PROCESS_STATE":
            completed = subprocess.run(
                [sys.executable, "-c", "print('running')"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            running = completed.returncode == 0 and completed.stdout.strip() == "running"
            obs = self._obs(
                check_id="process-state",
                probe_id="process-state",
                attempt_id=1,
                observation_type="process_inspection",
                status="PASS" if running else "FAIL",
                value="running" if running else "unknown",
                cost=0.25,
            )
            self._semantic_valid = running
            return StepOutcome((obs,), done=running)
        if action == "RETRY_HEALTHCHECK":
            obs = self._obs(check_id="health", probe_id="health", attempt_id=self._next_attempt("health"), observation_type="healthcheck", status="PASS", value="ok", cost=0.2)
            return StepOutcome((obs,), done=False)
        if action == "RESTART_PROCESS":
            obs = self._obs(check_id="restart", probe_id="restart", attempt_id=1, observation_type="mutation", status="WARN", value="destructive-repair-without-proof", cost=1.0)
            return StepOutcome((obs,), done=False)
        if action.startswith("DEFER_") or action == "DEFER":
            obs = self._obs(check_id="defer", probe_id="defer", attempt_id=1, observation_type="decision", status="FAIL", value=action)
            return StepOutcome((obs,), done=True)
        raise ValueError(f"unsupported CLI action: {action}")


class StructuredDataEnvironment(_BaseEnvironment):
    """Independent probes disagree; provenance inspection resolves authority."""

    def __init__(self, task: PublicTask) -> None:
        super().__init__(task)
        self._checked_provenance = False
        (self.root / "provenance.json").write_text(json.dumps({"source_a": "signed", "source_b": "cache-unanchored"}))

    def reset(self) -> tuple[Observation, ...]:
        return (
            self._obs(check_id="source-a", probe_id="sensor-a", attempt_id=1, observation_type="measurement", status="PASS", value=42),
            self._obs(check_id="source-b", probe_id="sensor-b", attempt_id=1, observation_type="measurement", status="PASS", value=99),
        )

    def step(self, action: str, *, candidate: str | None = None) -> StepOutcome:
        if action == "CHECK_PROVENANCE":
            data = json.loads((self.root / "provenance.json").read_text())
            self._checked_provenance = True
            obs = self._obs(
                check_id="provenance",
                probe_id="provenance",
                attempt_id=1,
                observation_type="provenance",
                status="PASS",
                value=data,
                cost=0.15,
                raw_payload_ref="provenance.json",
            )
            return StepOutcome((obs,), done=False)
        if action in {"SELECT_SOURCE_A", "SELECT_SOURCE_B"}:
            valid = self._checked_provenance and action == "SELECT_SOURCE_A"
            self._semantic_valid = valid
            obs = self._obs(
                check_id="selection",
                probe_id="selection",
                attempt_id=1,
                observation_type="decision",
                status="PROOF_CLOSED" if valid else "FAIL",
                value=action,
                cost=0.05,
            )
            return StepOutcome((obs,), done=True)
        if action.startswith("DEFER_") or action == "DEFER":
            obs = self._obs(check_id="defer", probe_id="defer", attempt_id=1, observation_type="decision", status="FAIL", value=action)
            return StepOutcome((obs,), done=True)
        raise ValueError(f"unsupported data action: {action}")


class RepositoryCodingEnvironment(_BaseEnvironment):
    """Run, mutate, and regress a tiny repository instead of predicting success."""

    def __init__(self, task: PublicTask) -> None:
        super().__init__(task)
        self.module = self.root / "calc.py"
        self.test = self.root / "test_calc.py"
        self.module.write_text("def clamp(x, lo, hi):\n    return min(lo, max(x, hi))\n")
        self.test.write_text(
            "import unittest\nfrom calc import clamp\n\n"
            "class T(unittest.TestCase):\n"
            "    def test_clamp(self):\n"
            "        self.assertEqual(clamp(5, 0, 10), 5)\n"
            "        self.assertEqual(clamp(-1, 0, 10), 0)\n"
            "        self.assertEqual(clamp(20, 0, 10), 10)\n\n"
            "if __name__ == '__main__': unittest.main()\n"
        )
        self._failed_once = False
        self._patched = False

    def reset(self) -> tuple[Observation, ...]:
        return (
            self._obs(check_id="static-review", probe_id="review", attempt_id=1, observation_type="claim", status="PASS", value="implementation looks plausible"),
            self._obs(check_id="historical-regression", probe_id="history", attempt_id=1, observation_type="claim", status="FAIL", value="clamp edge cases previously failed"),
        )

    def _run_test(self, check_id: str) -> Observation:
        completed = subprocess.run(
            [sys.executable, "-B", str(self.test)],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        passed = completed.returncode == 0
        return self._obs(
            check_id=check_id,
            probe_id="unittest",
            attempt_id=self._next_attempt(check_id),
            observation_type="execution",
            status="PASS" if passed else "FAIL",
            value={"returncode": completed.returncode},
            cost=0.5,
            raw_payload_ref="test_calc.py",
        )

    def step(self, action: str, *, candidate: str | None = None) -> StepOutcome:
        if action == "RUN_TARGETED_TEST":
            obs = self._run_test("targeted-test")
            self._failed_once = obs.status == "FAIL"
            return StepOutcome((obs,), done=False)
        if action == "APPLY_CANDIDATE_FIX":
            if not self._failed_once:
                obs = self._obs(check_id="patch", probe_id="patch", attempt_id=self._next_attempt("patch"), observation_type="mutation", status="FAIL", value="patch-before-execution-evidence", cost=0.25)
                return StepOutcome((obs,), done=False)
            self.module.write_text("def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n")
            self._patched = True
            obs = self._obs(check_id="patch", probe_id="patch", attempt_id=self._next_attempt("patch"), observation_type="mutation", status="PASS", value="candidate-applied", cost=0.25, raw_payload_ref="calc.py")
            return StepOutcome((obs,), done=False)
        if action == "RUN_REGRESSION":
            obs = self._run_test("regression")
            self._semantic_valid = self._patched and obs.status == "PASS"
            return StepOutcome((obs,), done=True)
        if action == "ROLLBACK":
            self.module.write_text("def clamp(x, lo, hi):\n    return min(lo, max(x, hi))\n")
            self._patched = False
            obs = self._obs(check_id="rollback", probe_id="rollback", attempt_id=self._next_attempt("rollback"), observation_type="mutation", status="PASS", value="rolled-back", cost=0.25)
            return StepOutcome((obs,), done=False)
        if action.startswith("DEFER_") or action == "DEFER":
            obs = self._obs(check_id="defer", probe_id="defer", attempt_id=1, observation_type="decision", status="FAIL", value=action)
            return StepOutcome((obs,), done=True)
        raise ValueError(f"unsupported repository action: {action}")


class SafeDeferEnvironment(_BaseEnvironment):
    """No admissible action has evidence; a typed defer is the correct terminal state."""

    def reset(self) -> tuple[Observation, ...]:
        return (
            self._obs(check_id="capability", probe_id="capability", attempt_id=1, observation_type="capability", status="FAIL", value="required actuator unavailable"),
        )

    def step(self, action: str, *, candidate: str | None = None) -> StepOutcome:
        valid = action in {"DEFER_NO_SUPPORT", "DEFER_NO_ADMISSIBLE"}
        self._semantic_valid = valid
        obs = self._obs(check_id="defer", probe_id="defer", attempt_id=1, observation_type="decision", status="PROOF_CLOSED" if valid else "FAIL", value=action)
        return StepOutcome((obs,), done=True)


class GenerativeFallbackEnvironment(_BaseEnvironment):
    """Generated text is candidate-only; first candidate is intentionally rejected."""

    def __init__(self, task: PublicTask) -> None:
        super().__init__(task)
        self._candidate_attempt = 0

    def reset(self) -> tuple[Observation, ...]:
        return (
            self._obs(check_id="bounded-actions", probe_id="capability", attempt_id=1, observation_type="capability", status="FAIL", value="no bounded action can synthesize missing key"),
        )

    def step(self, action: str, *, candidate: str | None = None) -> StepOutcome:
        if action != "CALL_GENERATIVE":
            self._semantic_valid = False
            obs = self._obs(check_id="fallback", probe_id="fallback", attempt_id=1, observation_type="decision", status="FAIL", value=action)
            return StepOutcome((obs,), done=True)
        self._candidate_attempt += 1
        valid = candidate == "key=stable-42"
        obs = self._obs(
            check_id="generated-candidate",
            probe_id="hidden-validator",
            attempt_id=self._candidate_attempt,
            observation_type="candidate_validation",
            status="PASS" if valid else "FAIL",
            value={"accepted": valid},
            cost=0.75,
        )
        self._semantic_valid = valid
        return StepOutcome((obs,), done=valid, accepted_candidate=valid)


def create_environment(task: PublicTask):
    runner = str(task.environment.get("runner"))
    if runner != "reference-v1":
        raise ValueError(f"unsupported runner: {runner}")
    snapshot = str(task.environment.get("snapshot_id"))
    if snapshot.startswith("cp-"):
        return CliProcessEnvironment(task)
    if snapshot.startswith("sd-"):
        return StructuredDataEnvironment(task)
    if snapshot.startswith("rc-"):
        return RepositoryCodingEnvironment(task)
    if snapshot.startswith("df-"):
        return SafeDeferEnvironment(task)
    if snapshot.startswith("gf-"):
        return GenerativeFallbackEnvironment(task)
    raise ValueError(f"unknown host snapshot: {snapshot}")
