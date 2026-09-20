from __future__ import annotations

from .core import AgentTaskView, DecisionKind, Observation, PolicyDecision


class ReferencePolicy:
    """A deterministic semantic reference, not a privileged answer oracle.

    It consumes only the public task contract and observations exposed through
    normal environment interaction. Consumers may replace this with any model,
    planner, RL policy, or human controller implementing the same typed surface.
    """

    def decide(
        self,
        task: AgentTaskView,
        observations: tuple[Observation, ...],
        trajectory: tuple[str, ...],
    ) -> PolicyDecision:
        actions = set(task.allowed_actions)
        statuses = {(item.check_id, item.status) for item in observations}

        if "INSPECT_PROCESS_STATE" in actions:
            if not any(item.check_id == "process-state" for item in observations):
                return PolicyDecision.select("INSPECT_PROCESS_STATE", "resolve conflicting temporal health evidence with an independent process observation")

        if "CHECK_PROVENANCE" in actions:
            provenance = next((item for item in observations if item.check_id == "provenance"), None)
            if provenance is None:
                return PolicyDecision.select("CHECK_PROVENANCE", "independent probes disagree; inspect source authority")
            if isinstance(provenance.value, dict) and provenance.value.get("source_a") == "signed":
                return PolicyDecision.select("SELECT_SOURCE_A", "select the source supported by admitted provenance")

        if "INSPECT_CONFIG_ORIGIN" in actions:
            origin = next((item for item in observations if item.check_id == "config-origin"), None)
            if origin is None:
                return PolicyDecision.select("INSPECT_CONFIG_ORIGIN", "configuration values conflict; inspect source origin")
            if isinstance(origin.value, dict) and origin.value.get("project") == "signed":
                return PolicyDecision.select("SELECT_PROJECT_CONFIG", "choose the configuration with signed origin")

        if "RUN_TARGETED_TEST" in actions:
            targeted = next((item for item in observations if item.check_id == "targeted-test"), None)
            patch = next((item for item in observations if item.check_id == "patch" and item.status == "PASS"), None)
            regression = next((item for item in observations if item.check_id == "regression"), None)
            if targeted is None:
                return PolicyDecision.select("RUN_TARGETED_TEST", "replace conflicting claims with execution evidence")
            if targeted.status == "FAIL" and patch is None:
                return PolicyDecision.select("APPLY_CANDIDATE_FIX", "mutation is allowed only after failure is demonstrated")
            if patch is not None and regression is None:
                return PolicyDecision.select("RUN_REGRESSION", "candidate must earn acceptance through execution")

        if "CALL_GENERATIVE" in actions:
            return PolicyDecision.call_generative("bounded policy cannot synthesize the missing candidate")

        if "DEFER" in actions or not actions:
            return PolicyDecision(DecisionKind.DEFER_NO_ADMISSIBLE, rationale="no supported action has enough evidence")

        return PolicyDecision(DecisionKind.DEFER_NO_SUPPORT, rationale=f"no reference transition for observed states: {sorted(statuses)}")


class ReferenceGenerator:
    """Reference-only generator used to test reject-before-accept behavior.

    Its outputs are known to the reference host and are never evidence of native
    synthesis competence.
    """

    def generate(self, task: AgentTaskView, observations: tuple[Observation, ...], attempt: int) -> str:
        # The first candidate is intentionally wrong so hidden validation proves
        # that generation is candidate-only. The second is a valid candidate.
        return "key=guess" if attempt == 1 else "key=stable-42"
