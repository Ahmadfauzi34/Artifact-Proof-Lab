# Proof-Gated Adaptive Agent Gym — Hardened Reference G1

This directory is an additive training/evaluation surface for Artifact-Proof-Lab.
It does not change proof-core semantics and it does not claim a trained agent.

Reference loop:

    host task descriptor
      -> sanitized AgentTaskView
      -> deterministic reset
      -> observations with identity + provenance
      -> typed policy decision
      -> real environment action
      -> new evidence
      -> host semantic evaluator
      -> provenance admission
      -> Artifact-Proof-Lab integrity bundle
      -> learning update (train split only, admitted evidence only)

Hard boundaries:

    reward != truth
    policy choice != truth
    generated candidate != proof
    integrity proof != semantic correctness
    semantic task success != proof closure
    learning update only after admitted evidence
    validation/holdout never update the writable training sink
    missing proof verifier = FAIL_CLOSED

## Public agent surface vs host surface

Policies and generators receive AgentTaskView, not PublicTask. The agent view
deliberately excludes split labels, skill targets, host snapshot/runner identity,
and benchmark metadata.

Host snapshot IDs are opaque env-hex identifiers and are resolved only by the
reference host registry. The ID itself does not encode the environment family.

The JSON files under gym/tasks are host descriptors for reproducible reference
contract tests. They are intentionally committed to the repository, so they are
not a native-competence benchmark. Every bundled descriptor is marked as
reference_contract with native_competence_claim=false.

For native or private holdout evaluation, stage only the sanitized agent view in
the agent workspace and mount host snapshots/evaluators outside that workspace.

## Reference machine, not restriction

The included environments are a reproducible standard-library reference. They
are not a ban on other runtimes, models, planners, RL implementations, or domain
adapters. Compatible upgrades must preserve the task-view, observation,
typed-decision, reset, evidence-admission, budget, and proof contracts.

## Domains in G1

- cli_process
- structured_data
- repository_coding
- capability_boundary
- planning_workflow
- filesystem_config (reference holdout domain with different local vocabulary)

The conflicting_evidence skill is intentionally exercised across domains; domain
vocabulary is not treated as the skill itself.

## Budget and evidence rules

max_steps, max_generative_calls, and max_tool_cost are executable contracts.
Exceeding tool cost terminates the episode as BUDGET_EXHAUSTED and cannot receive
semantic acceptance or learning credit.

A retry identity is bound to:

    (source_lineage_root, check_id, probe_id)

Reusing one check_id across a different source/probe identity is rejected as
provenance aliasing. Independent probes remain independent evidence.

## Reference score semantics

gym.run_reference is a contract-conformance smoke controller, not the target
agent. Its output explicitly sets evaluation_scope to
reference_contract_conformance_only and native_competence_claim to false.

Do not report reference-controller pass rate as agent capability.

## Run locally

    PYTHONPATH=src:. python -m gym.validate_baseline
    PYTHONPATH=src:. python -m unittest discover -s tests -v
    PYTHONPATH=src:. python -m gym.run_reference

## Upgrade rule

Future gym upgrades should preserve gym/BASELINE_CONTRACT.md unless a later
checkpoint explicitly replaces an invariant with a stricter contract.
