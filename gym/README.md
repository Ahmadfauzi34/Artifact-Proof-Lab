# Proof-Gated Adaptive Agent Gym — Reference G1

This directory is an additive training/evaluation surface for Artifact-Proof-Lab.
It does **not** change proof-core semantics and it does not claim a trained agent.

The reference loop is:

```text
Public task
  -> deterministic reset
  -> observations with identity + provenance
  -> typed policy decision
  -> real environment action
  -> new evidence
  -> host semantic evaluator
  -> provenance admission
  -> Artifact-Proof-Lab integrity bundle
  -> learning update (train split only, admitted evidence only)
```

Hard boundaries:

```text
reward != truth
policy choice != truth
generated candidate != proof
integrity proof != semantic correctness
learning update only after admitted evidence
validation/holdout never update the writable training sink
```

## Reference machine, not restriction

The included environments are a reproducible standard-library reference. They are
not a ban on other runtimes, models, planners, RL implementations, or domain
adapters. A new adapter is compatible when it satisfies the same public task,
observation, typed-decision, reset, and evidence-admission contracts.

## Domains in G1

- `cli_process` — temporal retry evidence versus independent process inspection.
- `structured_data` — conflicting independent measurements resolved by provenance.
- `repository_coding` — run -> patch -> regression, with actual subprocess execution.
- `capability_boundary` — safe typed defer when no action is supported.
- `planning_workflow` — generative fallback where generated output remains candidate-only.

The `conflicting_evidence` skill is intentionally exercised in multiple domains;
domain vocabulary is not treated as the skill itself.

## Run locally

From repository root:

```bash
PYTHONPATH=src:. python -m gym.validate_baseline
PYTHONPATH=src:. python -m unittest discover -s tests -v
PYTHONPATH=src:. python -m gym.run_reference
```

`gym.run_reference` is a smoke/reference controller, not the target agent. Replace
`ReferencePolicy` with any policy implementing the typed interface.

## Oracle separation

Public JSON task files contain no expected action, reference patch, semantic
label, or hidden validator detail. Host environment state is instantiated by an
opaque `snapshot_id` and is not part of `PublicTask.agent_view()`. In a real
holdout deployment, host snapshots/evaluators should be mounted outside the
agent workspace; this repository provides the contract and reference host.
