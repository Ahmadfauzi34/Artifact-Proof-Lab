# Proof-Gated Adaptive Agent Gym — Hardened G1 + G2 Boundary

This directory is an additive training/evaluation surface for Artifact-Proof-Lab.
It does not change proof-core semantics and it does not claim a trained agent.

Reference loop:

    host task descriptor
      -> private HostGateway
      -> sanitized AgentTaskView
      -> deterministic reset
      -> observations with identity + provenance
      -> typed policy decision
      -> real environment action
      -> new evidence
      -> hash-chained trajectory ledger
      -> host semantic evaluator
      -> provenance admission
      -> trajectory validation
      -> Artifact-Proof-Lab integrity bundle
      -> learning update (train split only, admitted evidence only)

Hard boundaries:

    reward != truth
    policy choice != truth
    generated candidate != proof
    integrity proof != semantic correctness
    trajectory integrity != semantic correctness
    semantic task success != proof closure
    learning update only after admitted evidence
    validation/holdout never update the writable training sink
    missing proof verifier = FAIL_CLOSED

## Public agent surface vs host surface

Policies and generators receive AgentTaskView, not PublicTask. PublicTask is now
explicitly treated as a host-side descriptor; its historical name is retained
for API compatibility.

The agent view deliberately excludes split labels, skill targets, host
snapshot/runner identity, and benchmark metadata.

Host snapshot IDs are opaque env-hex identifiers and are resolved only by the
reference host registry. The ID itself does not encode the environment family.

The JSON files under gym/tasks are host descriptors for reproducible reference
contract tests. They are intentionally committed to the repository, so they are
not a native-competence benchmark. Every bundled descriptor is marked as
reference_contract with native_competence_claim=false.

For native or private holdout evaluation, stage only the sanitized agent surface
in the agent workspace and mount HostGateway implementation, host snapshots,
evaluators, and oracle/reference state outside that workspace.

## G2 private host boundary

ReferenceGym still accepts the old:

    ReferenceGym(create_environment)

form. Internally that path is wrapped by LocalReferenceHost.

The explicit G2 form is:

    ReferenceGym(host_gateway=my_private_host)

A compatible host may run in another process or machine. The policy/generator
contract does not change. LocalReferenceHost is intentionally inspectable and is
reference/regression infrastructure only.

See:

- G1 invariants: gym/BASELINE_CONTRACT.md
- G2 strict-superset invariants: gym/G2_CONTRACT.md
- logical transport-neutral surface: gym/contracts/host_protocol.schema.json

## Proof-gated trajectory ledger

Each episode now carries an append-only ledger:

    EPISODE_START
      -> OBSERVATION_BATCH
      -> DECISION
      -> ACTION_RESULT
      -> ...
      -> TERMINAL

A DECISION commits to the exact observation frontier visible when it was made.
An ACTION_RESULT commits to the hash of the decision that caused it. If a
decision terminates the episode without executing an environment action, the
terminal receipt commits to that unresolved decision hash.

The genesis entry commits to both:

    hash(sanitized AgentTaskView)
    hash(host-side task descriptor)

Artifact-Proof verifies a bundle containing the sanitized task view, trajectory,
ledger, and admission receipt. An internally valid ledger from another task is
therefore rejected rather than accepted as a replay.

The ledger proves ordering and identity of the recorded trajectory. It does not
decide whether an action was semantically correct.

## G2.1 external private-host reference harness

G2.1 adds a strict JSONL subprocess transport for exercising the G2 HostGateway
across a real process boundary.

Reference usage:

    PYTHONPATH=src:. python -m gym.private_host_server --tasks gym/tasks

A controller normally launches that server through:

    SubprocessHostGateway([
        sys.executable,
        "-B",
        "-m",
        "gym.private_host_server",
        "--tasks",
        "gym/tasks",
    ])

The transport performs a versioned HELLO handshake and binds every response to a
request_id. Frames are strict UTF-8 JSON, size bounded, reject non-finite JSON,
and fail closed on malformed envelopes, unexpected fields, request-id mismatch,
EOF, or timeout.

START sends only task_id plus the host descriptor commitment. The response
contains only:

    session_id
    sanitized AgentTaskView
    observations
    host_task_commitment

It does not return split labels, skill targets, snapshot/runner identity,
metadata, evaluator details, or oracle/reference answers.

The reference server is intentionally inspectable and uses the bundled task
registry, so it remains regression/protocol evidence only. For a real private
holdout, place the server implementation, task registry, evaluator, snapshots,
and host files outside the evaluated agent's readable workspace and authority.

Subprocess stdio is a reference coordinate, not a transport restriction. A
socket, RPC framework, container bridge, VM channel, or remote service may
replace it while preserving the HostGateway, sanitization, identity, timeout,
and proof contracts.

See `gym/G21_CONTRACT.md` and
`gym/contracts/host_wire_protocol.schema.json`.

## Reference machine, not restriction

The included environments are a reproducible standard-library reference. They
are not a ban on other runtimes, models, planners, RL implementations, host
transports, or domain adapters. Compatible upgrades preserve the task-view,
observation, typed-decision, host-boundary, reset, evidence-admission, budget,
trajectory, and proof contracts.

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
Budget exhaustion cannot receive semantic acceptance or learning credit.

G2 adds typed terminal reasons:

    TOOL_COST_BUDGET_EXHAUSTED
    STEP_BUDGET_EXHAUSTED
    GENERATIVE_BUDGET_EXHAUSTED

A retry identity is bound to:

    (source_lineage_root, check_id, probe_id)

Reusing one check_id across a different source/probe identity is rejected as
provenance aliasing. Independent probes remain independent evidence.

## Admission dimensions

Admission is now the conjunction of four distinct facts:

    semantic_valid
    provenance_valid
    trajectory_valid
    integrity_valid

Do not collapse them into one score.

## Reference score semantics

gym.run_reference is a contract-conformance smoke controller, not the target
agent. Its output explicitly sets evaluation_scope to
reference_contract_conformance_only and native_competence_claim to false.

The reference output now also reports terminal_reason and ledger_root_sha256 so
an episode can be audited without treating the root digest as a capability
score.

Do not report reference-controller pass rate as agent capability.

## Run locally

    PYTHONPATH=src:. python -m gym.validate_baseline
    PYTHONPATH=src:. python -m unittest discover -s tests -v
    PYTHONPATH=src:. python -m gym.run_reference
    PYTHONPATH=src:. python -m gym.run_subprocess_reference

The last command runs the same reference fixtures through a separate host
process and the G2.1 JSONL protocol. It remains reference contract conformance,
not native competence evidence.

## Upgrade rule

Future gym upgrades should preserve gym/BASELINE_CONTRACT.md unless a later
checkpoint explicitly replaces an invariant with a stricter contract. G2 and
G2.1 are strict supersets; the G1 baseline file is intentionally unchanged.
