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

## G2.2 sealed private holdout packs

G2.2 adds a hash-gated host-side package for private holdout descriptors,
snapshots, hidden oracle state, and a runtime identity descriptor.

The reference pack lives at:

    gym/private_holdout_reference/

and is deliberately inspectable. It exists only to regression-test pack
validation and execution binding; it is not native competence evidence.

A pack is admitted only when:

- every declared task, snapshot, and runtime descriptor matches its SHA-256;
- all paths are canonical, contained, regular files with no symlink traversal;
- complete file coverage holds, so undeclared ambient files are rejected;
- every task split is holdout or external_real;
- snapshot id/domain agree with the HostTaskDescriptor;
- the runtime descriptor id matches the host's expected runtime contract.

After validation the snapshot is injected into host-only Environment state. For
the reference filesystem-config holdout, both observable config state and the
hidden semantic oracle are read from that verified snapshot. The policy still
receives only AgentTaskView plus observations.

The sealed HostTaskDescriptor receives only host-side digest metadata, so the
existing G2 task commitment and trajectory ledger bind execution to the verified
task/snapshot/runtime identities without exposing the raw pack to the policy.

Reference command:

    PYTHONPATH=src:. python -m gym.run_private_pack_reference
    PYTHONPATH=src:. python -m gym.run_isolated_private_pack_reference
    PYTHONPATH=src:. python -m gym.agent_socket_server --host 127.0.0.1 --port 8765
    PYTHONPATH=src:. python -m gym.run_external_agent_reference --agent-host 127.0.0.1 --agent-port 8765
    PYTHONPATH=src:. python -m gym.run_attested_external_reference --agent-host 127.0.0.1 --agent-port 8765
    PYTHONPATH=src:. python -m gym.run_curriculum_reference

The output explicitly reports
`evaluation_scope=reference_private_pack_conformance_only` and
`native_competence_claim=false`.

The runtime descriptor in G2.2 is identity metadata with
`attestation_scope=reference_identity_only`; it is not executable/binary,
container, VM, or remote-host attestation.

See `gym/G22_CONTRACT.md` and
`gym/contracts/private_holdout_pack.schema.json`.

## G2.3 agent-process isolation

G2.3 adds a second sanitized process boundary for the evaluated policy/generator.

The trusted controller keeps the sealed HostTaskDescriptor, private holdout pack,
HostGateway, evidence ledger, semantic/provenance validation, Artifact-Proof
authority, and learning gate. The agent endpoint receives only AgentTaskView,
visible observations, agent-visible trajectory, and generation attempt.

Reference topology:

    trusted controller
      |-- HostGateway -----> private host process
      |
      +-- AgentEndpoint ---> evaluated agent process

The reference agent transport is strict JSONL with protocol versioning,
request_id binding, monotonic session turns, size limits, duplicate-key and
non-finite-number rejection, timeout-without-retry, and stderr
drain-and-discard.

The trusted controller also creates an agent-boundary transcript. It records
only AgentTaskView commitment, visible observation-frontier digests, typed
decision/action, trajectory digest, generation attempt, candidate SHA-256, and
explicit close. It never stores policy rationale/chain-of-thought or raw
generated candidate text.

On the isolated-agent path, ReferenceGym is deliberately called with no learning
sink. Train learning may be forwarded only after the agent transcript is
structurally valid and cross-bound to the G2 trajectory ledger.

Reference command:

    PYTHONPATH=src:. python -m gym.run_isolated_private_pack_reference

This command launches both host and reference agent as separate subprocesses.
It still reports:

    evaluation_scope=reference_agent_process_isolation_conformance_only
    native_competence_claim=false
    filesystem_authority_isolated=false

That limitation is intentional: two subprocesses with access to the same
repository/filesystem do not prove private authority isolation. Native/private
evaluation must place the evaluated AgentEndpoint in a sandbox, container, VM,
remote service, or equivalent authority boundary that cannot read the sealed
pack/evaluator/oracle except through the AgentEndpoint protocol.

See `gym/G23_CONTRACT.md` and
`gym/contracts/agent_wire_protocol.schema.json`.

## G2.4 external agent harness

G2.4 removes evaluated-agent process launch from the trusted controller path.

The controller now supports `ExternalAgentEndpoint(host, port)`, which connects
to a **pre-existing** agent service over the same sanitized G2.3 logical
protocol. The client has no agent command, working directory, environment
variables, executable, repository path, or process-launch surface.

Reference topology:

    TRUSTED CONTROLLER
      |-- HostGateway ----------> private host
      |
      +-- ExternalAgentEndpoint ---- TCP ----> PRE-EXISTING AGENT SERVICE

The controller may still launch trusted-side private-host infrastructure. It does
not launch, stop, or own the external evaluated agent service. Disconnect closes
only the controller socket and never sends SHUTDOWN.

A reference TCP service is provided for transport regression:

    PYTHONPATH=src:. python -m gym.agent_socket_server --host 127.0.0.1 --port 8765

In another process, the trusted-side reference runner can connect to it:

    PYTHONPATH=src:. python -m gym.run_external_agent_reference \
      --agent-host 127.0.0.1 --agent-port 8765

The runner reports:

    evaluation_scope=external_agent_launch_decoupling_conformance_only
    controller_launched_agent=false
    agent_launch_mode=external_preexisting_endpoint
    authority_attestation=not_present
    filesystem_authority_isolated=unverified
    kernel_authority_isolated=unverified
    credential_authority_isolated=unverified
    native_competence_claim=false

These labels are deliberate. A TCP address proves neither filesystem/kernel
isolation nor model/machine identity. The reference TCP transport also provides
no peer authentication or confidentiality, so it is not suitable across an
untrusted network for native/private evaluation.

What G2.4 establishes is narrower and testable: the trusted controller no longer
chooses or spawns the evaluated agent process, while G2.3 request-id, turn,
transcript, ledger cross-binding, and learning gates remain intact.

See `gym/G24_CONTRACT.md`.

## G2.5 authority attestation boundary

G2.5 adds a separate authority-evidence path. Authority truth is not accepted
from the evaluated agent, reward, semantic success, candidate output, or policy
rationale.

Before an attested external episode, the trusted controller creates a fresh
256-bit challenge and binds it to:

- the actual connected TCP peer locator;
- exact HostTaskDescriptor commitment;
- exact AgentTaskView commitment;
- an evaluation-context commitment such as the sealed private-pack commitment.

That request goes to an `AuthorityAttestor`, and a trusted-side
`AuthorityVerifier` must authenticate the resulting receipt before
`AgentEndpoint.start()` is allowed.

The receipt has separate claims for:

    filesystem_isolated
    kernel_isolated
    credential_isolated
    network_isolated
    machine_identity_verified
    model_identity_verified

One claim never implies another.

After the episode, the authority request is cross-bound again to the G2
`EPISODE_START` ledger entry. Learning is still withheld until authority,
evidence admission, G2.3 agent transcript, and ledger binding all succeed.

The stdlib reference implementation uses HMAC only to regression-test receipt
authenticity, tamper detection, fresh challenge binding, and replay rejection:

    reference_hmac_conformance_only

It is explicitly **not** an independent hardware/cloud/VM attestation scheme and
cannot promote a native competence or independent-authority claim.

Reference mechanics can be exercised against the same pre-existing external
agent service:

    PYTHONPATH=src:. python -m gym.run_attested_external_reference \
      --agent-host 127.0.0.1 --agent-port 8765

The output reports:

    evaluation_scope=reference_authority_attestation_conformance_only
    native_competence_claim=false
    authority_independence_verified=false
    authority_cryptography=shared_hmac_reference_only

For production, replace the reference attestor/verifier with an independently
trusted deployment-specific issuer/verifier. See `gym/G25_CONTRACT.md` and
`gym/contracts/authority_attestation.schema.json`.

## G3 proof-gated training curriculum

G3 shifts the gym from boundary construction toward **training pressure**.

The bundled reference curriculum expands the reference training set from
3 train / 2 validation tasks to:

    9 train tasks
    5 validation tasks
    5 promotion stages

The additional tasks are not simple text duplicates. Host-side hidden scenario
state now varies across episodes.

Examples:

- a CLI process can genuinely be `running` or `stopped`;
- a stopped process requires inspect -> restart -> health recheck;
- structured-data source B can be the signed authority;
- validation uses different observation patterns and numeric regimes;
- safe defer appears in train rather than only validation;
- generative retry appears in train with reject-before-accept;
- repository repair is repeated under a tighter step/tool-cost budget.

Hidden scenario fields stay inside the HostTaskDescriptor. They are not copied
into AgentTaskView.

The reference curriculum is:

    gym/curriculum/g3_reference.json

Stages:

    0 ambiguity_foundation
    1 evidence_gated_recovery
    2 mutation_and_regression
    3 bounded_generation
    4 provenance_flip_generalization

Every stage has separate thresholds for:

    train admission
    validation admission
    budget exhaustion
    train learning updates

Validation is always read-only. Any validation EpisodeResult carrying
`learning_updated=true` invalidates promotion evidence.

Promotion is checkpointed as a SHA-256 chain. A stage receipt commits to the
previous checkpoint plus exact task IDs and observed metrics. A failed stage
stops the chain; later stages cannot be promoted on top of it.

Reference command:

    PYTHONPATH=src:. python -m gym.run_curriculum_reference

The runner reports
`evaluation_scope=reference_curriculum_conformance_only` and
`native_competence_claim=false`. A reference-controller promotion chain is
training-infrastructure evidence, not native agent capability.

See `gym/G3_CONTRACT.md` and
`gym/contracts/curriculum.schema.json`.

## Run locally

    PYTHONPATH=src:. python -m gym.validate_baseline
    PYTHONPATH=src:. python -m unittest discover -s tests -v
    PYTHONPATH=src:. python -m gym.run_reference
    PYTHONPATH=src:. python -m gym.run_subprocess_reference
    PYTHONPATH=src:. python -m gym.run_private_pack_reference

The subprocess command exercises the G2.1 JSONL host boundary. The private-pack
command exercises G2.2 sealed-pack verification plus that subprocess boundary.
Both remain reference contract conformance, not native competence evidence.

## Upgrade rule

Future gym upgrades should preserve gym/BASELINE_CONTRACT.md unless a later
checkpoint explicitly replaces an invariant with a stricter contract. G2,
G2.1, G2.2, G2.3, G2.4, and G2.5 are strict supersets; the G1 baseline file is
intentionally unchanged.


## G4 actual adaptive policy training

G4 introduces the first mutable policy state in the reference gym.

ProofGatedAdaptivePolicy learns Q values over canonical semantic intents from
fully admitted train EpisodeResults. It independently re-verifies each trajectory
ledger before credit is applied.

Training uses ReferencePolicy only as a demonstration source. The learner itself
is used for post-training validation and holdout execution.

The reference experiment measures:

    untrained validation/holdout
      -> admitted train demonstrations
      -> Q/state update
      -> learner-only validation
      -> learner-only cross-domain holdout
      -> policy hash unchanged during evaluation

Local action vocabulary is not the learned policy identity. In particular,
structured-data CHECK_PROVENANCE / SELECT_SOURCE_* and filesystem-config
INSPECT_CONFIG_ORIGIN / SELECT_*_CONFIG share canonical authority intents.

Reference command:

    PYTHONPATH=src:. python -m gym.run_adaptive_training_reference

The runner reports:

    evaluation_scope=reference_adaptive_training_conformance_only
    native_competence_claim=false
    training_mode=proof_gated_teacher_demonstration_q
    on_policy_exploration_claim=false
    negative_experience_learning_claim=false

This is real mutable policy learning, but it is not yet autonomous exploration or
negative-outcome RL. See gym/G4_CONTRACT.md.


## G4.1 proof-gated on-policy search

G4.1 removes ReferencePolicy from the training actor path.

BoundedExplorationPolicy exploits learned G4 Q state when available and otherwise
searches a typed canonical-intent family. Failed episodes may move only the
separate search cursor, and only when provenance, trajectory, and Artifact-Proof
integrity validate. Failed episodes never receive Q learning credit in G4.1.

Reference behavior:

    9 train tasks
    16 total training attempts
    7 validated rejected search attempts
    9 admitted Q-learning updates
    11 learned state-intent pairs
    validation 20% -> 100%
    holdout 0% -> 100%

Reference command:

    PYTHONPATH=src:. python -m gym.run_on_policy_training_reference

Binary role:

    proof-gym-runtime on-policy-train

The runner reports
`evaluation_scope=reference_on_policy_search_conformance_only` and
`reference_policy_used_for_training=false`.

The bounded candidate families are a hand-authored typed exploration prior.
G4.1 therefore establishes teacher-free bounded on-policy search, not arbitrary
unstructured exploration or negative-reward RL.

See `gym/G41_CONTRACT.md`.
