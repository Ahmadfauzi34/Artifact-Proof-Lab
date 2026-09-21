# G4.3 Contract — External-Agent Proof-Gated Training Receipts

G4.3 connects the G2.4 external AgentEndpoint boundary to the G4/G4.2 learning
admission rules without giving the trusted gym authority to mutate the external
agent's private state.

## Core separation

```text
Gym != AgentBrain
TrainingReceipt != LearningUpdate
Reward != Truth
PolicyChoice != Proof
PositiveReceipt requires full admission
TypedNegativeReceipt != Q
```

The trusted controller executes a train episode against a pre-existing external
AgentEndpoint. It verifies the same semantic, provenance, trajectory, integrity,
and agent-transcript boundaries already used by the gym, then emits a portable
receipt.

The gym never writes the external agent's model, Q table, memory database, or
other private state.

## Positive receipt gate

A receipt may be marked `experience_kind=positive` only when:

```text
split == train
accepted == true
semantic_valid == true
provenance_valid == true
trajectory_valid == true
integrity_valid == true
agent transcript valid and cross-bound to the episode
```

## Typed-negative receipt gate

A receipt may be marked `experience_kind=typed_negative` only when:

```text
split == train
accepted == false
semantic_valid == false
provenance_valid == true
trajectory_valid == true
integrity_valid == true
budget_exhausted == false
terminal_reason == ENVIRONMENT_DONE
agent transcript valid and cross-bound to the episode
```

This mirrors the G4.2 negative-experience admission boundary. It does not turn
semantic rejection into truth, proof, or ordinary positive Q credit.

All other train outcomes produce `experience_kind=none` with
`learning_admissible=false`.

## Receipt identity

Each receipt binds:

- task id, split and domain;
- admission dimensions;
- terminal reason and budget state;
- trajectory ledger root;
- agent-boundary transcript root;
- sanitized AgentTaskView commitment;
- separated reward channels;
- agent-visible action trajectory;
- sanitized decision/generation transcript entries;
- a SHA-256 commitment over the entire receipt body.

Generated candidate text is not copied into the receipt. The existing transcript
contains only its SHA-256 commitment.

## External consumer rule

`external-agent-train` is a training **host** role, not an agent-side learner.

```text
proof-gym-runtime external-agent-train
       -> actual train episode
       -> proof/admission
       -> portable training receipt
       -> external consumer admission
       -> external agent state update
```

The last state update is outside gym authority. An Assistant Dev consumer may
apply a valid receipt only through its own proof-gated learning boundary.

This allows iterative online training in environments where the gym binary and
agent state are transferred separately:

```text
run episode
-> verify receipt
-> update/freeze agent DB
-> restart or continue external agent
-> run next episode
```

## Scope and non-claims

G4.3 establishes a transportable, proof-gated learning-evidence boundary for an
external agent. It does not claim:

- that the bundled reference tasks are native-competence evidence;
- that the external endpoint has independent filesystem/kernel authority;
- that a receipt automatically changes an external agent;
- that reward is truth;
- that semantic rejection is globally false forever.

G2.4/G2.5 authority limitations still apply.
