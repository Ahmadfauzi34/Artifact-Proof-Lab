# G4.2 Contract — Typed Negative Experience

G4.2 introduces persistent negative experience without turning failure into
truth, proof, or ordinary Q credit.

## Core separation

```text
NegativeExperience != Truth
NegativeExperience != Proof
NegativeExperience != Q
SemanticRejection != GlobalFalse
FailureReceipt != RewardAuthority
```

Positive G4 Q is unchanged by a G4.2 negative update.

## Admission for negative experience

A rejected episode may create a NegativeExperienceReceipt only when all of the
following hold:

```text
split == train
accepted == false
semantic_valid == false
provenance_valid == true
trajectory_valid == true
integrity_valid == true
budget_exhausted == false
terminal_reason == ENVIRONMENT_DONE
trajectory ledger re-verifies
```

Budget exhaustion, invalid policy actions, unsupported actions, generator
unavailability, provenance failure, ledger failure, or Artifact-Proof failure
must not create negative preference.

## Causal scope

Only the decision bound to the terminal semantic rejection receives negative
experience.

Earlier correct steps inside a failed episode are not penalized.

Example:

```text
AUTHORITY_UNKNOWN
  -> INSPECT_AUTHORITY       # not negative
AUTHORITY_SLOT_1
  -> SELECT_AUTHORITY_SLOT_2 # terminal rejection; negative
```

## Decision use

NegativeAwareExplorationPolicy first consults admitted positive Q.

If positive Q has no unique supported preference, unresolved bounded candidates
are ranked by their typed rejection count. Lower rejection count is preferred;
the original typed prior order breaks ties deterministically.

In G4.2, negative memory does not erase or override an already admitted positive
Q preference. Positive-vs-negative conflict arbitration is deliberately left to
a later checkpoint.

## Persistent state

The decision-bearing negative state contains only:

```text
semantic_state
canonical_intent
rejection_count
```

It does not contain task id, split, description, snapshot id, oracle data, or
raw host metadata.

Receipts retain task/ledger identity for provenance auditing but are not the
decision-state key.

## Cross-domain transfer

Because negative memory uses canonical semantic state and canonical intent, it
can transfer across local action vocabularies.

The reference regression proves that a validated rejection learned from:

```text
structured_data
AUTHORITY_SLOT_1 -> SELECT_AUTHORITY_SLOT_2
```

can steer a fresh Q-empty policy on:

```text
filesystem_config
AUTHORITY_SLOT_1 -> SELECT_PROJECT_CONFIG
```

without importing positive Q state.

## Reference profile

The bundled deterministic run produces:

- 9 admitted positive train episodes;
- 7 typed negative receipts;
- 16 total attempts;
- 11 positive learned state-intent pairs;
- validation 20% -> 100%;
- holdout 0% -> 100%;
- no Q or negative-store mutation during validation/holdout.

These are reference regression coordinates, not capability scores.

## Non-claims

G4.2 does not claim:

- failure is globally wrong forever;
- negative reward RL;
- positive-vs-negative conflict arbitration;
- stochastic exploration;
- learned exploration priors;
- native competence.

A later checkpoint may introduce evidence-weighted conflict arbitration while
preserving the separation between preference, truth, and proof.
