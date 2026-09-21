# G4.1 Contract — Proof-Gated On-Policy Search

G4.1 removes ReferencePolicy from the training actor path.

The adaptive policy now chooses its own typed actions during training. It
exploits admitted G4 Q state when available and otherwise searches a bounded
canonical-intent set derived from the visible semantic state and allowed action
surface.

## Core separation

```text
SearchCursor != QState
SearchFailure != NegativeReward
SearchProgress != LearningCredit
PolicyChoice != Proof
Reward != Truth
```

A failed train episode may advance the search cursor only when:

```text
provenance_valid
AND trajectory_valid
AND integrity_valid
AND semantic_valid == false
```

That failed episode does not update Q.

Q remains writable only through the existing G4 gate:

```text
split == train
AND semantic_valid
AND provenance_valid
AND trajectory_valid
AND integrity_valid
```

## Training actor

The reference actor is `BoundedExplorationPolicy`.

It first asks `ProofGatedAdaptivePolicy` for a learned decision. If the state
has no learned preference, it selects from a bounded canonical-intent search
family.

The search families are typed priors, not answer labels. Several reference
states intentionally try a semantically wrong but contract-admissible intent
first, so the training path contains real rejected episodes before later search
attempts succeed.

The bundled G4.1 reference run does not call `ReferencePolicy.decide()` during
training.

## Failure handling

Validated rejected episodes may produce a `SearchReceipt` containing:

- task id;
- ledger root;
- semantic state;
- attempted canonical intent;
- previous search cursor;
- next search cursor.

A SearchReceipt is search bookkeeping only. It cannot alter Q, mint proof,
create reward truth, or make an episode admitted.

Failures with invalid provenance, invalid trajectory integrity, or failed
Artifact-Proof integrity cannot move the search cursor.

A rejected exploit with an already-positive Q value is not silently converted
into negative Q credit in G4.1.

## Reference behavior

The deterministic reference curriculum requires:

- 9 admitted train episodes;
- 16 total training attempts;
- 7 proof/integrity-valid rejected search attempts;
- 9 G4 learning receipts;
- 11 learned state-intent pairs;
- validation improvement from 20% to 100%;
- holdout improvement from 0% to 100%;
- no policy/search mutation during validation or holdout.

These counts are regression coordinates for the bundled reference tasks, not
general capability scores.

## Scope boundary

G4.1 proves:

```text
learner-driven typed action selection
+ bounded on-policy search
+ proof-gated successful learning
+ proof-gated failed-search bookkeeping
```

G4.1 does not yet prove:

- unstructured exploration over arbitrary actions;
- learned action masks;
- negative-reward RL from failed episodes;
- native competence;
- unseen real-world autonomy.

A later checkpoint may introduce typed negative experience credit or learn the
exploration prior, but it must keep failure evidence separate from truth and
proof authority.
