# G4 Contract — Actual Adaptive Policy Training

G4 introduces the first mutable decision policy in the gym.

G1-G3 built proof, isolation, hidden-host, holdout, attestation, curriculum, and
promotion boundaries. G4 keeps those boundaries intact and replaces the
MemoryLearningSink-only demonstration with a small policy whose state changes
from admitted training trajectories.

## Scope

The reference learner is ProofGatedAdaptivePolicy.

It is deliberately small and deterministic:

- fixed semantic state encoder;
- fixed local-action <-> canonical-intent adapter;
- learned state/intent Q values and visit counts;
- conservative unknown-state fallback;
- SHA-256 policy-state commitment;
- one learning receipt per admitted train episode.

The implementation is a baseline for adaptive policy learning, not a claim that
this particular table learner is the final agent architecture.

## New invariants

134. Learning changes decision state. A successful update must change the policy
     commitment or add evidence to an already-known state/intent pair.
135. The learner may update only from split=train EpisodeResults that are fully
     admitted across semantic, provenance, trajectory, and integrity boundaries.
136. The learner independently re-verifies the trajectory ledger before applying
     credit.
137. Validation, holdout, and external_real evaluation do not receive the learner
     as a writable learning sink.
138. Evaluation must not mutate the policy commitment.
139. Policy state keys must not contain task_id, split, description text, hidden
     scenario data, snapshot id, or oracle metadata.
140. Local action names are not policy identity. G4 learns canonical intents and
     maps them back to domain-local actions.
141. The reference cross-domain authority workflow maps structured-data
     CHECK_PROVENANCE/SELECT_SOURCE_* and filesystem-config
     INSPECT_CONFIG_ORIGIN/SELECT_*_CONFIG onto the same canonical intents.
142. Unknown states fail conservatively through typed defer rather than inventing
     an unsupported action.
143. Q value is preference, not truth. Proof admission remains the authority for
     whether a trajectory may teach the learner.
144. Teacher policy output is not itself learning credit. Only the resulting
     admitted EpisodeResult/ledger may update Q state.
145. The bundled reference training uses ReferencePolicy to generate admitted
     demonstrations. Therefore G4 proves adaptive policy update and held-out
     execution, but does not yet prove autonomous on-policy exploration.
146. The bundled reference generator remains a candidate source only. G4 learns
     when to call it; generation output still requires environment validation.
147. A policy-state digest is a reproducibility commitment, not a capability
     score.
148. Reference validation/holdout improvement remains reference-gym evidence and
     cannot be promoted to native competence.
149. Failure learning is not inferred from rejected episodes in G4. The current
     core calls the learning sink only after full admission; negative-experience
     admission requires a separate later contract.
150. Future learners may replace the table implementation if they preserve the
     G1-G4 learning and authority boundaries.

## Reference experiment

The G4 reference runner performs:

1. frozen untrained-policy evaluation on G3 validation and bundled holdout;
2. proof-gated training on the nine G3 train tasks using admitted teacher
   trajectories;
3. learner-only evaluation on the five G3 validation tasks;
4. learner-only evaluation on the filesystem-config holdout;
5. policy hash check proving validation/holdout did not mutate policy state.

The filesystem holdout intentionally has different local action vocabulary. Its
success after structured-data training exercises the canonical-intent adapter,
not raw action-name memorization.

## Non-claims

G4 is not:

- native competence evidence;
- autonomous exploration;
- negative-outcome RL;
- neural learning;
- proof that the fixed semantic encoder is universally correct;
- proof that the reference generator can solve unseen synthesis tasks.

Those are later checkpoints.
