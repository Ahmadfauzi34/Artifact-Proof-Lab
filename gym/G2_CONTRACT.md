# G2 Contract — Private Host Boundary and Proof-Gated Trajectory

G2 is a strict superset of `gym/BASELINE_CONTRACT.md`. The G1 baseline remains
the compatibility boundary and is not replaced by this file.

## Additional invariants

15. PublicTask is host-side state. Policies and generators receive only
    AgentTaskView plus observations and their own trajectory.
16. Environment objects stay behind HostGateway. A private/native benchmark may
    implement HostGateway in another process or machine without changing agent
    code.
17. A local in-repo host is reference/regression infrastructure only. Its pass
    rate is never native-competence evidence.
18. Every episode has an append-only trajectory ledger. The chain binds:
    episode start -> observation frontier -> decision -> action result -> terminal
    receipt.
19. A decision is committed against the exact observation frontier visible when
    it was made.
20. An action result carries the hash of the decision entry that caused it.
    Generated candidate text is represented in the ledger by digest, not promoted
    to proof by generation.
21. Terminal admission requires four distinct conditions:
    semantic success, provenance validity, trajectory-ledger validity, and
    Artifact-Proof integrity closure.
22. Trajectory-ledger validity is not semantic correctness. The ledger proves
    ordering and identity of the recorded transition history, not that the chosen
    action was good.
23. Budget termination is typed. Tool-cost, step, and generative-call exhaustion
    remain distinguishable terminal reasons.
24. Private holdout/native evaluation should keep host implementation, hidden
    evaluator, oracle/reference state, and snapshots outside the agent workspace.
    Only the sanitized host protocol crosses that boundary.

## Reference topology

    PRIVATE / HOST SIDE
      HostTaskDescriptor (PublicTask)
        -> HostGateway
        -> hidden Environment / evaluator / snapshots
        -> observations
              |
              | sanitized surface only
              v
    AGENT SIDE
      AgentTaskView + observations
        -> policy / generator
        -> typed decision or candidate
              |
              v
    HOST SIDE
      execute transition
        -> evidence
        -> trajectory ledger
        -> semantic evaluator
        -> provenance validation
        -> Artifact-Proof integrity bundle
        -> admission

## Ledger statement

A valid ledger can establish:

- which observation frontier preceded a decision;
- which decision caused a recorded action result;
- that the recorded event chain was not reordered or rewritten without changing
  the ledger root;
- which terminal reason closed the episode.

It does not establish semantic correctness by itself.

## Private-host implementation rule

`LocalReferenceHost` is intentionally inspectable and exists to prove the
protocol. Native-competence evaluation should supply another `HostGateway`
implementation from outside the agent workspace. No platform, transport, or
model family is mandated; the interface is the reference coordinate.
