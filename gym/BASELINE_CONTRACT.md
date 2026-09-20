# G1 Baseline Contract — Upgrade Handoff

Future upgrades should treat this file as a compatibility boundary.

## Non-negotiable invariants

1. Skill != domain. Semantic skills must work across more than one local vocabulary/environment.
2. Reward != truth. Reward channels are advisory learning signals only.
3. Policy choice != proof. Typed policy decisions never grant acceptance authority.
4. Generated candidate != accepted candidate. Validation happens after generation.
5. Semantic success != proof closure. Keep task outcome, provenance/integrity closure, and learning admission separate.
6. Only admitted train evidence may update learning state. Validation, holdout, and external-real are read-only to the training sink.
7. Proof authority is fail-closed. Missing artifact_proof cannot become a successful integrity receipt.
8. Agent view is sanitized. Split, skill labels, host snapshot identity, runner identity, oracle metadata, and hidden validator details stay host-only.
9. Retry identity is provenance-bound. Same-check retry means the same source_lineage_root, check_id, and probe_id with monotonic attempts.
10. Resource budgets are executable contracts. max_steps, max_generative_calls, and max_tool_cost are enforced, not documented only.
11. Repo-bundled fixtures are regression/reference only. They are not evidence of native competence because their host implementation is inspectable.
12. Native competence needs external/private holdout. Mount hidden evaluator/oracle outside the agent workspace and expose only the sanitized task surface.
13. Clean fixtures are regression controls, not capability claims. No clean-fixture score substitutes for noisy, holdout, or external evaluation.
14. Reference policy is replaceable. It is a deterministic contract controller, not the target agent.

## Promotion evidence

Later checkpoints should distinguish:

    reference contract conformance
    training performance
    validation performance
    private holdout performance
    external-real performance

Do not collapse these into one score.
