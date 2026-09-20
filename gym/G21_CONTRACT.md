# G2.1 Contract — External Private-Host Reference Harness

G2.1 is a strict superset of `gym/BASELINE_CONTRACT.md` and
`gym/G2_CONTRACT.md`. It adds a reference transport for moving HostGateway
execution into another process. It does not make subprocess/stdio the only valid
transport.

## Additional invariants

26. Process separation is not automatically privacy. A native/private benchmark
    must also prevent the evaluated agent from reading host files, process memory,
    task registries, evaluator code, or transport internals.
27. The reference wire protocol is versioned and fail-closed. Unknown protocol
    versions, malformed JSON, non-finite JSON values, oversized frames, missing
    request IDs, request/response ID mismatch, and unexpected wire fields are
    rejected.
28. Every request has exactly one request_id-bound response. A response from an
    earlier or unrelated request cannot satisfy the current transition.
29. The reference host registry is host-owned. START identifies a task by opaque
    task_id plus descriptor commitment; the server does not return HostTaskDescriptor,
    split, skill labels, snapshot ID, runner identity, metadata, evaluator details,
    or oracle/reference state.
30. START returns only the sanitized AgentTaskView, observations, session ID, and
    host task commitment required to bind the controller to the intended host
    descriptor.
31. Host errors are typed. Contract-level action rejection may become
    HostActionRejected; protocol/internal/transport failures remain hard failures
    and are not relabeled as policy mistakes.
32. Candidate text may cross the host boundary only when the task exposes
    CALL_GENERATIVE. It remains candidate material and is not proof by transport.
33. Private policy reasoning is not a wire requirement. The reference wire carries
    typed action/candidate data, never requires chain-of-thought or rationale.
34. A request timeout is a transport failure. The client terminates the reference
    subprocess rather than silently retrying a state-changing command.
35. The reference subprocess host is reusable across episodes, but each host
    session has an explicit CLOSE and the transport has an explicit SHUTDOWN.
36. The in-repo subprocess server and bundled task registry remain
    reference/regression infrastructure only. Their success is not native
    competence evidence.
37. Subprocess stderr is not a semantic/proof channel. Host internal diagnostics
    are redacted from client exceptions so hidden evaluator/oracle details cannot
    become an accidental side-channel.

## Reference topology

    controller / gym process
      HostTaskDescriptor
      ReferenceGym
      policy adapter
            |
            | HostGateway calls
            v
      SubprocessHostGateway
            |
            | strict JSONL v1
            | request_id bound
            v
    separate private-host process
      task registry
      environment factory
      hidden evaluator/snapshot
            |
            v
      LocalReferenceHost-compatible semantics

For a real private holdout, move the private-host process, task registry,
environment implementation, evaluator, and snapshots outside the evaluated
agent's readable workspace and authority.

## Wire exposure rule

The reference START response may expose:

    session_id
    AgentTaskView
    observations
    host_task_commitment

It must not expose:

    HostTaskDescriptor
    split
    skill_targets
    snapshot_id
    runner
    metadata
    hidden validator
    oracle/reference answer

## Reference machine, not transport restriction

JSONL over subprocess stdio is the reproducible G2.1 reference transport. A
socket, RPC framework, container bridge, VM channel, or remote service is
compatible when it preserves the same logical HostGateway, sanitization,
identity, timeout/failure, and proof boundaries.
