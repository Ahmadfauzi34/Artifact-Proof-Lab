# G2.3 Contract — AgentEndpoint Process Isolation

G2.3 is a strict superset of `gym/BASELINE_CONTRACT.md`,
`gym/G2_CONTRACT.md`, `gym/G21_CONTRACT.md`, and
`gym/G22_CONTRACT.md`.

G2.3 separates policy/generator execution from the trusted gym controller. The
trusted controller retains the HostTaskDescriptor, sealed private pack,
HostGateway, evidence ledger, semantic evaluator, provenance checks, proof
authority, and learning gate.

## Additional invariants

55. On the isolated-agent path, the trusted controller does not instantiate or
    call the evaluated policy/generator implementation directly. It talks to an
    AgentEndpoint using only the sanitized logical agent surface.
56. AgentEndpoint START receives AgentTaskView only. HostTaskDescriptor fields
    such as split, skill_targets, environment, metadata, private snapshot,
    private pack manifest, evaluator/oracle state, and host runtime paths are not
    part of the endpoint task message.
57. DECIDE receives only the active agent session id, monotonic turn, visible
    observations, and the agent-visible trajectory. GENERATE receives only the
    active session id, monotonic turn, visible observations, and generation
    attempt.
58. Agent responses expose only a typed PolicyDecision or candidate text needed
    for execution. Private rationale/chain-of-thought is neither requested nor
    accepted as proof material.
59. The reference agent wire protocol is versioned, strict JSON, size bounded,
    request_id bound, and fail-closed. Duplicate keys, non-finite values,
    malformed envelopes, unexpected fields, EOF, and timeout are rejected.
60. Every DECIDE/GENERATE event has a session-local positive monotonic turn.
    Replay, skip, reordering, or a response for another turn/session is a
    protocol failure.
61. A protocol desynchronization poisons the reference subprocess channel. The
    controller does not silently continue later episodes on that channel.
62. A request timeout never retries a state-changing DECIDE or GENERATE command.
    The reference subprocess is terminated.
63. Subprocess stderr is not an evidence/reasoning channel and is
    drain-and-discard. Hidden/private diagnostics are not reflected to the
    trusted controller as policy content.
64. The trusted controller records an append-only agent-boundary transcript.
    Its chain contains the AgentTaskView commitment, visible observation
    frontiers, typed decisions, trajectory digest, generation attempt, candidate
    digest, and explicit session close.
65. Raw generated candidate text is not stored in the agent-boundary transcript;
    only SHA-256 is committed. Rationale/chain-of-thought is never stored.
66. Agent transcript decisions must cross-bind to every DECISION entry in the G2
    trajectory ledger by exact typed decision and observation-frontier digest.
    The transcript trajectory digest must also equal the ordered ACTION_RESULT
    action history that preceded that decision.
67. An AGENT_GENERATION event is valid only immediately after a
    CALL_GENERATIVE decision, on the same visible observation frontier, with
    contiguous generation attempts. Its candidate digest must match both
    EpisodeResult generated candidates and candidate digests committed by G2
    ACTION_RESULT ledger entries.
68. On the isolated-agent path ReferenceGym receives no learning sink. A train
    learning update may occur only after normal evidence admission AND successful
    agent-transcript structural/cross-binding verification.
69. Agent endpoint/transport/transcript failure before that verification cannot
    update learning state.
70. Agent session close is explicit. If an episode fails, a close failure must
    not replace the primary failure. If the episode otherwise succeeds, close
    failure remains visible and prevents post-episode learning.
71. AgentTaskView is immutable for one endpoint session. A trusted-controller
    attempt to switch the task surface mid-session fails closed.
72. LocalReferenceAgentEndpoint and the in-repo reference agent worker are
    regression/conformance infrastructure only. Their performance is not native
    competence evidence.
73. A subprocess boundary by itself is not filesystem, kernel, credential, or
    network authority isolation. The G2.3 reference runner reports
    filesystem_authority_isolated=false.
74. Native/private evaluation must run the evaluated AgentEndpoint under an
    authority boundary that cannot read trusted controller state, private packs,
    hidden evaluator/oracle material, or private host files except through the
    sanitized AgentEndpoint protocol.
75. The agent transcript proves the sanitized request/response history observed
    by the trusted controller. It does not prove a specific model binary,
    container image, machine, or human identity unless an independent
    attestation layer is added.

## Trusted topology

    TRUSTED CONTROLLER PROCESS
      sealed HostTaskDescriptor / private pack
      ReferenceGym
      evidence ledger / proof / learning gate
           |                         |
           | HostGateway             | AgentEndpoint
           v                         v
    PRIVATE HOST PROCESS       EVALUATED AGENT PROCESS
      environment/evaluator      AgentTaskView only
      hidden oracle/snapshot     visible observations
      semantic verdict           typed decision/candidate

The two protocols are intentionally independent. Host messages never need to
cross the agent boundary, and agent messages never grant the agent access to the
HostTaskDescriptor or private pack.

## Agent-boundary transcript

The trusted controller records:

    AGENT_START
      -> AGENT_DECISION / AGENT_GENERATION (monotonic turns)
      -> AGENT_CLOSE

A valid transcript establishes ordering and controller-observed binding. For an
episode to pass the G2.3 isolated path, the transcript is then cross-checked
against the G2 trajectory ledger.

## Reference limitation

The bundled `gym.agent_worker` is deliberately inspectable and the reference
runner launches it with repository Python code available. This proves protocol
and transcript behavior only. It must not be described as native holdout
isolation or native agent competence.
