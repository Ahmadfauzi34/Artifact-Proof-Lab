# G2.4 Contract — External Agent Harness / Launch Decoupling

G2.4 is a strict superset of `gym/BASELINE_CONTRACT.md`,
`gym/G2_CONTRACT.md`, `gym/G21_CONTRACT.md`,
`gym/G22_CONTRACT.md`, and `gym/G23_CONTRACT.md`.

G2.4 removes evaluated-agent process launch from the trusted controller path.
It does **not** claim that a TCP connection proves filesystem, kernel,
credential, network, model, machine, or human identity isolation.

## Additional invariants

76. ExternalAgentEndpoint connects only to a pre-existing endpoint address. Its
    API has no agent command, cwd, environment, executable, repository path, or
    process-launch parameter.
77. ExternalAgentEndpoint never invokes subprocess/Popen or any equivalent agent
    launcher. `controller_launched_agent=false` is therefore a property of the
    G2.4 client path, not a self-reported agent claim.
78. The trusted controller may still launch trusted-side private-host
    infrastructure. G2.4 launch decoupling applies specifically to the evaluated
    agent authority.
79. The external endpoint reuses the G2.3 sanitized logical AgentEndpoint
    contract: START gets AgentTaskView only; DECIDE/GENERATE receive only
    agent-visible session state.
80. G2.3 request_id, strict-JSON, typed decision, monotonic turn, transcript, and
    learning-gate invariants remain mandatory for the external endpoint path.
81. A connect, read, or write timeout is fail-closed. State-changing commands are
    never silently retried on a fresh connection.
82. Any malformed response, request_id mismatch, turn mismatch, unexpected
    field, or protocol violation poisons the external connection. The client
    cannot silently reconnect and continue the same evaluation authority.
83. ExternalAgentEndpoint disconnect closes only the controller-side socket. It
    never sends SHUTDOWN because the trusted controller does not own the external
    agent service lifecycle.
84. The reference TCP service rejects SHUTDOWN from the external controller.
85. The reference TCP service is inspectable conformance infrastructure only.
    It may be launched manually or by a test harness; its performance is not
    native competence evidence.
86. Plain reference TCP provides no confidentiality, peer authentication, or
    machine identity attestation. It must not be used across an untrusted network
    for native/private evaluation.
87. A remote TCP address is not authority evidence. Filesystem, kernel,
    credential, network, model, and machine isolation remain `unverified`
    unless an independent trusted attestation layer verifies them.
88. The G2.4 reference runner requires an already-running agent host/port and
    contains no code path that launches the agent. It reports
    `agent_launch_mode=external_preexisting_endpoint`.
89. The G2.4 reference runner must report
    `native_competence_claim=false` and
    `authority_attestation=not_present`.
90. Private holdout inputs, evaluator/oracle state, HostTaskDescriptor, and
    trusted host runtime remain absent from the external agent wire surface.
91. The existing G2.3 agent-boundary transcript remains the controller-observed
    proof of sanitized I/O ordering and cross-binding to the G2 trajectory
    ledger.
92. Train learning remains withheld from ReferenceGym until G2.3 transcript
    verification succeeds. G2.4 does not weaken the learning gate.
93. Launch decoupling is necessary but insufficient for native/private authority
    isolation. Promotion to an authority-isolated claim requires a later
    independent attestation/evidence layer; G2.4 itself cannot promote that
    claim.

## Topology

    TRUSTED CONTROLLER
      sealed private pack
      ReferenceGym / proof / learning gate
          |
          +-- HostGateway --> trusted/private host
          |
          +-- TCP AgentEndpoint --------------------+
                                                    |
                                            EXTERNAL AGENT AUTHORITY
                                            pre-existing service
                                            sanitized AgentTaskView
                                            observations
                                            typed decisions/candidates

The trusted controller knows only the endpoint address. It does not know or
choose the agent executable, working directory, environment variables, process
tree, repository mount, container image, VM image, or remote machine runtime.

## What G2.4 proves

- controller-side agent launch has been removed from this execution path;
- the same sanitized G2.3 protocol can cross a process/machine boundary;
- protocol failures remain fail-closed;
- the transcript/ledger/learning gates survive external transport.

## What G2.4 does not prove

- that the external endpoint cannot read the repository;
- that the external endpoint is in a container or VM;
- that filesystem/network/credential isolation is enforced;
- that the endpoint is a specific model or machine;
- that the TCP peer is authenticated;
- that native competence has been demonstrated.

Those claims require independent authority/identity attestation outside the
evaluated agent.
