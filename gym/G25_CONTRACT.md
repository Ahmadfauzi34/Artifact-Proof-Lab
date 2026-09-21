# G2.5 Contract — Authority Attestation Boundary

G2.5 is a strict superset of `gym/BASELINE_CONTRACT.md`,
`gym/G2_CONTRACT.md`, `gym/G21_CONTRACT.md`,
`gym/G22_CONTRACT.md`, `gym/G23_CONTRACT.md`, and
`gym/G24_CONTRACT.md`.

G2.4 proved controller-side launch decoupling but deliberately left filesystem,
kernel, credential, network, model, and machine authority claims unverified.

G2.5 introduces an authority-evidence boundary. An authority claim may be
treated as verified only when a trusted-side verifier accepts a receipt from an
attestor interface that is separate from the evaluated agent protocol.

## Additional invariants

94. Authority evidence is not accepted from AgentEndpoint messages,
    PolicyDecision rationale, generated candidates, reward, semantic success, or
    trajectory outcome. It enters through a distinct AuthorityAttestor boundary.
95. Every attestation request contains a fresh 256-bit controller challenge.
96. An attestation request binds the actual connected TCP peer locator, exact
    HostTaskDescriptor commitment, exact AgentTaskView commitment, and an
    evaluation-context commitment such as a sealed private-pack commitment.
97. Receipt fields are exact and typed. Digests and challenges use 64 lowercase
    hexadecimal characters.
98. Authority claims are separate booleans for filesystem, kernel, credential,
    network, machine identity, and model identity. One verified claim does not
    imply another.
99. A receipt must be authenticated by the configured trusted-side verifier.
    Claim text without successful authenticity verification is not authority
    evidence.
100. A verified challenge is one-time. Replaying the same receipt/challenge to
     the same verifier fails closed.
101. Receipt tampering, issuer mismatch, assurance mismatch, challenge mismatch,
     endpoint mismatch, HostTaskDescriptor mismatch, AgentTaskView mismatch, or
     evaluation-context mismatch fails closed.
102. If authority verification fails, the external endpoint is disconnected and
     no episode is started on that endpoint.
103. G2.5 still calls the G2.3 isolated episode path with no learning sink.
     Learning may be forwarded only after authority verification, ordinary
     evidence admission, agent-transcript verification, and authority-to-ledger
     cross-binding all succeed.
104. The authority request HostTaskDescriptor and AgentTaskView commitments must
     match the G2 EPISODE_START ledger entry exactly before learning can occur.
105. Authority receipt identity is recorded by SHA-256 for audit correlation.
     Raw attestation secrets/keys are not ledger or agent inputs.
106. The reference HMAC attestor/verifier is stdlib-only conformance
     infrastructure. It tests authenticity, tamper detection, binding, and
     replay mechanics but is not hardware/cloud/VM attestation.
107. A repository-bundled or controller-generated reference HMAC receipt cannot
     promote `native_competence_claim` or an independent-authority claim.
108. Production authority assurance requires replacing the reference attestor
     and verifier with an independent trusted issuer/verifier appropriate to the
     deployment (for example a cloud, VM, container, hardware, or organization
     attestation system).
109. G2.5 does not collapse authority proof into competence proof. Verified
     isolation says where/how an agent ran; competence still requires accepted
     private/external evaluation evidence.
110. Failure of authority verification or authority/ledger cross-binding cannot
     produce a learning update.

## Binding chain

    fresh controller challenge
             |
             v
    AuthorityAttestationRequest
      endpoint locator digest
      HostTaskDescriptor digest
      AgentTaskView digest
      evaluation-context digest
             |
             v
      independent attestor
             |
             v
    authenticated receipt
             |
             v
      trusted verifier
             |
             +---- invalid ---> STOP / disconnect / no episode
             |
             v
      run isolated episode
             |
             v
    EPISODE_START ledger
      host-task commitment -------- must match receipt request
      agent-view commitment ------- must match receipt request
             |
             v
    evidence + transcript + proof
             |
             v
    learning update only if every gate passes

## Reference HMAC limitation

`ReferenceHMACAuthorityAttestor` and
`ReferenceHMACAuthorityVerifier` deliberately use a shared HMAC key so the
full contract can be regression tested using Python standard library only.

Its assurance label is:

    reference_hmac_conformance_only

This is not an asymmetric independent attestation scheme and must not be used to
claim native/private authority isolation by itself.
