# G2.2 Contract — Sealed Private Holdout Pack

G2.2 is a strict superset of `gym/BASELINE_CONTRACT.md`,
`gym/G2_CONTRACT.md`, and `gym/G21_CONTRACT.md`.

It adds a host-side, hash-gated package for private holdout task descriptors,
snapshots, hidden oracle state, and a runtime identity descriptor.

## Additional invariants

38. A private holdout pack is host-side material. The evaluated policy receives
    only AgentTaskView and observations; task metadata, raw snapshot state,
    hidden oracle fields, pack manifest, and runtime descriptor are not policy
    inputs.
39. Every task file, snapshot file, and runtime descriptor is SHA-256 gated by
    PRIVATE_HOLDOUT_PACK.json before parsing or execution.
40. Pack paths are canonical relative POSIX paths. Absolute paths, traversal,
    symlinks, missing files, and non-regular files fail closed.
41. Pack coverage is complete. Any unlisted file or undeclared symlink rejects
    the pack rather than becoming ambient hidden state.
42. Private packs may contain only holdout or external_real tasks. A train or
    validation task in a private pack is a contract violation.
43. Snapshot identity in the task descriptor must match the manifest snapshot
    identity, and the snapshot domain must match the task domain.
44. A verified snapshot is injected only into host-side Environment state. Its
    hidden oracle is never copied into AgentTaskView or observations merely
    because it exists.
45. Environment semantics must consume the verified private snapshot when a
    private snapshot is present. A manifest hash that is never used by execution
    is insufficient evidence of snapshot binding.
46. The sealed HostTaskDescriptor is augmented with host-only metadata containing
    the pack id and verified resource digests. Therefore the existing G2 host-task
    commitment and trajectory ledger bind the episode to those verified resources
    without exposing them to the policy surface.
47. The pack commitment is distinct from a competence claim. It proves identity
    of the validated package inputs, not that the agent solved them independently.
48. The in-repo private_holdout_reference pack is intentionally inspectable and
    remains reference/regression evidence only. It must report
    native_competence_claim=false.
49. The G2.2 runtime descriptor has attestation_scope=reference_identity_only.
    It binds a declared runtime contract identity; it is not a binary, container,
    VM, or remote-machine attestation.
50. Native/private competence evidence requires the pack, snapshots, evaluator,
    and oracle to exist outside the evaluated agent's readable workspace and
    authority. Process separation alone is insufficient.
51. Learning updates remain impossible for private-pack tasks because their split
    is restricted to holdout/external_real and the G1 learning gate remains
    unchanged.
52. Tampering with a task, snapshot, runtime descriptor, manifest path, or file
    coverage fails before an episode starts.
53. Pack JSON and host-wire JSON reject duplicate object keys and non-finite
    numeric values, including finite-token overflow such as 1e999. Hashed bytes
    are not allowed to rely on parser-specific last-key-wins behavior.
54. A private pack or task descriptor cannot self-assert native competence.
    native_competence_claim must remain false (or absent on a task). Capability
    evidence must be issued by the evaluation/admission path, not by tested input.

## Reference pack topology

    PRIVATE_HOLDOUT_PACK.json
      |
      +-- runtime/reference-runtime.json  -- SHA-256 gated
      |
      +-- tasks/<task>.json               -- SHA-256 gated
      |      |
      |      +-- HostTaskDescriptor
      |
      +-- snapshots/<snapshot>.json       -- SHA-256 gated
             |
             +-- state
             +-- hidden oracle

After verification:

    task + snapshot + verified digests
             |
             v
      sealed HostTaskDescriptor
             |
             +-- AgentTaskView ----------> policy
             |
             +-- private_snapshot -------> host Environment
             |
             +-- host metadata ----------> G2 task commitment / ledger

## Pack commitment

PrivateHoldoutPack computes an opaque SHA-256 commitment over:

- pack id;
- exact manifest SHA-256;
- runtime id and runtime-descriptor SHA-256;
- every sealed task descriptor SHA-256.

This commitment is useful for host-side audit correlation. G2.2 does not expose
the raw pack or hidden oracle to the policy.

## Reference limitation

The repository-bundled pack exists so the pack parser, binding rules, and host
execution can be regression tested. Since an evaluated agent with repository
access could inspect it, passing this pack is not native holdout evidence.
