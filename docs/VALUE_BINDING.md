# Value Binding Reference Contract

`value_binding` is a schema-neutral semantic consistency primitive for
Artifact-Proof manifests.

It exists for cases where individually hash-correct files can still describe
incompatible state. A file SHA-256 proves byte identity; it does not prove that
a digest, checkpoint, revision, receipt identity, or recovery identity copied
into another declared document still names those same bytes.

## Proof statement

For a declared check with operands `v[0]..v[n-1]`, where `n >= 2`:

```text
PASS iff canonical_json(v[0]) == ... == canonical_json(v[n-1])
```

Supported operand kinds are:

```text
json_pointer(path, pointer)
file_sha256(path)
```

A `json_pointer` operand resolves a value from a declared JSON file using JSON
Pointer syntax. An empty pointer selects the complete JSON document.

A `file_sha256` operand resolves to the lowercase SHA-256 string of the current
bounded artifact bytes for the declared file.

The check does not know application field names. A manifest may therefore bind
backup identity, checkpoint identity, artifact revision, receipt digests, or
other state without changing the verifier implementation.

## Boundary

```text
file hash correctness
        !=
cross-document semantic coherence

value equality
        !=
publisher identity
        !=
authority attestation
        !=
semantic truth of the asserted value
```

For example, three documents can consistently contain the same false claim.
`value_binding` proves their declared values agree; it does not prove the claim
is externally true unless one operand is itself bound to independent evidence
such as a physical file digest that the claim is supposed to identify.

A detached Artifact-Proof manifest anchor remains the publisher/trust-anchor
boundary. Gym authority attestation remains a separate authority boundary.

## Fail-closed behavior

A declared binding fails when:

- fewer than two operands are declared;
- an operand kind is unsupported;
- an operand references an undeclared file;
- JSON is malformed, non-UTF-8, contains duplicate keys, non-standard numeric
  constants, or non-finite numeric overflow;
- JSON Pointer syntax is invalid;
- a selected object member or array index is absent/invalid;
- any canonical selected value differs from the first operand.

JSON files are read through the normal bounded `ArtifactSource`; no separate
value-binding byte ceiling is introduced. Existing caller-configurable source
limits remain authoritative.

## Report privacy

The finding does not copy selected raw values into the proof report. It records
operand coordinates plus SHA-256 identities of canonical selected values.
This keeps the equality result auditable without unnecessarily widening the
report data surface.

## Example

```json
{
  "id": "backup-identity-coherence",
  "type": "value_binding",
  "operands": [
    {
      "kind": "json_pointer",
      "path": "AGENT_MANIFEST.json",
      "pointer": "/state_contract/backup_zip_sha256_at_packaging"
    },
    {
      "kind": "json_pointer",
      "path": "STATE_RECOVERY_CONTRACT.json",
      "pointer": "/backup_zip_sha256_at_packaging"
    },
    {
      "kind": "file_sha256",
      "path": "state/brain-backup.zip"
    }
  ]
}
```

The names above are only an example consumer. They are not built into
Artifact-Proof.
