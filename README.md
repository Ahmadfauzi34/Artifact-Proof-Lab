# Artifact Proof Lab

Artifact Proof Lab is a fail-closed, local-first verifier for software bundles,
agent capsules, and other artifacts whose integrity must be demonstrated rather
than guessed.

The project distinguishes four different statements:

1. a container is structurally safe to inspect;
2. its files match a declared manifest;
3. semantic checks such as SQLite integrity and cold-backup identity pass;
4. the manifest itself is anchored by an independently supplied digest.

A passing internal manifest without a detached anchor proves consistency, not
publisher identity. The report preserves that distinction as a warning instead
of silently promoting it to trust.

## Current capabilities

- verify directories or ZIP archives without extracting the outer ZIP;
- reject traversal paths, ambiguous paths, symlinks, duplicate members,
  encrypted entries, excessive expansion ratios, and configured size limits;
- enforce complete file coverage and SHA-256 identities;
- distinguish immutable files from mutable live state;
- run SQLite `quick_check` or `integrity_check` read-only;
- prove that a member of a nested cold-backup ZIP matches a target file;
- bind values selected by JSON Pointer to each other and/or to a physical
  file SHA-256 without hard-coding an application schema;
- compare the host to a binding or non-binding reference environment;
- freeze the completed finding set so a rejected proof cannot be rewritten as
  passing after validation;
- finalize a stable source snapshot identity so an earlier successful check
  cannot hide source drift later in the same verification run;
- emit stable human-readable or JSON proof reports.

## Quick start

The runtime has no third-party dependencies.

```bash
PYTHONPATH=src python -m artifact_proof verify ./my-artifact
PYTHONPATH=src python -m artifact_proof verify ./my-artifact.zip --json
PYTHONPATH=src python -m artifact_proof verify ./my-artifact.zip \
  --manifest-sha256 <detached-sha256> --require-trust-anchor
```

Exit status is `0` for a passing proof state and `1` for a rejected artifact.
CLI usage errors use argparse's normal non-zero status.

## Manifest example

```json
{
  "format": "artifact-proof-manifest-v1",
  "artifact": {"name": "assistant-dev", "version": "1"},
  "files": {
    "runtime/agent.py": {"sha256": "<64 lowercase hex>"},
    "state/brain.db": {"sha256": "<64 lowercase hex>", "mutable": true},
    "state/brain-backup.zip": {"sha256": "<64 lowercase hex>", "mutable": true},
    "AGENT_MANIFEST.json": {"sha256": "<64 lowercase hex>"},
    "STATE_RECOVERY_CONTRACT.json": {"sha256": "<64 lowercase hex>"}
  },
  "coverage": {"complete": true, "allow_unlisted": []},
  "checks": [
    {
      "id": "brain-sqlite",
      "type": "sqlite_integrity",
      "path": "state/brain.db",
      "mode": "full"
    },
    {
      "id": "cold-backup-identity",
      "type": "zip_member_matches",
      "archive": "state/brain-backup.zip",
      "member": "brain.db",
      "target": "state/brain.db",
      "only_member": true,
      "profiles": ["sealed"]
    },
    {
      "id": "backup-metadata-binding",
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
    },
    {
      "id": "reference-machine",
      "type": "environment",
      "python_implementation": "cpython",
      "python_major_minor": [3, 13],
      "platform_system": "Linux",
      "enforce": false
    }
  ]
}
```

In `sealed` profile every declared file is hash-gated. In `live` profile,
files explicitly marked `mutable` must still exist and pass applicable semantic
checks, but their packaging hash is not treated as immutable.

`value_binding` is deliberately schema-neutral. Each operand resolves to a JSON
value: either a selected value from a declared JSON file or the SHA-256 string
of a declared physical file. The check passes only when every canonical value
is identical. Reports commit to the canonical values by SHA-256 rather than
copying the raw values into proof output.

## Reference machine, not a platform restriction

The authoritative development reference is CPython 3.13 on Linux. CI also runs
the same proof suite on CPython 3.11 and 3.12. A new environment becomes a
supported compatibility target by passing the same validators; it is not
accepted or rejected merely from a label.

See:

- [Architecture](docs/ARCHITECTURE.md)
- [Threat model](docs/THREAT_MODEL.md)
- [Reference machine](docs/REFERENCE_MACHINE.md)

## Development proof command

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
