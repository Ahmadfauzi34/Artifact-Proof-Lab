# Architecture

## Proof pipeline

Artifact Proof Lab uses a one-way verification pipeline:

```text
untrusted artifact
  -> bounded source inventory
  -> strict manifest parse
  -> detached-anchor observation
  -> coverage and byte identity
  -> declared semantic checks
  -> immutable proof report
```

No semantic check executes code from the artifact. SQLite databases are copied
to an ephemeral file and opened read-only with `immutable=1`. Outer ZIP files
are inspected in place. Nested backup ZIPs are bounded before their selected
member is read.

## Boundaries

- `source.py` owns safe container access and resource bounds.
- `manifest.py` owns the versioned contract and rejects unknown fields.
- `checks.py` owns observations; checks return findings and do not mutate state.
- `engine.py` owns ordering and converts deterministic rejection into a report.
- `cli.py` is presentation only and cannot change verification meaning.

## Profiles

`sealed` describes an immutable transfer artifact. Every declared file hash and
every sealed-only relationship must match packaging state.

`live` describes a capsule after legitimate state evolution. Immutable runtime
and harness files remain hash-gated. Explicitly mutable files are not forced
back to their packaging hashes, while applicable integrity checks still run.

This prevents both extremes: treating mutable learned state as an executable,
and disabling integrity for the whole capsule merely because one state file may
advance.

## Trust boundary

The manifest is deliberately not allowed to hash itself. Its SHA-256 can be
supplied out-of-band. Without that anchor, a PASS report means the artifact is
internally consistent with its included contract; it does not identify who
authored that contract.
