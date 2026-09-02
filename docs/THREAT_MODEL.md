# Threat model

## In scope

- ZIP path traversal and platform-ambiguous member names;
- duplicate members and symlink-based indirection;
- encrypted entries that cannot be inspected deterministically;
- decompression bombs bounded by entries, file size, total size, and ratio;
- externally supplied source limits whose invalid numeric domain could disable
  a bound;
- missing, additional, or byte-modified files;
- corrupted SQLite state;
- cold backup that differs from the sealed active state;
- mutable-state hashes incorrectly treated as immutable during live operation;
- unanchored or mismatched manifest identity;
- environment declarations confused with demonstrated compatibility.
- post-validation mutation that removes a failed finding and rewrites the
  reported verdict.

## Out of scope for v0.1

- malware analysis of declared files;
- authenticity without an external digest or signature;
- cryptographic signing and key management;
- execution of build scripts or arbitrary validators from the artifact;
- remote transparency logs;
- recovery or repair of a rejected artifact.

Failing verification never triggers automatic repair. Mutation belongs to a
separate, explicitly authorized transition.
