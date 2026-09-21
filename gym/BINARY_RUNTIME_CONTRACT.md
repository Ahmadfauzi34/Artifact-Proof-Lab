# Binary Runtime Contract — Cython Embedded Baseline

This contract defines the first executable packaging coordinate for the gym.
It is a deployment/runtime contract, not a new proof authority and not a native
competence claim.

## Goal

Runtime consumers should be able to invoke a binary artifact rather than
reconstructing a Python environment with `git clone`, `pip install`, or
`python -m ...` commands.

The baseline bundle is:

```text
proof-gym-runtime        ELF executable
lib/
  gym/                   gym runtime payload + contracts/tasks
  artifact_proof/        proof implementation payload
BINARY_RUNTIME_MANIFEST.json
```

The executable is built with Cython `--embed` plus GCC. The payload remains
Python source/data in this baseline so auditability stays high while the launch
surface becomes a binary.

## Runtime properties

- runtime entry is an ELF executable on Linux;
- the user invokes `proof-gym-runtime`, not a Python command;
- no `git clone` is required by the runtime bundle;
- no `pip install` is required by the runtime bundle;
- no Python executable command is required at runtime;
- the baseline is **not self-contained** and dynamically requires the matching
  system `libpython` ABI plus normal system libraries;
- later Nuitka/standalone packaging may remove that dependency without changing
  gym/proof semantics.

## Multi-role/self-spawn semantics

The same executable exposes named roles including:

```text
adaptive-train
curriculum
reference
private-pack
private-host
agent-worker
agent-socket-server
validate
```

It also accepts the restricted compatibility form:

```text
proof-gym-runtime -B -m gym.private_host_server ...
proof-gym-runtime -B -m gym.agent_worker ...
```

Only explicitly allowlisted gym modules are accepted through `-m`.

This preserves existing subprocess code that uses `sys.executable -B -m ...`.
One binary artifact may therefore execute as multiple OS processes while keeping
host and agent logical/process boundaries intact.

## Authority boundary

Binary packaging does not collapse authorities.

A process launched from the same executable does not automatically gain the
private host's semantic authority, proof authority, evaluator state, or learning
write authority. Existing G1-G4 protocol/admission rules remain controlling.

For native/private deployment, evaluated-agent authority still needs an
independent sandbox/container/VM/remote boundary as described by G2.3-G2.5.

## Build identity

`BINARY_RUNTIME_MANIFEST.json` records:

- binary SHA-256;
- exact payload-file SHA-256 values;
- source commit supplied to the builder;
- Python version and SOABI;
- build backend;
- self-contained/runtime-dependency flags;
- allowed runtime roles.

Audit or release builds should always pass an exact source commit to
`--source-commit`.

## Behavior-equivalence rule

Binary availability is not sufficient for promotion.

For a checkpoint to claim binary runtime validation, the same logical workload
must be executed through source and binary coordinates and their semantic output
must agree. The preferred strict check for deterministic reference runs is
byte-identical output.

```text
Behavior(source) == Behavior(binary)
```

The G4 standalone smoke established this rule interactively before this contract
was added: the Cython/GCC binary and Python source produced byte-identical output
for policy learning, validation simulation, holdout transfer, and ledger-tamper
rejection.

That smoke does **not** by itself establish full ReferenceGym + Artifact-Proof
binary validation. The full repository binary bundle must be built and executed
before making that stronger claim.
