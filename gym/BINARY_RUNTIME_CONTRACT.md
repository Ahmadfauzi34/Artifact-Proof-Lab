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


## Restricted temporary-script compatibility

Some trusted reference environments intentionally execute generated Python
probes through `sys.executable -B <script.py>`. In a Cython-embedded runtime,
`sys.executable` points back to `proof-gym-runtime`, so the binary must
preserve the relevant Python CLI semantics without becoming a general-purpose
interpreter.

The binary therefore accepts script-path execution only when all of these hold:

- the path is absolute and ends in `.py`;
- the file is a regular file;
- the path is inside the system temporary root;
- its first relative directory starts with `gym-`;
- no path component is a symlink.

During that restricted execution, the script directory is temporarily inserted
at `sys.path[0]`, matching the import behavior required by
`python /path/to/script.py`. This is what allows a generated
`test_calc.py` to import its sibling `calc.py`.

Arbitrary scripts outside the bounded `gym-*` temporary workspace remain
rejected. The compatibility path exists for trusted host-created probes and does
not grant the evaluated external AgentEndpoint general script-execution
authority.

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

The full G4 execution path was subsequently reconstructed from GitHub blob-bound
runtime modules and executed through both source and binary coordinates. The
initial binary run exposed two repository-probe compatibility defects:
unsupported script-path self-spawn and missing script-directory `sys.path[0]`
semantics. After the restricted compatibility fix, the deterministic G4 source
and binary outputs were byte-identical across actual ReferenceGym, LocalReferenceHost,
Artifact-Proof verification, nine train tasks, five validation tasks, and the
bundled holdout.

This establishes reference G4 binary-runtime equivalence for that execution
path. It remains reference-gym evidence, not native competence or independent
authority evidence.
