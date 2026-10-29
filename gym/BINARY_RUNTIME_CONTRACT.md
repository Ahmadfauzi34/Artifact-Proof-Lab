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
artifact-verify
validate
```

It also accepts the restricted compatibility form:

```text
proof-gym-runtime -B -m gym.private_host_server ...
proof-gym-runtime -B -m gym.agent_worker ...
proof-gym-runtime artifact-verify verify /path/to/artifact --json
```

Only explicitly allowlisted modules are accepted through `-m`. The host-facing
`artifact-verify` role maps only to the bundled `artifact_proof` verifier; it
does not expose arbitrary module execution.

This preserves existing subprocess code that uses `sys.executable -B -m ...`.
One binary artifact may therefore execute as multiple OS processes while keeping
host and agent logical/process boundaries intact.



## Host-facing Artifact-Proof verification

The bundle carries the Artifact-Proof implementation as part of its proof
surface. The `artifact-verify` role makes that verifier reachable through the
same executable coordinate:

```text
proof-gym-runtime artifact-verify verify /path/to/artifact --profile sealed --json
```

This role is read/verify oriented. It does not grant evaluated agents access to
host files, learning state, private holdout material, or arbitrary Python module
execution. File visibility remains whatever authority the invoking host process
already has. The role changes deployment reachability, not proof or workflow
authority.

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


## CI artifact coordinate

The repository CI has a dedicated Python 3.13 Linux binary job that invokes
`scripts/build_binary_runtime.py --source-commit "$GITHUB_SHA"`, validates the
manifest identity, runs the bundled validator, and compares deterministic
on-policy and typed-negative reference outputs between source and binary
coordinates.

The successful bundle is uploaded as:

```text
proof-gym-runtime-linux-x86_64-cpython313
```

The artifact remains ABI-bound because this baseline requires matching system
`libpython`. Consumers must verify `BINARY_RUNTIME_MANIFEST.json`, including
the exact source commit, binary/payload SHA-256 values, Python version and SOABI,
before execution.

The runtime also exposes `external-agent-train`. That role does not mutate an
external agent. It executes proof-gated train episodes against a pre-existing
AgentEndpoint and emits G4.3 portable training receipts for a separate agent-side
learning boundary.


## Cross-version training compatibility

A binary runtime identity and a training compatibility identity are deliberately
different.

```text
source commit / bundle payload / executable SHA
                 !=
training semantic compatibility
```

New builds include `compatibility` metadata in
`BINARY_RUNTIME_MANIFEST.json`. The canonical contract lives at
`gym/contracts/binary_runtime_compatibility.json`.

The compatibility fingerprint covers the proof engine, trajectory ledger,
agent protocol, external-agent boundary, host boundary and G4.3 receipt
generation path. Documentation-only or build-provenance changes outside this
surface do not automatically make historical training evidence incompatible.

Evidence from two runtime bundles may be aggregated only when all required
conditions hold:

```text
exact Python SOABI
+ same compatibility-contract version
+ same agent protocol
+ same external-training receipt format
+ external-agent-train role present
+ same training_surface_sha256
+ explicit behavior-equivalence evidence
```

A source-commit mismatch or executable-SHA mismatch alone is not sufficient to
reject evidence. Conversely, matching ABI alone is never sufficient to merge
evidence.

If a required compatibility dimension differs, the required action is
`ISOLATE_EVIDENCE`. If contract/surface checks match but behavior equivalence
has not been established, the runtime remains
`REQUIRE_BEHAVIOR_PROOF`. Only a full match may reach
`ALLOW_AGGREGATION`.

This compatibility result is provenance/admission metadata only. It is not
truth, proof of task success, native competence, or promotion authority.
