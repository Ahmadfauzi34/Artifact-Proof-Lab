from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_payload(project_root: Path, output: Path) -> None:
    lib = output / "lib"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    shutil.copytree(project_root / "gym", lib / "gym", ignore=ignore)
    shutil.copytree(
        project_root / "src" / "artifact_proof",
        lib / "artifact_proof",
        ignore=ignore,
    )


def _python_config() -> str:
    versioned = shutil.which(
        f"python{sys.version_info.major}.{sys.version_info.minor}-config"
    )
    fallback = shutil.which("python3-config")
    config = versioned or fallback
    if config is None:
        raise SystemExit("matching python-config is required to build the binary runtime")
    return config


def compile_launcher(project_root: Path, output: Path) -> None:
    entry = project_root / "gym" / "binary_runtime_entry.py"
    if not entry.is_file():
        raise SystemExit(f"binary entrypoint missing: {entry}")
    if shutil.which("gcc") is None:
        raise SystemExit("gcc is required to build the binary runtime")

    python_config = _python_config()
    with tempfile.TemporaryDirectory(prefix="proof-gym-build-") as tmp:
        tmpdir = Path(tmp)
        generated_c = tmpdir / "proof_gym_runtime.c"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "cython",
                "--embed",
                "-3",
                str(entry),
                "-o",
                str(generated_c),
            ],
            check=True,
        )
        cflags = shlex.split(
            subprocess.check_output(
                [python_config, "--embed", "--cflags"],
                text=True,
            )
        )
        ldflags = shlex.split(
            subprocess.check_output(
                [python_config, "--embed", "--ldflags"],
                text=True,
            )
        )
        subprocess.run(
            [
                "gcc",
                "-O2",
                "-s",
                *cflags,
                str(generated_c),
                "-o",
                str(output / "proof-gym-runtime"),
                *ldflags,
            ],
            check=True,
        )


def build_manifest(
    output: Path,
    source_commit: str | None,
) -> dict:
    payload = {}
    for path in sorted((output / "lib").rglob("*")):
        if path.is_file():
            payload[str(path.relative_to(output))] = {
                "sha256": sha256_file(path)
            }

    binary = output / "proof-gym-runtime"
    return {
        "format": "proof-gym-binary-runtime-v1",
        "source_commit": source_commit,
        "artifact": {
            "binary": "proof-gym-runtime",
            "binary_sha256": sha256_file(binary),
            "payload": payload,
        },
        "build": {
            "backend": "cython-embed-gcc",
            "python_version": sys.version.split()[0],
            "python_soabi": sysconfig.get_config_var("SOABI"),
            "self_contained": False,
            "requires_python_command_at_runtime": False,
            "requires_system_libpython": True,
        },
        "runtime_roles": sorted(
            [
                "adaptive-train",
                "negative-train",
                "on-policy-train",
                "curriculum",
                "reference",
                "subprocess-reference",
                "private-pack",
                "isolated-private-pack",
                "external-agent-reference",
                "attested-external-reference",
                "private-host",
                "agent-worker",
                "agent-socket-server",
                "validate",
            ]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build Artifact-Proof-Lab binary runtime bundle"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/proof-gym-runtime"),
    )
    parser.add_argument("--source-commit")
    args = parser.parse_args(argv)

    root = args.project_root.resolve()
    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    copy_payload(root, output)
    compile_launcher(root, output)

    document = build_manifest(output, args.source_commit)
    (output / "BINARY_RUNTIME_MANIFEST.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
