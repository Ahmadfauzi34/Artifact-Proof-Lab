from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile


COMPATIBILITY_CONTRACT_RELATIVE = Path(
    "lib/gym/contracts/binary_runtime_compatibility.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(document: object) -> str:
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def copy_payload(project_root: Path, output: Path) -> None:
    lib = output / "lib"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    shutil.copytree(project_root / "gym", lib / "gym", ignore=ignore)
    shutil.copytree(
        project_root / "src" / "artifact_proof",
        lib / "artifact_proof",
        ignore=ignore,
    )


def _load_compatibility_contract(output: Path) -> dict:
    path = output / COMPATIBILITY_CONTRACT_RELATIVE
    if not path.is_file():
        raise SystemExit(f"binary compatibility contract missing: {path}")
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"binary compatibility contract is unreadable: {exc}"
        ) from exc
    required = {
        "schema",
        "compatibility_version",
        "training_role",
        "agent_protocol",
        "external_training_receipt_format",
        "learning_write_authority",
        "required_admission_dimensions",
        "aggregation_policy",
        "training_surface_files",
    }
    if set(document) != required:
        raise SystemExit("binary compatibility contract fields are invalid")
    files = document["training_surface_files"]
    if (
        not isinstance(files, list)
        or not files
        or any(not isinstance(item, str) or not item for item in files)
        or len(files) != len(set(files))
    ):
        raise SystemExit("training_surface_files must be unique non-empty paths")
    return document


def build_compatibility_metadata(output: Path) -> dict:
    contract = _load_compatibility_contract(output)
    entries = []
    for relative in sorted(contract["training_surface_files"]):
        path = output / relative
        if not path.is_file():
            raise SystemExit(f"training surface file missing: {relative}")
        entries.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
            }
        )
    surface_document = {
        "schema": "proof-gym-training-surface-v1",
        "files": entries,
    }
    return {
        "schema": str(contract["schema"]),
        "contract_version": int(contract["compatibility_version"]),
        "contract_path": COMPATIBILITY_CONTRACT_RELATIVE.as_posix(),
        "contract_sha256": sha256_file(
            output / COMPATIBILITY_CONTRACT_RELATIVE
        ),
        "training_role": str(contract["training_role"]),
        "agent_protocol": str(contract["agent_protocol"]),
        "external_training_receipt_format": str(
            contract["external_training_receipt_format"]
        ),
        "learning_write_authority": bool(
            contract["learning_write_authority"]
        ),
        "required_admission_dimensions": list(
            contract["required_admission_dimensions"]
        ),
        "aggregation_policy": dict(contract["aggregation_policy"]),
        "training_surface_file_count": len(entries),
        "training_surface_sha256": sha256_json(surface_document),
    }


def _runtime_roles_from_payload(output: Path) -> list[str]:
    entry = output / "lib" / "gym" / "binary_runtime_entry.py"
    if not entry.is_file():
        raise SystemExit(f"binary runtime entry payload missing: {entry}")
    try:
        namespace = runpy.run_path(str(entry))
    except Exception as exc:
        raise SystemExit(f"binary runtime roles are unreadable: {exc}") from exc
    roles = namespace.get("ROLE_MODULES")
    if (
        not isinstance(roles, dict)
        or not roles
        or any(
            not isinstance(role, str)
            or not role
            or not isinstance(module, str)
            or not module
            for role, module in roles.items()
        )
    ):
        raise SystemExit("binary runtime ROLE_MODULES is invalid")
    return sorted(roles)


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
    source_head_commit: str | None,
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
        "source_head_commit": source_head_commit,
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
        "compatibility": build_compatibility_metadata(output),
        "runtime_roles": _runtime_roles_from_payload(output),
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
    parser.add_argument("--source-head-commit")
    args = parser.parse_args(argv)

    root = args.project_root.resolve()
    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    copy_payload(root, output)
    compile_launcher(root, output)

    document = build_manifest(
        output,
        args.source_commit,
        args.source_head_commit,
    )
    (output / "BINARY_RUNTIME_MANIFEST.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
