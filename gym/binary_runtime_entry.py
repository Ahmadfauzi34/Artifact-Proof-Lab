from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys
import tempfile

ROLE_MODULES = {
    "negative-train": "gym.run_negative_experience_training_reference",
    "on-policy-train": "gym.run_on_policy_training_reference",
    "adaptive-train": "gym.run_adaptive_training_reference",
    "curriculum": "gym.run_curriculum_reference",
    "reference": "gym.run_reference",
    "subprocess-reference": "gym.run_subprocess_reference",
    "private-pack": "gym.run_private_pack_reference",
    "isolated-private-pack": "gym.run_isolated_private_pack_reference",
    "external-agent-reference": "gym.run_external_agent_reference",
    "external-agent-train": "gym.run_external_agent_training",
    "attested-external-reference": "gym.run_attested_external_reference",
    "private-host": "gym.private_host_server",
    "agent-worker": "gym.agent_worker",
    "agent-socket-server": "gym.agent_socket_server",
    "validate": "gym.validate_baseline",
}
ALLOWED_MODULES = frozenset(ROLE_MODULES.values())


def _bundle_root() -> Path:
    return Path(sys.argv[0]).resolve().parent


def _bootstrap_bundle() -> None:
    lib = _bundle_root() / "lib"
    if not lib.is_dir():
        raise SystemExit(f"runtime payload missing: {lib}")
    sys.path.insert(0, str(lib))
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def _usage() -> str:
    roles = "\n  ".join(sorted(ROLE_MODULES))
    return (
        "usage: proof-gym-runtime <role> [args...]\n"
        "       proof-gym-runtime -B -m <allowed-gym-module> [args...]\n\n"
        f"roles:\n  {roles}"
    )


def _allowed_temp_script(raw: str) -> Path | None:
    candidate = Path(raw)
    if not candidate.is_absolute() or candidate.suffix != ".py":
        return None

    temp_root = Path(tempfile.gettempdir()).resolve()
    try:
        relative = candidate.relative_to(temp_root)
    except ValueError:
        return None
    if (
        not relative.parts
        or not relative.parts[0].startswith("gym-")
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        return None

    current = temp_root
    try:
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                return None
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None

    if not resolved.is_file():
        return None
    try:
        resolved.relative_to(temp_root)
    except ValueError:
        return None
    return resolved


def _resolve(argv: list[str]) -> tuple[str, str, list[str]]:
    args = list(argv)
    while args and args[0] == "-B":
        args.pop(0)
    if not args:
        raise SystemExit(_usage())
    if args[0] == "-m":
        if len(args) < 2:
            raise SystemExit("-m requires a module name")
        module = args[1]
        rest = args[2:]
        if module not in ALLOWED_MODULES:
            raise SystemExit(f"module is not allowed by binary runtime: {module}")
        return "module", module, rest

    script = _allowed_temp_script(args[0])
    if script is not None:
        return "script", str(script), args[1:]

    role = args[0]
    try:
        module = ROLE_MODULES[role]
    except KeyError as exc:
        raise SystemExit(f"unknown runtime role: {role}\n{_usage()}") from exc
    return "module", module, args[1:]


def _execute(mode: str, target: str, rest: list[str]) -> int:
    sys.argv = [target, *rest]
    try:
        if mode == "script":
            script_dir = str(Path(target).resolve().parent)
            sys.path.insert(0, script_dir)
            try:
                runpy.run_path(target, run_name="__main__")
            finally:
                if sys.path and sys.path[0] == script_dir:
                    sys.path.pop(0)
        else:
            runpy.run_module(target, run_name="__main__", alter_sys=True)
    except SystemExit as exc:
        if exc.code is None:
            return 0
        if isinstance(exc.code, int):
            return exc.code
        print(exc.code, file=sys.stderr)
        return 1
    return 0


def main() -> int:
    _bootstrap_bundle()
    mode, target, rest = _resolve(sys.argv[1:])
    return _execute(mode, target, rest)



if __name__ == "__main__":
    raise SystemExit(main())
