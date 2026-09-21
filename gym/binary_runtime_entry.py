from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys

ROLE_MODULES = {
    "adaptive-train": "gym.run_adaptive_training_reference",
    "curriculum": "gym.run_curriculum_reference",
    "reference": "gym.run_reference",
    "subprocess-reference": "gym.run_subprocess_reference",
    "private-pack": "gym.run_private_pack_reference",
    "isolated-private-pack": "gym.run_isolated_private_pack_reference",
    "external-agent-reference": "gym.run_external_agent_reference",
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


def _resolve(argv: list[str]) -> tuple[str, list[str]]:
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
        return module, rest
    role = args[0]
    try:
        module = ROLE_MODULES[role]
    except KeyError as exc:
        raise SystemExit(f"unknown runtime role: {role}\n{_usage()}") from exc
    return module, args[1:]


def main() -> int:
    _bootstrap_bundle()
    module, rest = _resolve(sys.argv[1:])
    sys.argv = [module, *rest]
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    except SystemExit as exc:
        if exc.code is None:
            return 0
        if isinstance(exc.code, int):
            return exc.code
        print(exc.code, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
