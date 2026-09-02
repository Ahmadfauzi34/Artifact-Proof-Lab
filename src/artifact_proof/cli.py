from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .engine import verify_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="artifact-proof", description="Verify an artifact against an explicit proof manifest.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify", help="verify a directory or ZIP artifact")
    verify.add_argument("artifact", type=Path)
    verify.add_argument("--manifest", default="ARTIFACT_PROOF.json", help="manifest path inside the artifact")
    verify.add_argument("--profile", choices=("sealed", "live"), default="sealed")
    verify.add_argument("--manifest-sha256", help="detached trust anchor for the manifest")
    verify.add_argument("--require-trust-anchor", action="store_true")
    verify.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = verify_artifact(
        args.artifact,
        manifest_path=args.manifest,
        profile=args.profile,
        manifest_sha256=args.manifest_sha256,
        require_trust_anchor=args.require_trust_anchor,
    )
    if args.json_output:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"Artifact Proof: {report.status}")
        print(f"artifact : {report.artifact}")
        print(f"profile  : {report.profile}")
        if report.manifest_sha256:
            print(f"manifest : {report.manifest_sha256}")
        for finding in report.findings:
            print(f"[{finding.status.value:4}] {finding.check_id}: {finding.message}")
    return 0 if report.passed else 1
