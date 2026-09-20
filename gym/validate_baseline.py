from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any

from .core import PublicTask


FORBIDDEN_PUBLIC_KEYS = {
    "ground_truth",
    "oracle",
    "oracle_facts",
    "expected_action",
    "expected_semantic_class",
    "reference_solution",
    "reference_patch",
    "hidden_validator",
    "reward_decomposition",
}
FORBIDDEN_AGENT_VIEW_FIELDS = {
    "split",
    "skill_targets",
    "environment",
    "metadata",
    "snapshot_id",
    "runner",
}
REQUIRED_TASK_KEYS = {
    "task_id",
    "domain",
    "skill_targets",
    "description",
    "environment",
    "allowed_actions",
    "budget",
    "split",
}
REQUIRED_OBSERVATION_KEYS = {
    "observation_id",
    "source_lineage_root",
    "check_id",
    "probe_id",
    "attempt_id",
    "sequence_index",
    "logical_tick",
    "observation_type",
    "status",
    "value",
    "cost",
}
OPAQUE_SNAPSHOT = re.compile(r"^env-[0-9a-f]{6,}$")


def _walk_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            found.add(str(key))
            found |= _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            found |= _walk_keys(child)
    return found


def _valid_optional_number(value: Any) -> bool:
    return value is None or (type(value) in {int, float} and math.isfinite(float(value)) and value >= 0)


def validate(root: Path) -> list[str]:
    errors: list[str] = []
    task_files = sorted((root / "tasks").glob("*/*.json"))
    if not task_files:
        return ["no host task files found"]
    domains: set[str] = set()
    train_domains: set[str] = set()
    holdout_domains: set[str] = set()
    skills: dict[str, set[str]] = {}
    splits: set[str] = set()
    ids: set[str] = set()
    snapshot_ids: set[str] = set()

    for path in task_files:
        try:
            raw = json.loads(path.read_text())
        except Exception as exc:
            errors.append(f"{path}: invalid JSON: {exc}")
            continue
        missing = REQUIRED_TASK_KEYS - set(raw)
        if missing:
            errors.append(f"{path}: missing keys {sorted(missing)}")
        forbidden = FORBIDDEN_PUBLIC_KEYS & _walk_keys(raw)
        if forbidden:
            errors.append(f"{path}: oracle leakage keys {sorted(forbidden)}")
        task_id = raw.get("task_id")
        if task_id in ids:
            errors.append(f"{path}: duplicate task_id {task_id}")
        ids.add(task_id)
        domain = raw.get("domain")
        if isinstance(domain, str):
            domains.add(domain)
        split = raw.get("split")
        if isinstance(split, str):
            splits.add(split)
            if split == "train" and isinstance(domain, str):
                train_domains.add(domain)
            if split == "holdout" and isinstance(domain, str):
                holdout_domains.add(domain)
        for skill in raw.get("skill_targets", []):
            skills.setdefault(skill, set()).add(domain)

        environment = raw.get("environment", {})
        if environment.get("runner") != "reference-v1":
            errors.append(f"{path}: unsupported environment runner")
        snapshot_id = environment.get("snapshot_id")
        if not isinstance(snapshot_id, str) or not OPAQUE_SNAPSHOT.fullmatch(snapshot_id):
            errors.append(f"{path}: snapshot_id must use opaque env-<hex> form")
        elif snapshot_id in snapshot_ids:
            errors.append(f"{path}: duplicate snapshot_id {snapshot_id}")
        else:
            snapshot_ids.add(snapshot_id)

        budget = raw.get("budget", {})
        if type(budget.get("max_steps")) is not int or budget.get("max_steps", 0) < 1:
            errors.append(f"{path}: invalid max_steps")
        if not _valid_optional_number(budget.get("max_tool_cost")):
            errors.append(f"{path}: invalid max_tool_cost")
        max_generative = budget.get("max_generative_calls")
        if max_generative is not None and (type(max_generative) is not int or max_generative < 0):
            errors.append(f"{path}: invalid max_generative_calls")

        metadata = raw.get("metadata", {})
        if metadata.get("evaluation_class") != "reference_contract" or metadata.get("native_competence_claim") is not False:
            errors.append(f"{path}: repo-bundled task must be reference_contract with native_competence_claim=false")

        try:
            view = PublicTask.from_dict(raw).agent_view()
            exposed = set(vars(view)) & FORBIDDEN_AGENT_VIEW_FIELDS
            if exposed:
                errors.append(f"{path}: agent view exposes host-only fields {sorted(exposed)}")
        except Exception as exc:
            errors.append(f"{path}: cannot build sanitized agent view: {exc}")

    if len(domains) < 5:
        errors.append(f"expected at least 5 domains, found {len(domains)}")
    if len(skills) < 5:
        errors.append(f"expected at least 5 skill families, found {len(skills)}")
    if not any(len(domain_set) >= 2 for domain_set in skills.values()):
        errors.append("no semantic skill is exercised across at least 2 domains")
    for required in ("train", "validation", "holdout"):
        if required not in splits:
            errors.append(f"missing {required} split")
    if holdout_domains and not (holdout_domains - train_domains):
        errors.append("holdout must include at least one domain absent from train")

    observation_schema = json.loads((root / "contracts" / "observation_contract.schema.json").read_text())
    schema_required = set(observation_schema.get("required", []))
    if not REQUIRED_OBSERVATION_KEYS <= schema_required:
        errors.append("observation contract does not require full identity/provenance surface")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Proof-Gated Adaptive Agent Gym hardened G1 invariants")
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    args = parser.parse_args(argv)
    errors = validate(args.root)
    if errors:
        print("GYM BASELINE: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("GYM BASELINE: PASS")
    print("- sanitized agent task surface: PASS")
    print("- opaque host snapshot identity: PASS")
    print("- observation identity/provenance contract: PASS")
    print("- >=5 domains and >=5 skill families: PASS")
    print("- cross-domain semantic skill: PASS")
    print("- train/validation/holdout separation: PASS")
    print("- holdout includes domain absent from train: PASS")
    print("- repo-bundled tasks are reference-contract only: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
