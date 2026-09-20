from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def validate(root: Path) -> list[str]:
    errors: list[str] = []
    task_files = sorted((root / "tasks").glob("*/*.json"))
    if not task_files:
        return ["no public task files found"]
    domains: set[str] = set()
    skills: dict[str, set[str]] = {}
    splits: set[str] = set()
    ids: set[str] = set()
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
        for skill in raw.get("skill_targets", []):
            skills.setdefault(skill, set()).add(domain)
        if raw.get("environment", {}).get("runner") != "reference-v1":
            errors.append(f"{path}: unsupported environment runner")
        budget = raw.get("budget", {})
        if not isinstance(budget.get("max_steps"), int) or budget.get("max_steps", 0) < 1:
            errors.append(f"{path}: invalid max_steps")

    if len(domains) < 3:
        errors.append(f"expected at least 3 domains, found {len(domains)}")
    if not any(len(domain_set) >= 2 for domain_set in skills.values()):
        errors.append("no semantic skill is exercised across at least 2 domains")
    for required in ("train", "validation", "holdout"):
        if required not in splits:
            errors.append(f"missing {required} split")

    observation_schema = json.loads((root / "contracts" / "observation_contract.schema.json").read_text())
    schema_required = set(observation_schema.get("required", []))
    if not REQUIRED_OBSERVATION_KEYS <= schema_required:
        errors.append("observation contract does not require full identity/provenance surface")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Proof-Gated Adaptive Agent Gym baseline invariants")
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    args = parser.parse_args(argv)
    errors = validate(args.root)
    if errors:
        print("GYM BASELINE: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("GYM BASELINE: PASS")
    print("- public/oracle surface separation: PASS")
    print("- observation identity/provenance contract: PASS")
    print("- >=3 domains: PASS")
    print("- cross-domain semantic skill: PASS")
    print("- train/validation/holdout separation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
