from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import EpisodeResult, PublicTask
from .ledger import ZERO_HASH, sha256_json


CURRICULUM_FORMAT = "proof-gym-curriculum-v1"
PROMOTION_RECEIPT_FORMAT = "proof-gym-promotion-receipt-v1"


class CurriculumError(RuntimeError):
    """Curriculum definition or promotion evidence violated a hard invariant."""


@dataclass(frozen=True)
class CurriculumStage:
    index: int
    name: str
    train_task_ids: tuple[str, ...]
    validation_task_ids: tuple[str, ...]
    required_skill_targets: tuple[str, ...]
    min_train_admission_rate: float
    min_validation_admission_rate: float
    max_budget_exhaustion_rate: float
    require_train_learning_updates: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CurriculumStage":
        expected = {
            "index",
            "name",
            "train_task_ids",
            "validation_task_ids",
            "required_skill_targets",
            "min_train_admission_rate",
            "min_validation_admission_rate",
            "max_budget_exhaustion_rate",
            "require_train_learning_updates",
        }
        if set(raw) != expected:
            raise CurriculumError("curriculum stage fields mismatch")
        index = raw.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise CurriculumError("curriculum stage index must be a non-negative integer")
        name = _required_string(raw, "name")
        train = _string_tuple(raw.get("train_task_ids"), "train_task_ids")
        validation = _string_tuple(
            raw.get("validation_task_ids"),
            "validation_task_ids",
        )
        skills = _string_tuple(
            raw.get("required_skill_targets"),
            "required_skill_targets",
        )
        if not train:
            raise CurriculumError("curriculum stage must contain train tasks")
        if not validation:
            raise CurriculumError("curriculum stage must contain validation tasks")
        if set(train).intersection(validation):
            raise CurriculumError("train and validation tasks must be disjoint")
        min_train = _rate(raw.get("min_train_admission_rate"), "min_train_admission_rate")
        min_validation = _rate(
            raw.get("min_validation_admission_rate"),
            "min_validation_admission_rate",
        )
        max_budget = _rate(
            raw.get("max_budget_exhaustion_rate"),
            "max_budget_exhaustion_rate",
        )
        require_updates = raw.get("require_train_learning_updates")
        if not isinstance(require_updates, bool):
            raise CurriculumError("require_train_learning_updates must be boolean")
        return cls(
            index=index,
            name=name,
            train_task_ids=train,
            validation_task_ids=validation,
            required_skill_targets=skills,
            min_train_admission_rate=min_train,
            min_validation_admission_rate=min_validation,
            max_budget_exhaustion_rate=max_budget,
            require_train_learning_updates=require_updates,
        )


@dataclass(frozen=True)
class Curriculum:
    curriculum_id: str
    stages: tuple[CurriculumStage, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Curriculum":
        if set(raw) != {"format", "curriculum_id", "stages"}:
            raise CurriculumError("curriculum fields mismatch")
        if raw.get("format") != CURRICULUM_FORMAT:
            raise CurriculumError("unsupported curriculum format")
        curriculum_id = _required_string(raw, "curriculum_id")
        stages_raw = raw.get("stages")
        if not isinstance(stages_raw, list) or not stages_raw:
            raise CurriculumError("curriculum stages must be a non-empty list")
        stages = tuple(CurriculumStage.from_mapping(item) for item in stages_raw)
        if tuple(stage.index for stage in stages) != tuple(range(len(stages))):
            raise CurriculumError("curriculum stage indexes must be contiguous from zero")
        names = [stage.name for stage in stages]
        if len(names) != len(set(names)):
            raise CurriculumError("curriculum stage names must be unique")
        return cls(curriculum_id=curriculum_id, stages=stages)


@dataclass(frozen=True)
class PromotionReceipt:
    format: str
    curriculum_id: str
    stage_index: int
    stage_name: str
    previous_checkpoint_sha256: str
    train_task_ids: tuple[str, ...]
    validation_task_ids: tuple[str, ...]
    train_admission_rate: float
    validation_admission_rate: float
    budget_exhaustion_rate: float
    train_learning_update_rate: float
    promoted: bool
    reasons: tuple[str, ...]
    checkpoint_sha256: str

    def payload(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "curriculum_id": self.curriculum_id,
            "stage_index": self.stage_index,
            "stage_name": self.stage_name,
            "previous_checkpoint_sha256": self.previous_checkpoint_sha256,
            "train_task_ids": list(self.train_task_ids),
            "validation_task_ids": list(self.validation_task_ids),
            "train_admission_rate": self.train_admission_rate,
            "validation_admission_rate": self.validation_admission_rate,
            "budget_exhaustion_rate": self.budget_exhaustion_rate,
            "train_learning_update_rate": self.train_learning_update_rate,
            "promoted": self.promoted,
            "reasons": list(self.reasons),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.payload(),
            "checkpoint_sha256": self.checkpoint_sha256,
        }


def load_curriculum(path: Path | str) -> Curriculum:
    raw = _strict_json(Path(path).read_bytes())
    return Curriculum.from_mapping(raw)


def validate_curriculum_tasks(
    curriculum: Curriculum,
    tasks: Sequence[PublicTask],
) -> None:
    by_id = {task.task_id: task for task in tasks}
    if len(by_id) != len(tasks):
        raise CurriculumError("task registry contains duplicate task_id")

    seen_train: set[str] = set()
    seen_validation: set[str] = set()
    for stage in curriculum.stages:
        stage_skills: set[str] = set()
        for task_id in stage.train_task_ids:
            task = _task(by_id, task_id)
            if task.split != "train":
                raise CurriculumError(
                    f"curriculum train task {task_id} has split {task.split}"
                )
            if task_id in seen_train:
                raise CurriculumError(
                    f"train task reused across curriculum stages: {task_id}"
                )
            seen_train.add(task_id)
            stage_skills.update(task.skill_targets)
        for task_id in stage.validation_task_ids:
            task = _task(by_id, task_id)
            if task.split != "validation":
                raise CurriculumError(
                    f"curriculum validation task {task_id} has split {task.split}"
                )
            if task_id in seen_validation:
                raise CurriculumError(
                    f"validation task reused across curriculum stages: {task_id}"
                )
            seen_validation.add(task_id)
            stage_skills.update(task.skill_targets)

        missing_skills = sorted(set(stage.required_skill_targets) - stage_skills)
        if missing_skills:
            raise CurriculumError(
                f"stage {stage.name} missing required skill coverage: {missing_skills}"
            )

    if seen_train.intersection(seen_validation):
        raise CurriculumError("task cannot be both train and validation in curriculum")


def evaluate_stage(
    curriculum: Curriculum,
    stage: CurriculumStage,
    train_results: Sequence[EpisodeResult],
    validation_results: Sequence[EpisodeResult],
    *,
    previous_checkpoint_sha256: str,
) -> PromotionReceipt:
    if stage not in curriculum.stages:
        raise CurriculumError("stage is not part of curriculum")
    _require_sha256(previous_checkpoint_sha256, "previous_checkpoint_sha256")

    train = _index_results(train_results, "train")
    validation = _index_results(validation_results, "validation")
    if set(train) != set(stage.train_task_ids):
        raise CurriculumError("train result set does not match curriculum stage")
    if set(validation) != set(stage.validation_task_ids):
        raise CurriculumError("validation result set does not match curriculum stage")

    if any(result.split != "train" for result in train.values()):
        raise CurriculumError("non-train result supplied to train promotion evidence")
    if any(result.split != "validation" for result in validation.values()):
        raise CurriculumError(
            "non-validation result supplied to validation promotion evidence"
        )
    if any(result.learning_updated for result in validation.values()):
        raise CurriculumError("validation result attempted a learning update")

    train_rate = _accepted_rate(train.values())
    validation_rate = _accepted_rate(validation.values())
    all_results = tuple(train.values()) + tuple(validation.values())
    budget_rate = (
        sum(1 for result in all_results if result.budget_exhausted)
        / len(all_results)
    )
    train_learning_rate = (
        sum(1 for result in train.values() if result.learning_updated)
        / len(train)
    )

    reasons: list[str] = []
    if train_rate < stage.min_train_admission_rate:
        reasons.append("train_admission_below_threshold")
    if validation_rate < stage.min_validation_admission_rate:
        reasons.append("validation_admission_below_threshold")
    if budget_rate > stage.max_budget_exhaustion_rate:
        reasons.append("budget_exhaustion_above_threshold")
    if (
        stage.require_train_learning_updates
        and train_learning_rate < train_rate
    ):
        reasons.append("admitted_train_episode_missing_learning_update")

    promoted = not reasons
    payload = {
        "format": PROMOTION_RECEIPT_FORMAT,
        "curriculum_id": curriculum.curriculum_id,
        "stage_index": stage.index,
        "stage_name": stage.name,
        "previous_checkpoint_sha256": previous_checkpoint_sha256,
        "train_task_ids": list(stage.train_task_ids),
        "validation_task_ids": list(stage.validation_task_ids),
        "train_admission_rate": train_rate,
        "validation_admission_rate": validation_rate,
        "budget_exhaustion_rate": budget_rate,
        "train_learning_update_rate": train_learning_rate,
        "promoted": promoted,
        "reasons": reasons,
    }
    checkpoint_sha256 = sha256_json(payload)
    return PromotionReceipt(
        format=PROMOTION_RECEIPT_FORMAT,
        curriculum_id=curriculum.curriculum_id,
        stage_index=stage.index,
        stage_name=stage.name,
        previous_checkpoint_sha256=previous_checkpoint_sha256,
        train_task_ids=stage.train_task_ids,
        validation_task_ids=stage.validation_task_ids,
        train_admission_rate=train_rate,
        validation_admission_rate=validation_rate,
        budget_exhaustion_rate=budget_rate,
        train_learning_update_rate=train_learning_rate,
        promoted=promoted,
        reasons=tuple(reasons),
        checkpoint_sha256=checkpoint_sha256,
    )


def verify_promotion_chain(
    curriculum: Curriculum,
    receipts: Sequence[PromotionReceipt],
) -> tuple[bool, str]:
    if len(receipts) > len(curriculum.stages):
        return False, "promotion chain longer than curriculum"
    previous = ZERO_HASH
    for expected_stage, receipt in zip(curriculum.stages, receipts):
        if receipt.format != PROMOTION_RECEIPT_FORMAT:
            return False, "unsupported promotion receipt format"
        if receipt.curriculum_id != curriculum.curriculum_id:
            return False, "promotion receipt curriculum mismatch"
        if receipt.stage_index != expected_stage.index:
            return False, "promotion stage index mismatch"
        if receipt.stage_name != expected_stage.name:
            return False, "promotion stage name mismatch"
        if receipt.previous_checkpoint_sha256 != previous:
            return False, "promotion checkpoint predecessor mismatch"
        if receipt.checkpoint_sha256 != sha256_json(receipt.payload()):
            return False, "promotion checkpoint hash mismatch"
        if not receipt.promoted and receipt is not receipts[-1]:
            return False, "promotion chain continues after failed stage"
        previous = receipt.checkpoint_sha256
    return True, "promotion chain valid"


def _index_results(
    results: Sequence[EpisodeResult],
    label: str,
) -> dict[str, EpisodeResult]:
    by_id = {result.task_id: result for result in results}
    if len(by_id) != len(results):
        raise CurriculumError(f"duplicate {label} episode result")
    if not by_id:
        raise CurriculumError(f"{label} episode results must not be empty")
    return by_id


def _accepted_rate(results) -> float:
    items = tuple(results)
    return sum(1 for result in items if result.accepted) / len(items)


def _task(by_id: Mapping[str, PublicTask], task_id: str) -> PublicTask:
    try:
        return by_id[task_id]
    except KeyError as exc:
        raise CurriculumError(f"curriculum references unknown task: {task_id}") from exc


def _strict_json(data: bytes) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CurriculumError("curriculum is not UTF-8") from exc

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        raw = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise CurriculumError(f"curriculum is not strict JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise CurriculumError("curriculum must be a JSON object")
    _reject_nonfinite(raw)
    return raw


def _reject_nonfinite(value: Any) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CurriculumError("curriculum contains non-finite number")
        return
    if isinstance(value, list):
        for item in value:
            _reject_nonfinite(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _reject_nonfinite(item)
        return
    raise CurriculumError("curriculum contains unsupported JSON value")


def _required_string(value: Mapping[str, Any], field: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item:
        raise CurriculumError(f"{field} must be a non-empty string")
    return item


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise CurriculumError(f"{field} must be a list")
    if any(not isinstance(item, str) or not item for item in value):
        raise CurriculumError(f"{field} entries must be non-empty strings")
    if len(value) != len(set(value)):
        raise CurriculumError(f"{field} must not contain duplicates")
    return tuple(value)


def _rate(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CurriculumError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise CurriculumError(f"{field} must be within [0, 1]")
    return number


def _require_sha256(value: str, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise CurriculumError(f"{field} must be 64 lowercase hex characters")
