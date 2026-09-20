from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .core import PublicTask


def load_task(path: Path | str) -> PublicTask:
    return PublicTask.from_dict(json.loads(Path(path).read_text()))


def load_tasks(root: Path | str) -> tuple[PublicTask, ...]:
    base = Path(root)
    return tuple(load_task(path) for path in sorted(base.glob("*/*.json")))


def task_paths(root: Path | str) -> Iterable[Path]:
    return sorted(Path(root).glob("*/*.json"))
