from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
import unicodedata

from .errors import SourceError


_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")


def canonical_relative_path(raw: str) -> str:
    """Return a strict portable path or reject ambiguous/escaping input."""
    if not isinstance(raw, str) or not raw:
        raise SourceError("path must be a non-empty string")
    if "\x00" in raw:
        raise SourceError("path contains NUL")
    if "\\" in raw:
        raise SourceError(f"path must use '/' separators: {raw!r}")
    if raw.startswith("/") or _DRIVE_PREFIX.match(raw):
        raise SourceError(f"absolute path is forbidden: {raw!r}")
    if unicodedata.normalize("NFC", raw) != raw:
        raise SourceError(f"path is not NFC-normalized: {raw!r}")
    parts = raw.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise SourceError(f"path is not canonical: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or path.as_posix() != raw:
        raise SourceError(f"path is not canonical: {raw!r}")
    return raw


def safe_join(root: Path, relative: str) -> Path:
    canonical = canonical_relative_path(relative)
    root_resolved = root.resolve()
    candidate = (root_resolved / canonical).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise SourceError(f"path escapes artifact root: {relative!r}") from exc
    return candidate
