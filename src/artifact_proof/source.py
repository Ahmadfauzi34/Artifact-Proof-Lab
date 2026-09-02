from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import stat
from typing import BinaryIO
import zipfile

from .errors import SourceError
from .paths import canonical_relative_path


@dataclass(frozen=True)
class SourceLimits:
    max_entries: int = 4096
    max_file_bytes: int = 64 * 1024 * 1024
    max_total_bytes: int = 256 * 1024 * 1024
    max_compression_ratio: float = 200.0

    def __post_init__(self) -> None:
        for field_name in ("max_entries", "max_file_bytes", "max_total_bytes"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")

        ratio = self.max_compression_ratio
        ratio_is_valid = bool(
            type(ratio) is int and ratio >= 0
            or type(ratio) is float and math.isfinite(ratio) and ratio >= 0
        )
        if not ratio_is_valid:
            raise ValueError("max_compression_ratio must be a finite non-negative number")


class ArtifactSource:
    def list_files(self) -> tuple[str, ...]:
        raise NotImplementedError

    def read_bytes(self, relative: str, *, max_bytes: int | None = None) -> bytes:
        raise NotImplementedError

    def sha256(self, relative: str) -> str:
        return hashlib.sha256(self.read_bytes(relative)).hexdigest()

    def close(self) -> None:
        return None

    def __enter__(self) -> "ArtifactSource":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class DirectorySource(ArtifactSource):
    def __init__(self, root: Path, limits: SourceLimits) -> None:
        if not root.is_dir():
            raise SourceError(f"artifact directory does not exist: {root}")
        self.root = root.resolve()
        self.limits = limits
        self._snapshots = self._inventory()

    def _inventory(self) -> dict[str, tuple[int, int, int, int]]:
        files: dict[str, tuple[int, int, int, int]] = {}
        total = 0
        for current, dirs, names in os.walk(self.root, followlinks=False):
            current_path = Path(current)
            for name in list(dirs):
                path = current_path / name
                if path.is_symlink():
                    raise SourceError(f"directory symlink is forbidden: {path.relative_to(self.root)}")
            for name in names:
                path = current_path / name
                relative = path.relative_to(self.root).as_posix()
                canonical_relative_path(relative)
                mode = path.lstat().st_mode
                if not stat.S_ISREG(mode):
                    raise SourceError(f"non-regular file is forbidden: {relative}")
                metadata = path.stat()
                size = metadata.st_size
                if size > self.limits.max_file_bytes:
                    raise SourceError(f"file exceeds size limit: {relative}")
                total += size
                files[relative] = (metadata.st_dev, metadata.st_ino, size, metadata.st_mtime_ns)
        if len(files) > self.limits.max_entries:
            raise SourceError("artifact exceeds entry-count limit")
        if total > self.limits.max_total_bytes:
            raise SourceError("artifact exceeds total-size limit")
        return files

    def list_files(self) -> tuple[str, ...]:
        return tuple(sorted(self._snapshots))

    def read_bytes(self, relative: str, *, max_bytes: int | None = None) -> bytes:
        canonical = canonical_relative_path(relative)
        expected = self._snapshots.get(canonical)
        if expected is None:
            raise SourceError(f"artifact file was not present in the bounded inventory: {canonical}")
        path = self.root / canonical
        current = self.root
        for part in canonical.split("/"):
            current = current / part
            try:
                metadata = current.lstat()
            except FileNotFoundError as exc:
                raise SourceError(f"artifact file disappeared after inventory: {canonical}") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise SourceError(f"artifact path became a symlink after inventory: {canonical}")
        limit = self.limits.max_file_bytes if max_bytes is None else min(max_bytes, self.limits.max_file_bytes)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise SourceError(f"artifact file cannot be opened safely: {canonical}: {exc}") from exc
        try:
            before = os.fstat(descriptor)
            observed_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            if observed_before != expected or not stat.S_ISREG(before.st_mode):
                raise SourceError(f"artifact file changed after inventory: {canonical}")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                data = handle.read(limit + 1)
            after = os.fstat(descriptor)
            observed_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            if observed_after != observed_before:
                raise SourceError(f"artifact file changed while being read: {canonical}")
        finally:
            os.close(descriptor)
        if len(data) > limit:
            raise SourceError(f"file exceeds read limit: {canonical}")
        return data


class ZipSource(ArtifactSource):
    def __init__(self, archive: Path, limits: SourceLimits) -> None:
        if not archive.is_file():
            raise SourceError(f"artifact ZIP does not exist: {archive}")
        try:
            self.archive = zipfile.ZipFile(archive, "r")
        except zipfile.BadZipFile as exc:
            raise SourceError("artifact is not a valid ZIP") from exc
        self.limits = limits
        try:
            self._members = self._inventory()
        except Exception:
            self.archive.close()
            raise

    def _inventory(self) -> dict[str, zipfile.ZipInfo]:
        infos = self.archive.infolist()
        if len(infos) > self.limits.max_entries:
            raise SourceError("ZIP exceeds entry-count limit")
        members: dict[str, zipfile.ZipInfo] = {}
        total = 0
        for info in infos:
            raw = info.filename[:-1] if info.is_dir() and info.filename.endswith("/") else info.filename
            canonical_relative_path(raw)
            if info.is_dir():
                continue
            if raw in members:
                raise SourceError(f"duplicate ZIP member: {raw}")
            unix_mode = info.external_attr >> 16
            file_type = stat.S_IFMT(unix_mode)
            if file_type == stat.S_IFLNK:
                raise SourceError(f"ZIP symlink is forbidden: {raw}")
            if file_type not in {0, stat.S_IFREG}:
                raise SourceError(f"non-regular ZIP member is forbidden: {raw}")
            if info.flag_bits & 0x1:
                raise SourceError(f"encrypted ZIP member is unsupported: {raw}")
            if info.file_size > self.limits.max_file_bytes:
                raise SourceError(f"ZIP member exceeds size limit: {raw}")
            ratio = info.file_size / max(info.compress_size, 1)
            if ratio > self.limits.max_compression_ratio:
                raise SourceError(f"ZIP member exceeds compression-ratio limit: {raw}")
            total += info.file_size
            members[raw] = info
        if total > self.limits.max_total_bytes:
            raise SourceError("ZIP exceeds total uncompressed-size limit")
        return members

    def list_files(self) -> tuple[str, ...]:
        return tuple(sorted(self._members))

    def read_bytes(self, relative: str, *, max_bytes: int | None = None) -> bytes:
        canonical = canonical_relative_path(relative)
        info = self._members.get(canonical)
        if info is None:
            raise SourceError(f"ZIP member is missing: {canonical}")
        limit = self.limits.max_file_bytes if max_bytes is None else min(max_bytes, self.limits.max_file_bytes)
        if info.file_size > limit:
            raise SourceError(f"ZIP member exceeds read limit: {canonical}")
        with self.archive.open(info, "r") as handle:
            data = _bounded_read(handle, limit)
        return data

    def close(self) -> None:
        self.archive.close()


def _bounded_read(handle: BinaryIO, limit: int) -> bytes:
    data = handle.read(limit + 1)
    if len(data) > limit:
        raise SourceError("decompressed data exceeds read limit")
    return data


def open_source(path: Path | str, limits: SourceLimits | None = None) -> ArtifactSource:
    source_path = Path(path)
    effective_limits = limits or SourceLimits()
    if source_path.is_dir():
        return DirectorySource(source_path, effective_limits)
    if source_path.is_file() and zipfile.is_zipfile(source_path):
        return ZipSource(source_path, effective_limits)
    raise SourceError("artifact must be a directory or ZIP archive")
