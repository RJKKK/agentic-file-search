"""Object storage abstraction for uploaded documents."""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol

_DEFAULT_OBJECT_STORE_DIR = "data/object_store"
_FILENAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class BlobHead:
    """Metadata about one stored blob."""

    object_key: str
    storage_uri: str
    size: int
    absolute_path: str


class BlobStore(Protocol):
    """Minimal blob store interface used by the document library."""

    def put(
        self,
        *,
        object_key: str,
        data: BinaryIO,
    ) -> BlobHead:
        """Store a blob and return its persisted metadata."""

    def get(self, *, object_key: str) -> bytes:
        """Read the full blob payload."""

    def materialize(self, *, object_key: str) -> str:
        """Return a local readable path for the blob."""

    def delete(self, *, object_key: str) -> bool:
        """Best-effort delete for one blob."""

    def head(self, *, object_key: str) -> BlobHead | None:
        """Return blob metadata if present."""

    def delete_prefix(self, *, prefix: str) -> int:
        """Best-effort delete for all blobs under a prefix."""

    def list_prefix(self, *, prefix: str) -> list[BlobHead]:
        """List blobs under one prefix."""


def resolve_object_store_dir() -> Path:
    """Resolve the configured local object-store root."""
    configured = os.getenv("FS_EXPLORER_OBJECT_STORE_DIR", _DEFAULT_OBJECT_STORE_DIR)
    return Path(configured).expanduser().resolve()


def sanitize_filename(filename: str) -> str:
    """Keep uploaded display names safe for object keys."""
    base = Path(filename or "").name.strip()
    if not base:
        return "document"
    sanitized = _FILENAME_SANITIZE_RE.sub("-", base).strip("-.")
    return sanitized or "document"


class LocalBlobStore:
    """Filesystem-backed blob store used as the v1 object storage adapter."""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir or resolve_object_store_dir()).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_key_path(self, object_key: str) -> Path:
        safe_key = object_key.strip().replace("\\", "/").lstrip("/")
        target = (self.root_dir / safe_key).resolve()
        target.relative_to(self.root_dir)
        return target

    def _head_for_path(self, object_key: str, target: Path) -> BlobHead:
        stat = target.stat()
        return BlobHead(
            object_key=object_key,
            storage_uri=f"blob://library/default/{object_key}",
            size=int(stat.st_size),
            absolute_path=str(target),
        )

    def _prune_empty_parents(self, start: Path) -> None:
        current = start
        while current != self.root_dir:
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent

    def put(
        self,
        *,
        object_key: str,
        data: BinaryIO,
    ) -> BlobHead:
        target = self._resolve_key_path(object_key)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            shutil.copyfileobj(data, handle)
        return self._head_for_path(object_key, target)

    def get(self, *, object_key: str) -> bytes:
        return self._resolve_key_path(object_key).read_bytes()

    def materialize(self, *, object_key: str) -> str:
        target = self._resolve_key_path(object_key)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"Blob not found: {object_key}")
        return str(target)

    def delete(self, *, object_key: str) -> bool:
        target = self._resolve_key_path(object_key)
        if not target.exists():
            return False
        target.unlink(missing_ok=True)
        self._prune_empty_parents(target.parent)
        return True

    def head(self, *, object_key: str) -> BlobHead | None:
        target = self._resolve_key_path(object_key)
        if not target.exists() or not target.is_file():
            return None
        return self._head_for_path(object_key, target)

    def delete_prefix(self, *, prefix: str) -> int:
        target = self._resolve_key_path(prefix.rstrip("/"))
        if not target.exists():
            return 0
        deleted = 0
        if target.is_file():
            target.unlink(missing_ok=True)
            return 1
        for child in sorted(target.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink(missing_ok=True)
                deleted += 1
            elif child.is_dir():
                child.rmdir()
        target.rmdir()
        self._prune_empty_parents(target.parent)
        return deleted

    def list_prefix(self, *, prefix: str) -> list[BlobHead]:
        target = self._resolve_key_path(prefix.rstrip("/"))
        if not target.exists():
            return []
        if target.is_file():
            return [self._head_for_path(prefix.rstrip("/"), target)]
        results: list[BlobHead] = []
        for child in sorted(target.rglob("*")):
            if not child.is_file():
                continue
            object_key = str(child.relative_to(self.root_dir)).replace("\\", "/")
            results.append(self._head_for_path(object_key, child))
        return results
