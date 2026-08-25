"""Content-addressed disk cache.

Exists mainly so development is survivable: without it, every code change in a
later stage re-downloads, re-normalizes, re-transcribes, and re-diarizes the
whole video before reaching the line being worked on. Re-running a feature-
length film from scratch for a one-line change is the difference between a
minute and half an hour.

Keyed by content where possible (audio hash for a transcript) and by source URL
otherwise (for the download itself, whose content is not knowable in advance).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from dialogue_extractor.infra.logging import get_logger

log = get_logger(__name__)

_CACHE_VERSION = "v1"
"""Bumping this invalidates every entry.

Necessary whenever the *meaning* of cached data changes -- a different Whisper
model, a changed alignment rule -- because the key would otherwise still match
while the value no longer means the same thing.
"""


def make_key(*parts: str) -> str:
    """Stable hash of the given parts, namespaced by cache version."""
    digest = hashlib.sha256()
    digest.update(_CACHE_VERSION.encode())
    for part in parts:
        digest.update(b"\x00")
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()[:32]


class DiskCache:
    """JSON values plus associated binary artefacts, under one directory."""

    def __init__(self, root: Path, *, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _meta_path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def path_for(self, key: str, suffix: str) -> Path:
        """Stable path for a binary artefact belonging to `key`."""
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        return self.root / f"{key}{suffix}"

    def get(self, key: str) -> Any | None:
        """Return the cached value, or None on miss.

        A corrupt entry is treated as a miss and deleted rather than raised: a
        cache is an optimization, and a half-written file from an interrupted
        run should never be able to fail the pipeline.
        """
        if not self.enabled:
            return None
        meta = self._meta_path(key)
        if not meta.exists():
            return None
        try:
            with meta.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            log.warning("cache.corrupt_entry_evicted", key=key, error=type(exc).__name__)
            self.evict(key)
            return None
        log.debug("cache.hit", key=key)
        return payload

    def put(self, key: str, value: Any) -> None:
        """Store `value`, written atomically so an interrupted run leaves no
        half-written entry that a later run would read back as valid."""
        if not self.enabled:
            return
        meta = self._meta_path(key)
        tmp = meta.with_suffix(".json.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(value, handle)
            tmp.replace(meta)
        except (OSError, TypeError) as exc:
            log.warning("cache.write_failed", key=key, error=type(exc).__name__)
            tmp.unlink(missing_ok=True)
            return
        log.debug("cache.stored", key=key)

    def store_file(self, key: str, source: Path, suffix: str) -> Path:
        """Copy `source` into the cache and return its cached path."""
        if not self.enabled:
            return source
        destination = self.path_for(key, suffix)
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        return destination

    def has_file(self, key: str, suffix: str) -> bool:
        if not self.enabled:
            return False
        path = self.path_for(key, suffix)
        return path.exists() and path.stat().st_size > 0

    def evict(self, key: str) -> None:
        for path in self.root.glob(f"{key}*"):
            path.unlink(missing_ok=True)

    def clear(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)


class NullCache:
    """No-op cache. Satisfies the same shape when caching is switched off."""

    enabled = False

    def get(self, key: str) -> Any | None:
        return None

    def put(self, key: str, value: Any) -> None:
        return None

    def path_for(self, key: str, suffix: str) -> Path:
        return Path(f"{key}{suffix}")
