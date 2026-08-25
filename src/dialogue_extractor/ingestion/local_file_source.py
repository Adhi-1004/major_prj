"""Local-file acquisition -- the manual fallback for ingestion failures.

Not a convenience. Acquisition can fail for reasons entirely outside our
control: a site changes its player, a video is geo-blocked, a link expires. When
that happens the rest of the pipeline is still perfectly demonstrable given the
file, so this keeps a failed download from being a dead end.

The file is copied into the working directory rather than read in place, so a
run never mutates or depends on the user's original.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from dialogue_extractor.domain.models import RawMedia
from dialogue_extractor.infra.logging import get_logger, stage
from dialogue_extractor.ingestion.url_validator import resolve_local_path
from dialogue_extractor.interfaces.ports import VideoSource

log = get_logger(__name__)


class LocalFileSource(VideoSource):
    """Accepts a filesystem path or a file:// URL."""

    def supports(self, uri: str) -> bool:
        candidate = uri.strip()
        if candidate.lower().startswith("file://"):
            return True
        if re.match(r"^[a-zA-Z]:[\/]", candidate):
            return True
        return not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", candidate)

    def acquire(self, uri: str, dest_dir: Path) -> RawMedia:
        source = resolve_local_path(uri)
        dest_dir.mkdir(parents=True, exist_ok=True)
        destination = dest_dir / f"source{source.suffix or '.mp4'}"

        with stage("ingestion", source=source.name, mode="local"):
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            else:
                destination = source

        log.info("ingest.local", path=str(destination), size_bytes=destination.stat().st_size)
        return RawMedia(path=destination, source_uri=str(source), title=source.stem)
