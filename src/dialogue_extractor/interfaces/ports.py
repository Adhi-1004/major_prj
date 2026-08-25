"""Abstract ports -- the dependency-inversion boundary.

`pipeline/orchestrator.py` depends only on these, never on a concrete adapter.
Each port is deliberately narrow (one job, few methods) so an implementation is
cheap to write and cheap to fake in a test.

Adding a new backend -- a different ASR, a local active-speaker detector, an OCR
cross-check -- means writing a new class here-shaped, not editing the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from dialogue_extractor.domain.models import (
    ASDResult,
    MediaManifest,
    RawMedia,
    SpeakerTurn,
    Transcript,
)


class VideoSource(ABC):
    """Acquires a local media file from a URI."""

    @abstractmethod
    def supports(self, uri: str) -> bool:
        """True if this source can handle `uri`."""

    @abstractmethod
    def acquire(self, uri: str, dest_dir: Path) -> RawMedia:
        """Fetch the media. Raises IngestionError on failure."""


class Transcriber(ABC):
    """Speech to word-level timed text."""

    @abstractmethod
    def transcribe(self, audio_path: Path) -> Transcript:
        """Raises TranscriptionError on failure."""


class Diarizer(ABC):
    """Audio to speaker turns."""

    @abstractmethod
    def diarize(self, audio_path: Path) -> tuple[SpeakerTurn, ...]:
        """Raises DiarizationError on failure."""


class ASDProvider(ABC):
    """Verifies whether a visible face is producing the audio.

    Every implementation returns the same ASDResult, so `frame_selector` cannot
    tell which one answered -- that substitutability is what lets the hosted
    NVIDIA backend and a local model be swapped without touching the pipeline.
    """

    name: str = "abstract"

    @abstractmethod
    def detect(
        self,
        clip_path: Path,
        audio_path: Path | None,
        diarization: dict[str, Any],
        clip_start_frame: int,
        fps: float,
    ) -> ASDResult:
        """Raises ASDError (or a subclass) on failure."""

    @abstractmethod
    def health_check(self) -> None:
        """Raises if the provider is not usable (bad credentials, missing weights)."""


class FrameExtractor(ABC):
    """Pulls one exact frame out of a video file."""

    @abstractmethod
    def extract(self, video_path: Path, frame_number: int, dest: Path) -> Path:
        """Raises FrameExtractionError if the frame cannot be decoded."""


class SceneDetector(ABC):
    """Finds shot-cut boundaries, so a speaking run never spans a cut."""

    @abstractmethod
    def detect_cuts(self, video_path: Path, start: float, end: float) -> tuple[float, ...]:
        """Cut timestamps in seconds, within [start, end]."""


class MediaNormalizer(ABC):
    """Standardizes arbitrary input to the one format the pipeline assumes."""

    @abstractmethod
    def normalize(self, raw: RawMedia, work_dir: Path) -> MediaManifest:
        """Raises NormalizationError on failure."""


@runtime_checkable
class Cache(Protocol):
    """Content-addressed disk cache.

    A Protocol rather than an ABC: callers only need the shape, and a plain dict
    or a no-op stands in fine for a test.
    """

    def get(self, key: str) -> Any | None: ...

    def put(self, key: str, value: Any) -> None: ...

    def path_for(self, key: str, suffix: str) -> Path: ...
