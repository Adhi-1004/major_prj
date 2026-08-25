"""Frozen data contracts passed between pipeline stages.

Every stage boundary in the pipeline is one of these types. Freezing them (and
using tuples rather than lists for sequences) means a later stage cannot quietly
mutate an earlier stage's output, which keeps each stage independently testable.

Pure: no I/O, no third-party imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dialogue_extractor.domain.enums import AmbiguityReason, Mode, Status
from dialogue_extractor.domain.timing import format_timestamp

# --- Media -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RawMedia:
    """A source file as acquired, before any normalization."""

    path: Path
    source_uri: str
    title: str | None = None


@dataclass(frozen=True, slots=True)
class MediaManifest:
    """Measured properties of the normalized media.

    `cfr_forced` is recorded rather than assumed: invariant 2
    (timestamp == frame / fps) is only sound when it is True.
    """

    video_path: Path
    audio_path: Path
    fps: float
    frame_count: int
    duration: float
    width: int
    height: int
    cfr_forced: bool
    source_was_vfr: bool
    video_sha256: str
    audio_sha256: str

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


# --- Speech ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Word:
    """One transcribed word with its time span and (once aligned) its speaker.

    `speaker_id` is None until alignment runs, and stays None for words that
    fall in a diarization gap -- those are kept, not dropped, so the transcript
    stays complete.
    """

    text: str
    start: float
    end: float
    probability: float = 1.0
    speaker_id: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    """A contiguous stretch attributed to one speaker by diarization."""

    speaker_id: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    """Consecutive words sharing a speaker, with no long internal silence."""

    words: tuple[Word, ...]
    speaker_id: str | None

    @property
    def text(self) -> str:
        """Derived from the words, never stored separately.

        Storing it would let the two drift apart after any word-level edit.
        """
        return " ".join(w.text for w in self.words).strip()

    @property
    def start(self) -> float:
        return self.words[0].start if self.words else 0.0

    @property
    def end(self) -> float:
        return self.words[-1].end if self.words else 0.0

    @property
    def mean_probability(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.probability for w in self.words) / len(self.words)


@dataclass(frozen=True, slots=True)
class Transcript:
    """Full ASR output for one audio file."""

    words: tuple[Word, ...]
    language: str
    duration: float

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words).strip()


# --- Matching ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Candidate:
    """A time window worth verifying visually.

    Produced by phrase matching (Mode A) or by segment enumeration (Mode B).
    `match_score` is None in discovery mode -- there is no target phrase to
    score against, and substituting 1.0 would inflate the confidence of every
    discovery result for a term that was never actually checked.
    """

    start: float
    end: float
    speaker_id: str | None
    text: str
    mode: Mode
    match_score: float | None = None
    words: tuple[Word, ...] = ()

    @property
    def duration(self) -> float:
        return self.end - self.start


# --- Active speaker detection ------------------------------------------------


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class Face:
    """One detected face within one frame, as reported by the ASD service.

    `diarized_speaker_id` is the service echoing back which of *our* diarized
    speakers it associated with this face. It is an **integer**, not a label:
    the proto field is an int32, and the documented JSON's string speaker ids
    are converted on the way in. Translating it back to a label is what
    SpeakerCodec is for.

    None means no association. The wire cannot express that -- an unset int32
    arrives as 0 -- so the provider normalizes 0 to None before constructing
    this. A response in which *every* face is unassociated means the diarization
    payload never reached the session; see DiarizationDroppedError.
    """

    face_id: int
    bbox: BoundingBox
    is_speaking: bool
    face_detection_confidence: float
    diarized_speaker_id: int | None = None


@dataclass(frozen=True, slots=True)
class ASDFrame:
    """Per-frame detection result. `frame_id` is clip-relative (invariant 1)."""

    frame_id: int
    faces: tuple[Face, ...] = ()

    @property
    def speaking_faces(self) -> tuple[Face, ...]:
        return tuple(f for f in self.faces if f.is_speaking)


@dataclass(frozen=True, slots=True)
class ASDResult:
    """Everything the ASD provider returned for one candidate clip."""

    frames: tuple[ASDFrame, ...]
    clip_start_frame: int
    fps: float
    provider: str

    @property
    def any_face_detected(self) -> bool:
        return any(f.faces for f in self.frames)

    @property
    def any_speaker_id_present(self) -> bool:
        """False means diarization was dropped by the service (G16)."""
        return any(
            face.diarized_speaker_id is not None for frame in self.frames for face in frame.faces
        )


@dataclass(frozen=True, slots=True)
class FrameHit:
    """The frame chosen as the answer, plus the evidence behind it."""

    global_frame: int
    timestamp: float
    face: Face
    run_length: int
    mean_face_confidence: float


# --- Results -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConfidenceBreakdown:
    """Per-term explanation of a composite score.

    Emitted alongside the score so a result is explainable rather than an
    unexplained number, per the Explainability NFR.
    """

    terms: dict[str, float]
    weights: dict[str, float]
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"score": round(self.score, 4), "terms": self.terms, "weights": self.weights}


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """One verified (or rejected) on-screen dialogue occurrence."""

    status: Status
    text: str | None = None
    timestamp: float | None = None
    frame_number: int | None = None
    speaker_id: str | None = None
    confidence: float | None = None
    face_detection_confidence: float | None = None
    image: str | None = None
    ambiguity_reason: AmbiguityReason | None = None
    detail: str | None = None
    breakdown: ConfidenceBreakdown | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": self.status.value}
        if self.timestamp is not None:
            payload["timestamp"] = format_timestamp(self.timestamp)
        if self.frame_number is not None:
            payload["frame_number"] = self.frame_number
        if self.text is not None:
            payload["text"] = self.text
        if self.speaker_id is not None:
            payload["speaker"] = self.speaker_id
        if self.confidence is not None:
            payload["confidence"] = round(self.confidence, 4)
        if self.face_detection_confidence is not None:
            payload["face_detection_confidence"] = round(self.face_detection_confidence, 4)
        if self.image is not None:
            payload["image"] = self.image
        if self.ambiguity_reason is not None:
            payload["ambiguity_reason"] = self.ambiguity_reason.value
        if self.detail is not None:
            payload["detail"] = self.detail
        if self.breakdown is not None:
            payload["confidence_breakdown"] = self.breakdown.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Mode B aggregate: confirmed results plus an explicit ambiguous bucket.

    `truncated` is reported honestly when a budget stopped the sweep early,
    rather than presenting a partial list as if it were exhaustive.
    """

    results: tuple[ExtractionResult, ...] = ()
    ambiguous: tuple[ExtractionResult, ...] = ()
    truncated: bool = False
    candidates_processed: int = 0
    candidates_total: int = 0
    errors: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": Status.SUCCESS.value if self.results else Status.NOT_FOUND.value,
            "results": [r.to_dict() for r in self.results],
            "ambiguous": [r.to_dict() for r in self.ambiguous],
            "candidates_processed": self.candidates_processed,
            "candidates_total": self.candidates_total,
        }
        if self.truncated:
            payload["truncated"] = True
        if self.errors:
            payload["errors"] = list(self.errors)
        return payload
