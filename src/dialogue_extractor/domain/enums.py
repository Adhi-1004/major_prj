"""Enumerations shared across the pipeline.

Pure: no I/O, no third-party imports.
"""

from __future__ import annotations

from enum import StrEnum


class Mode(StrEnum):
    """Which of the two pipeline strategies is running."""

    TARGET = "target"      # Mode A -- a target phrase was supplied
    DISCOVERY = "discovery"  # Mode B -- enumerate every on-screen dialogue


class Status(StrEnum):
    """Terminal status of a single extraction attempt."""

    SUCCESS = "success"
    AMBIGUOUS = "ambiguous"
    NOT_ON_SCREEN = "not_on_screen"
    NOT_FOUND = "not_found"
    ERROR = "error"


class StageName(StrEnum):
    """Pipeline stage, used to attribute failures precisely.

    Every StageError carries one of these so a structured error can say *where*
    it broke, per the "graceful degradation" NFR in docs/approach.md.
    """

    INGESTION = "ingestion"
    VALIDATION = "validation"
    NORMALIZATION = "normalization"
    TRANSCRIPTION = "transcription"
    DIARIZATION = "diarization"
    ALIGNMENT = "alignment"
    MATCHING = "matching"
    CLIP_BUILD = "clip_build"
    ACTIVE_SPEAKER_DETECTION = "active_speaker_detection"
    FRAME_SELECTION = "frame_selection"
    FRAME_EXTRACTION = "frame_extraction"
    CONFIG = "config"


class AmbiguityReason(StrEnum):
    """Machine-readable reason a result landed in the ambiguous band.

    The brief asks explicitly that uncertain results be *reported* rather than
    guessed at; a bare "ambiguous" with no reason would not satisfy that.
    """

    LOW_CONFIDENCE = "low_confidence"
    """Composite score fell in the ambiguous band."""

    LOW_FACE_CONFIDENCE = "low_face_confidence"
    """A face was tracked, but the detector was unsure it was a face."""

    WEAK_TEXT_MATCH = "weak_text_match"
    """Fuzzy match cleared the floor but not by much -- possible ASR drift."""

    SHORT_SPEAKING_RUN = "short_speaking_run"
    """Speaking run met the minimum but showed little temporal support."""

    MULTIPLE_SIMULTANEOUS_SPEAKERS = "multiple_simultaneous_speakers"
    """More than one visible face was speaking; could not disambiguate."""

    MAX_SPEAKERS_REACHED = "max_speakers_reached"
    """Face count hit the service's per-frame speaker cap; faces may be missing."""

    SPEAKER_MISMATCH = "speaker_mismatch"
    """A face was speaking, but not the diarized speaker of the matched words."""

    NO_FACE_DETECTED = "no_face_detected"
    """No face was found anywhere in the candidate window."""
