"""Structured error hierarchy.

Every failure in the pipeline surfaces as a StageError so the orchestrator can
turn it into a JSON document naming the stage and the reason, rather than
letting a traceback escape. See the "Input and infrastructure" edge-case table
in docs/approach.md: "Any stage failure ... never an unhandled crash."

Pure: no I/O, no third-party imports.
"""

from __future__ import annotations

from typing import Any

from dialogue_extractor.domain.enums import StageName, Status


class StageError(Exception):
    """Base for every recoverable, attributable pipeline failure.

    Attributes:
        stage: Which pipeline stage failed.
        reason: Short machine-ish reason slug, e.g. "no_audio_stream".
        detail: Human-readable explanation shown to the user.
        remediation: What the user can do about it, when there is something.
    """

    stage: StageName = StageName.CONFIG

    def __init__(
        self,
        reason: str,
        detail: str,
        *,
        remediation: str | None = None,
        stage: StageName | None = None,
    ) -> None:
        self.reason = reason
        self.detail = detail
        self.remediation = remediation
        if stage is not None:
            self.stage = stage
        super().__init__(f"[{self.stage}/{reason}] {detail}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the documented error shape."""
        payload: dict[str, Any] = {
            "status": Status.ERROR.value,
            "stage": self.stage.value,
            "reason": self.reason,
            "detail": self.detail,
        }
        if self.remediation:
            payload["remediation"] = self.remediation
        return payload


# --- Ingestion ---------------------------------------------------------------


class IngestionError(StageError):
    stage = StageName.INGESTION


class InvalidURLError(IngestionError):
    """URL was malformed, or used a scheme we refuse to fetch."""


class DownloadError(IngestionError):
    """Acquisition failed after retries (network, 404, site change)."""


class GeoRestrictedError(IngestionError):
    """Source refused the request on geographic grounds."""


class DRMProtectedError(IngestionError):
    """Source is DRM-protected and cannot be legally decoded."""


# --- Validation / normalization ---------------------------------------------


class ValidationError(StageError):
    stage = StageName.VALIDATION


class NoAudioStreamError(ValidationError):
    """Container has no audio track; every speech stage would fail downstream."""


class NoVideoStreamError(ValidationError):
    """Container has no video track; there are no frames to return."""


class CorruptMediaError(ValidationError):
    """ffprobe could not parse the file."""


class NormalizationError(StageError):
    stage = StageName.NORMALIZATION


class FFmpegError(NormalizationError):
    """An ffmpeg/ffprobe invocation failed; detail carries the stderr tail."""


class FFmpegNotFoundError(NormalizationError):
    """ffmpeg or ffprobe is not on PATH, or is too old to use."""


# --- Audio -------------------------------------------------------------------


class TranscriptionError(StageError):
    stage = StageName.TRANSCRIPTION


class DiarizationError(StageError):
    stage = StageName.DIARIZATION


class GatedModelError(DiarizationError):
    """The pyannote pipeline is gated and the token lacks accepted terms."""


# --- Matching ----------------------------------------------------------------


class MatchingError(StageError):
    stage = StageName.MATCHING


class DialogueNotFoundError(MatchingError):
    """Target phrase does not appear in the transcript above the match floor."""


# --- Vision ------------------------------------------------------------------


class ClipBuildError(StageError):
    stage = StageName.CLIP_BUILD


class ASDError(StageError):
    stage = StageName.ACTIVE_SPEAKER_DETECTION


class ASDAuthError(ASDError):
    """Credentials rejected by the endpoint."""


class ASDQuotaError(ASDError):
    """Rate limited, out of credits, or payload too large."""


class ASDTimeoutError(ASDError):
    """The endpoint did not respond within the deadline."""


class ASDUnavailableError(ASDError):
    """The endpoint could not be reached at all."""


class MalformedDiarizationError(ASDError):
    """The service rejected our diarization payload."""


class DiarizationDroppedError(ASDError):
    """Every returned face had a null diarized_speaker_id.

    In streaming mode the service treats a session with no diarization at the
    first inference request as having none at all, and returns *confidently
    wrong* results rather than an error. Detecting that and raising is the only
    way to stop a silent wrong answer -- see G16 in the implementation plan.
    """


class FrameExtractionError(StageError):
    stage = StageName.FRAME_EXTRACTION


class ConfigError(StageError):
    stage = StageName.CONFIG


class MissingCredentialError(ConfigError):
    """A required API key or token was not configured."""
