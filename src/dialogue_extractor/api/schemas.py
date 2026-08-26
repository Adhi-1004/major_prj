"""Request and response models for the HTTP layer.

Validation lives here so a bad request is refused at the edge with a 422 that
names the field, rather than surfacing three stages later as a download error.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from dialogue_extractor.config import constants
from dialogue_extractor.domain.errors import InvalidURLError
from dialogue_extractor.ingestion.url_validator import classify_source
from dialogue_extractor.pipeline.orchestrator import PipelineRequest


class ExtractRequest(BaseModel):
    """A submitted extraction.

    Omitting `dialogue` selects discovery mode, exactly as omitting `--dialogue`
    does on the command line -- the two entry points must not disagree about
    what the same input means.
    """

    url: str = Field(
        ...,
        min_length=1,
        description="Public video URL, or a path the server can read.",
        examples=["https://www.youtube.com/watch?v=..."],
    )
    dialogue: str | None = Field(
        default=None,
        description="The line to find. Omit to sweep for every on-screen line.",
        examples=["my mind rebels at stagnation"],
    )
    max_candidates: int = Field(
        default=constants.DEFAULT_MAX_CANDIDATES,
        ge=0,
        description="Discovery only. 0 means no limit.",
    )
    time_budget_seconds: float = Field(
        default=constants.DEFAULT_TIME_BUDGET_SECONDS,
        ge=0,
        description="Discovery only. 0 means no limit.",
    )
    match_score_floor: float = Field(
        default=constants.MATCH_SCORE_FLOOR, ge=0, le=100
    )
    window_pad_seconds: float = Field(
        default=constants.WINDOW_PAD_SECONDS, ge=0, le=30
    )
    min_speaking_ms: float = Field(
        default=constants.MIN_SPEAKING_MS, gt=0, le=5000
    )
    detect_scene_cuts: bool = True

    @field_validator("url")
    @classmethod
    def _url_is_usable(cls, value: str) -> str:
        """Reject at the edge what ingestion would reject anyway.

        `classify_source` already refuses empty input and schemes we will not
        fetch. Reusing it here means the API and the CLI cannot disagree about
        which URLs are acceptable, and the caller learns immediately rather than
        after a job id and a poll.

        Its `InvalidURLError` is re-raised as a `ValueError` so pydantic reports
        it as a 422 alongside every other field problem. Letting the StageError
        escape produced a 400 for a bad scheme and a 422 for an empty string --
        two codes for one class of mistake, which a client would have to special
        case. The remediation text is carried across so nothing is lost.
        """
        try:
            classify_source(value)
        except InvalidURLError as exc:
            message = exc.detail
            if exc.remediation:
                message = f"{message} {exc.remediation}"
            raise ValueError(message) from exc
        return value.strip()

    @field_validator("dialogue")
    @classmethod
    def _blank_dialogue_means_discovery(cls, value: str | None) -> str | None:
        """A whitespace-only phrase is not a target.

        Left as-is it would select target mode and then match nothing, reporting
        `not_found` for a search the caller never meant to run.
        """
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    def to_pipeline_request(self) -> PipelineRequest:
        return PipelineRequest(
            url=self.url,
            dialogue=self.dialogue,
            max_candidates=self.max_candidates,
            time_budget_seconds=self.time_budget_seconds,
            match_score_floor=self.match_score_floor,
            window_pad_seconds=self.window_pad_seconds,
            min_speaking_ms=self.min_speaking_ms,
            detect_scene_cuts=self.detect_scene_cuts,
        )


class JobAccepted(BaseModel):
    """What `POST /extract` returns: an id and where to poll."""

    job_id: str
    state: str
    mode: str
    poll: str = Field(description="Path to poll for this job's progress.")


class HealthResponse(BaseModel):
    """Liveness plus the facts that most often explain a failed run.

    Reporting whether credentials are *present* is the point -- a missing key is
    the most common setup mistake, and finding out at /health beats finding out
    after a five-minute transcription. The values themselves are never included.
    """

    status: str
    version: str
    ffmpeg: str | None = None
    providers: list[str]
    credentials: dict[str, bool]
    jobs: dict[str, int]


class ErrorResponse(BaseModel):
    """The documented failure shape, matching the CLI's output byte for byte."""

    status: str
    stage: str
    reason: str
    detail: str
    remediation: str | None = None

    @classmethod
    def of(cls, payload: dict[str, Any]) -> ErrorResponse:
        return cls(**payload)
