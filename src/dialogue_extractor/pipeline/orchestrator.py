"""The wired pipeline, and the boundary that keeps failures structured.

Every stage so far has been exercised through its own `tools/*_check.py` script.
Those were built to be run by hand, one stage at a time, each re-reading the
previous stage's JSON from disk. This module is the single path that runs all of
them in memory, and it is what both the CLI and the HTTP API call -- so there is
one pipeline rather than two implementations that drift apart.

**The error boundary is the point of this module.** The design document promises
that any stage failure surfaces as structured JSON naming the stage and the
reason, never an unhandled crash. That promise is only worth something if it
also covers the failures nobody anticipated, so this catches `Exception`, not
just `StageError`: an unforeseen `KeyError` deep inside a decoder is exactly the
case where a caller most needs to be told which stage was running.

Only ports are imported here. The concrete transcriber, diarizer and detection
provider arrive as factories from the composition root, which keeps torch and
pyannote -- several seconds and a few gigabytes of import -- off the path of an
API process that may only ever answer /health.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dialogue_extractor.audio.aligner import assign_speakers, build_segments
from dialogue_extractor.config import constants
from dialogue_extractor.domain.enums import Mode, StageName, Status
from dialogue_extractor.domain.errors import StageError
from dialogue_extractor.domain.models import MediaManifest, SpeechSegment, Word
from dialogue_extractor.infra.logging import get_logger
from dialogue_extractor.interfaces.ports import (
    ASDProvider,
    Diarizer,
    FrameExtractor,
    MediaNormalizer,
    Transcriber,
    VideoSource,
)
from dialogue_extractor.matching.candidate_builder import (
    candidates_from_matches,
    candidates_from_segments,
)
from dialogue_extractor.matching.fuzzy_matcher import find_matches
from dialogue_extractor.pipeline import mode_a, mode_b
from dialogue_extractor.pipeline.mode_b import Budget

log = get_logger(__name__)

ProgressFn = Callable[[StageName, str], None]
"""Called as each stage starts. The API turns these into job status updates."""


@dataclass(frozen=True, slots=True)
class PipelineRequest:
    """One extraction job.

    `dialogue` is what selects the mode, rather than a separate flag that could
    contradict it: a target phrase means Mode A, its absence means discovery.
    """

    url: str
    dialogue: str | None = None
    max_candidates: int = constants.DEFAULT_MAX_CANDIDATES
    time_budget_seconds: float = constants.DEFAULT_TIME_BUDGET_SECONDS
    match_score_floor: float = constants.MATCH_SCORE_FLOOR
    window_pad_seconds: float = constants.WINDOW_PAD_SECONDS
    min_speaking_ms: float = constants.MIN_SPEAKING_MS
    detect_scene_cuts: bool = True

    @property
    def mode(self) -> Mode:
        return Mode.TARGET if self.dialogue else Mode.DISCOVERY


@dataclass(frozen=True, slots=True)
class Stages:
    """The concrete implementations, supplied by the composition root.

    Factories rather than instances for the three expensive ones. Building a
    Whisper model to serve a request that fails URL validation would be pure
    waste, and in the API process it would happen at import time.
    """

    source_for: Callable[[str], VideoSource]
    normalizer: MediaNormalizer
    transcriber: Callable[[], Transcriber]
    diarizer: Callable[[], Diarizer]
    asd: Callable[[], ASDProvider]
    extractor: FrameExtractor


@dataclass
class _StageTracker:
    """Remembers which stage is running so an unexpected failure can name it."""

    current: StageName = StageName.CONFIG
    on_progress: ProgressFn | None = None

    def enter(self, stage: StageName, message: str) -> None:
        self.current = stage
        log.info("pipeline.stage", stage=stage.value, message=message)
        if self.on_progress is not None:
            self.on_progress(stage, message)


@dataclass(frozen=True, slots=True)
class PipelineOutcome:
    """The result document, plus what it took to produce it."""

    payload: dict[str, Any]
    manifest: MediaManifest | None = None
    words: tuple[Word, ...] = ()
    stage_reached: StageName = StageName.CONFIG
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return str(self.payload.get("status", Status.ERROR.value))

    @property
    def ok(self) -> bool:
        """Whether a frame was actually produced.

        `ambiguous` counts -- a frame was found and the uncertainty about it is
        reported alongside. `not_on_screen` and `not_found` do not: they are
        correct answers, but not the answer the caller asked for.
        """
        return self.status in {Status.SUCCESS.value, Status.AMBIGUOUS.value}


class Orchestrator:
    """Runs the whole pipeline for one request."""

    def __init__(
        self,
        stages: Stages,
        *,
        work_dir: Path,
        output_dir: Path,
    ) -> None:
        self._stages = stages
        self._work_dir = Path(work_dir)
        self._output_dir = Path(output_dir)

    def run(
        self,
        request: PipelineRequest,
        *,
        on_progress: ProgressFn | None = None,
    ) -> PipelineOutcome:
        """Run the pipeline, returning a result document either way.

        Never raises for a pipeline failure. A caller that wants to distinguish
        outcomes reads `status`; one that wants an exit code reads `ok`.
        """
        tracker = _StageTracker(on_progress=on_progress)
        try:
            return self._run(request, tracker)
        except StageError as exc:
            log.warning("pipeline.failed", stage=exc.stage.value, reason=exc.reason)
            return PipelineOutcome(payload=exc.to_dict(), stage_reached=exc.stage)
        except Exception as exc:
            # An unforeseen failure is the case where naming the stage matters
            # most. The traceback goes to the log and never into the response:
            # it can carry absolute paths, and from a misbehaving dependency,
            # the arguments it was called with.
            log.error(
                "pipeline.unexpected_error",
                stage=tracker.current.value,
                error=type(exc).__name__,
                traceback=traceback.format_exc(),
            )
            return PipelineOutcome(
                payload={
                    "status": Status.ERROR.value,
                    "stage": tracker.current.value,
                    "reason": "unexpected_error",
                    "detail": (
                        f"An unexpected {type(exc).__name__} occurred during "
                        f"{tracker.current.value}."
                    ),
                    "remediation": "Re-run with --log-level DEBUG for the traceback.",
                },
                stage_reached=tracker.current,
            )

    # --- the pipeline itself -------------------------------------------------

    def _run(self, request: PipelineRequest, tracker: _StageTracker) -> PipelineOutcome:
        stages = self._stages

        tracker.enter(StageName.INGESTION, f"acquiring {request.url}")
        source = stages.source_for(request.url)
        raw = source.acquire(request.url, self._work_dir)

        tracker.enter(StageName.NORMALIZATION, "normalizing to CFR H.264 and 16 kHz WAV")
        manifest = stages.normalizer.normalize(raw, self._work_dir)

        tracker.enter(StageName.TRANSCRIPTION, "transcribing")
        transcript = stages.transcriber().transcribe(manifest.audio_path)

        tracker.enter(StageName.DIARIZATION, "identifying speakers")
        turns = stages.diarizer().diarize(manifest.audio_path)

        tracker.enter(StageName.ALIGNMENT, "attributing words to speakers")
        words = assign_speakers(transcript.words, turns)
        segments = build_segments(words)

        diagnostics: dict[str, Any] = {
            "mode": request.mode.value,
            "fps": round(manifest.fps, 6),
            "duration": round(manifest.duration, 3),
            "frame_count": manifest.frame_count,
            "resolution": manifest.resolution,
            "words": len(words),
            "segments": len(segments),
            "speakers": len({t.speaker_id for t in turns}),
        }

        if request.mode is Mode.TARGET:
            payload = self._run_target(request, manifest, words, tracker, diagnostics)
        else:
            payload = self._run_discovery(
                request, manifest, words, segments, tracker, diagnostics
            )

        return PipelineOutcome(
            payload=payload,
            manifest=manifest,
            words=words,
            stage_reached=StageName.FRAME_EXTRACTION,
            diagnostics=diagnostics,
        )

    def _run_target(
        self,
        request: PipelineRequest,
        manifest: MediaManifest,
        words: tuple[Word, ...],
        tracker: _StageTracker,
        diagnostics: dict[str, Any],
    ) -> dict[str, Any]:
        target = request.dialogue or ""

        tracker.enter(StageName.MATCHING, f"searching for {target!r}")
        matches = find_matches(words, target, score_floor=request.match_score_floor)
        candidates = candidates_from_matches(
            matches, manifest.duration, pad=request.window_pad_seconds
        )
        diagnostics["matches"] = len(matches)
        diagnostics["candidates"] = len(candidates)

        tracker.enter(
            StageName.ACTIVE_SPEAKER_DETECTION,
            f"verifying {len(candidates)} candidate(s), earliest first",
        )
        result, attempts = mode_a.run(
            candidates,
            manifest,
            words,
            self._stages.asd(),
            self._stages.extractor,
            target=target,
            work_dir=self._work_dir / "clips",
            output_dir=self._output_dir,
            min_speaking_ms=request.min_speaking_ms,
            detect_scene_cuts=request.detect_scene_cuts,
        )
        diagnostics["candidates_verified"] = len(attempts)

        payload = result.to_dict()
        payload["mode"] = Mode.TARGET.value
        return payload

    def _run_discovery(
        self,
        request: PipelineRequest,
        manifest: MediaManifest,
        words: tuple[Word, ...],
        segments: tuple[SpeechSegment, ...],
        tracker: _StageTracker,
        diagnostics: dict[str, Any],
    ) -> dict[str, Any]:
        candidates = candidates_from_segments(segments, manifest.duration)
        diagnostics["candidates"] = len(candidates)

        tracker.enter(
            StageName.ACTIVE_SPEAKER_DETECTION,
            f"sweeping {len(candidates)} candidate(s)",
        )
        result = mode_b.run(
            candidates,
            manifest,
            words,
            self._stages.asd(),
            self._stages.extractor,
            work_dir=self._work_dir / "clips",
            output_dir=self._output_dir / "frames",
            budget=Budget(
                max_candidates=request.max_candidates,
                time_budget_seconds=request.time_budget_seconds,
            ),
            min_speaking_ms=request.min_speaking_ms,
            detect_scene_cuts=request.detect_scene_cuts,
        )

        payload = result.to_dict()
        payload["mode"] = Mode.DISCOVERY.value
        return payload
