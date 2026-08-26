"""Mode B: find every on-screen dialogue in the video.

Where Mode A stops at the first verified occurrence, discovery checks all of
them -- which turns the cost model inside out. Mode A on the sample video needs
one detection request; discovery needs one per candidate, and a feature-length
film produces hundreds. That is why this module is mostly about budgets and
about being honest when one runs out.

Three commitments shape the output:

**A partial sweep is never presented as exhaustive.** If a budget stops the run,
`truncated` says so and the counts show how far it got. Silently returning
fewer results than exist would make "every on-screen line" a false claim.

**Uncertain results are separated, not discarded or promoted.** Anything that
verified but scored below the success band goes to `ambiguous` with its reason,
because the brief asks for uncertainty to be reported rather than guessed at.

**Lines ruled off camera are reported, not dropped.** Telling on-screen dialogue
from voice-over is the product, so the exclusions are half the answer and each
one carries its reason.

**One candidate failing does not end the run.** A network blip on chunk 12 of 37
costs chunk 12, not the other 36. Failures are collected and reported alongside
the results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from dialogue_extractor.config.constants import (
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_TIME_BUDGET_SECONDS,
    MIN_SPEAKING_MS,
)
from dialogue_extractor.domain.enums import Status
from dialogue_extractor.domain.errors import ASDError
from dialogue_extractor.domain.models import (
    Candidate,
    DiscoveryResult,
    ExtractionResult,
    MediaManifest,
    Word,
)
from dialogue_extractor.infra.logging import get_logger
from dialogue_extractor.interfaces.ports import ASDProvider, FrameExtractor
from dialogue_extractor.pipeline import result_builder
from dialogue_extractor.pipeline.mode_a import verify_candidate
from dialogue_extractor.vision.providers.nvidia_asd import (
    assert_diarization_reached_service,
)
from dialogue_extractor.vision.pyav_extractor import frame_filename

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Budget:
    """Limits on how much of the video a discovery run will examine."""

    max_candidates: int = DEFAULT_MAX_CANDIDATES
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS

    @property
    def unlimited(self) -> bool:
        return self.max_candidates <= 0 and self.time_budget_seconds <= 0


class BudgetTracker:
    """Decides when to stop, and remembers why.

    Separate from the loop so the stopping rule can be tested without a
    detection service, and so the reason survives into the reported output.
    """

    def __init__(self, budget: Budget, total: int) -> None:
        self._budget = budget
        self._total = total
        self._started = time.monotonic()
        self.processed = 0
        self.stopped_reason: str | None = None

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def may_continue(self) -> bool:
        """Whether there is room for one more candidate."""
        if self._budget.max_candidates > 0 and self.processed >= self._budget.max_candidates:
            self.stopped_reason = (
                f"candidate budget reached ({self._budget.max_candidates})"
            )
            return False
        if (
            self._budget.time_budget_seconds > 0
            and self.elapsed >= self._budget.time_budget_seconds
        ):
            self.stopped_reason = (
                f"time budget reached ({self._budget.time_budget_seconds:.0f}s)"
            )
            return False
        return True

    def record(self) -> None:
        self.processed += 1

    @property
    def truncated(self) -> bool:
        return self.processed < self._total


def run(
    candidates: tuple[Candidate, ...],
    manifest: MediaManifest,
    words: tuple[Word, ...],
    provider: ASDProvider,
    extractor: FrameExtractor,
    *,
    work_dir: Path,
    output_dir: Path,
    budget: Budget | None = None,
    min_speaking_ms: float = MIN_SPEAKING_MS,
    detect_scene_cuts: bool = True,
    extract_frames: bool = True,
) -> DiscoveryResult:
    """Verify every candidate, bucketing the results by confidence."""
    budget = budget or Budget()
    tracker = BudgetTracker(budget, len(candidates))

    verified: list[ExtractionResult] = []
    uncertain: list[ExtractionResult] = []
    off_screen: list[ExtractionResult] = []
    failures: list[dict[str, object]] = []
    unassociated = 0

    for index, candidate in enumerate(candidates):
        if not tracker.may_continue():
            log.info(
                "mode_b.budget_reached",
                reason=tracker.stopped_reason,
                processed=tracker.processed,
                total=len(candidates),
            )
            break

        log.info(
            "mode_b.verifying",
            candidate=index + 1,
            of=len(candidates),
            window=f"{candidate.start:.2f}-{candidate.end:.2f}",
            speaker=candidate.speaker_id,
        )

        try:
            outcome = verify_candidate(
                candidate,
                manifest,
                words,
                provider,
                work_dir,
                name=f"chunk{index:03d}",
                min_speaking_ms=min_speaking_ms,
                detect_scene_cuts=detect_scene_cuts,
            )
        except ASDError as exc:
            log.warning(
                "mode_b.candidate_failed",
                candidate=index + 1,
                reason=exc.reason,
                detail=exc.detail[:200],
            )
            failures.append(
                {
                    "candidate": index,
                    "window": [candidate.start, candidate.end],
                    "reason": exc.reason,
                    "detail": exc.detail[:300],
                }
            )
            tracker.record()
            continue

        tracker.record()

        # A window where no face matched any diarized speaker is ordinary --
        # the speaker was off camera. Every window being like that is not.
        if outcome.frames_with_face and not outcome.speaker_codes_seen:
            unassociated += 1

        if not outcome.found:
            # Not an error, and reported rather than dropped: "this line is
            # spoken off camera" is a finding, not an absence of one. On the
            # sample video these are the film's internal-monologue scenes, which
            # is precisely the judgement a caller would want to check.
            off_screen.append(
                result_builder.build_not_on_screen(
                    candidate, outcome, with_timestamp=True
                )
            )
            log.debug(
                "mode_b.not_on_screen",
                candidate=index + 1,
                reason=outcome.reason.value if outcome.reason else None,
            )
            continue

        image_name: str | None = None
        if extract_frames and outcome.hit is not None:
            image_name = frame_filename(outcome.hit.global_frame)
            output_dir.mkdir(parents=True, exist_ok=True)
            extractor.extract(
                manifest.video_path, outcome.hit.global_frame, output_dir / image_name
            )

        result = result_builder.build_success(candidate, outcome, image_name=image_name)

        if result.status is Status.SUCCESS:
            verified.append(result)
        else:
            uncertain.append(result)

        log.info(
            "mode_b.found",
            candidate=index + 1,
            frame=result.frame_number,
            status=result.status.value,
            confidence=round(result.confidence or 0.0, 3),
        )

    # Checked once, over the whole run: the failure this guards against is
    # systematic by nature, so a single window cannot distinguish it from a
    # perfectly ordinary off-camera line.
    assert_diarization_reached_service(unassociated, tracker.processed)

    return DiscoveryResult(
        results=tuple(verified),
        ambiguous=tuple(uncertain),
        not_on_screen=tuple(off_screen),
        truncated=tracker.truncated,
        candidates_processed=tracker.processed,
        candidates_total=len(candidates),
        errors=tuple(failures),
    )
