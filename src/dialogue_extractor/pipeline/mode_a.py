"""Mode A: find the first on-screen occurrence of a target phrase.

The early-exit strategy the design document is built around. Candidates are
tried in chronological order and the first one that verifies wins, so a
feature-length film costs one detection request rather than hundreds -- measured
at 0.8% of the footage on the sample video.

Ordering is what makes the answer correct, not just cheap. The brief asks for
the frame where the dialogue *first* appears on screen, and a phrase said twice
has two transcript matches. Verifying in time order and stopping means the
earliest verified occurrence is the one reported, rather than whichever happened
to be checked first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dialogue_extractor.audio.diarization_export import (
    SpeakerCodec,
    to_nvidia_diarization,
)
from dialogue_extractor.config.constants import MIN_SPEAKING_MS, ONSET_TOLERANCE_SECONDS
from dialogue_extractor.domain.errors import ASDError, StageError
from dialogue_extractor.domain.models import (
    Candidate,
    ExtractionResult,
    MediaManifest,
    Word,
)
from dialogue_extractor.domain.timing import first_frame_at_or_after
from dialogue_extractor.infra.logging import get_logger
from dialogue_extractor.interfaces.ports import ASDProvider, FrameExtractor
from dialogue_extractor.pipeline import result_builder
from dialogue_extractor.vision.clip_builder import Clip, build_clip
from dialogue_extractor.vision.frame_selector import SelectionOutcome, select_frame
from dialogue_extractor.vision.pyav_extractor import frame_filename
from dialogue_extractor.vision.scene_detector import cut_frames, detect_cuts

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CandidateAttempt:
    """What happened for one candidate, kept whether or not it verified.

    Every attempt is recorded so a final "not on screen" can explain what was
    tried, rather than reporting only the last failure.
    """

    candidate: Candidate
    outcome: SelectionOutcome | None
    error: StageError | None = None

    @property
    def verified(self) -> bool:
        return self.outcome is not None and self.outcome.found


def onset_frame_for(
    candidate: Candidate,
    clip: Clip,
    *,
    tolerance: float = ONSET_TOLERANCE_SECONDS,
) -> int | None:
    """Clip-relative frame at which the candidate's line begins.

    The first matched word's ASR start, moved earlier by `tolerance` to absorb
    timestamp jitter, never earlier than the clip itself. None when the
    candidate carries no words to anchor on.
    """
    if not candidate.words:
        return None
    onset_seconds = max(clip.start, candidate.words[0].start - tolerance)
    global_onset = first_frame_at_or_after(onset_seconds, clip.fps)
    return max(0, global_onset - clip.clip_start_frame)


def verify_candidate(
    candidate: Candidate,
    manifest: MediaManifest,
    words: tuple[Word, ...],
    provider: ASDProvider,
    work_dir: Path,
    *,
    name: str,
    min_speaking_ms: float = MIN_SPEAKING_MS,
    detect_scene_cuts: bool = True,
    onset_constraint: bool = True,
    speaker_check: bool = True,
) -> SelectionOutcome:
    """Send one candidate for detection and decide whether it verifies.

    `speaker_check=False` is ablation A1 (accept any speaking face); it exists
    for evaluation and is never turned off in normal use.
    """
    clip = build_clip(candidate, manifest, work_dir, name=name)

    window = tuple(w for w in words if candidate.start <= w.start <= candidate.end)
    codec = SpeakerCodec.from_words(window)
    diarization = to_nvidia_diarization(window, clip_start=candidate.start, codec=codec)

    result = provider.detect(
        clip.path, clip.audio_path, diarization, clip.clip_start_frame, clip.fps
    )

    cuts: frozenset[int] = frozenset()
    if detect_scene_cuts:
        cuts = cut_frames(detect_cuts(str(clip.path)), clip.fps)

    return select_frame(
        result,
        codec.encode(candidate.speaker_id) if candidate.speaker_id else None,
        min_speaking_ms=min_speaking_ms,
        cut_frames=cuts,
        onset_frame=onset_frame_for(candidate, clip) if onset_constraint else None,
        match_speaker=speaker_check,
    )


def run(
    candidates: tuple[Candidate, ...],
    manifest: MediaManifest,
    words: tuple[Word, ...],
    provider: ASDProvider,
    extractor: FrameExtractor,
    *,
    target: str,
    work_dir: Path,
    output_dir: Path,
    min_speaking_ms: float = MIN_SPEAKING_MS,
    detect_scene_cuts: bool = True,
    onset_constraint: bool = True,
    speaker_check: bool = True,
) -> tuple[ExtractionResult, tuple[CandidateAttempt, ...]]:
    """Verify candidates in order, stopping at the first that is on screen.

    Returns the result and every attempt made, so the caller can report what was
    tried even when nothing verified.
    """
    if not candidates:
        return result_builder.build_not_found(target), ()

    attempts: list[CandidateAttempt] = []

    for index, candidate in enumerate(candidates):
        log.info(
            "mode_a.verifying",
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
                name=f"cand{index}",
                min_speaking_ms=min_speaking_ms,
                detect_scene_cuts=detect_scene_cuts,
                onset_constraint=onset_constraint,
                speaker_check=speaker_check,
            )
        except ASDError as exc:
            # One candidate failing does not doom the rest -- a later occurrence
            # may verify. The failure is recorded rather than swallowed.
            log.warning(
                "mode_a.candidate_failed",
                candidate=index + 1,
                reason=exc.reason,
                detail=exc.detail[:200],
            )
            attempts.append(CandidateAttempt(candidate, outcome=None, error=exc))
            continue

        attempts.append(CandidateAttempt(candidate, outcome=outcome))

        if outcome.found and outcome.hit is not None:
            image_name = frame_filename(outcome.hit.global_frame)
            output_dir.mkdir(parents=True, exist_ok=True)
            extractor.extract(
                manifest.video_path, outcome.hit.global_frame, output_dir / image_name
            )

            log.info(
                "mode_a.verified",
                candidate=index + 1,
                frame=outcome.hit.global_frame,
                run_length=outcome.hit.run_length,
            )
            return (
                result_builder.build_success(candidate, outcome, image_name=image_name),
                tuple(attempts),
            )

    return _best_failure(attempts, target), tuple(attempts)


def _best_failure(
    attempts: list[CandidateAttempt], target: str
) -> ExtractionResult:
    """Report the most informative failure among the attempts.

    An attempt where the speaker was visible but too briefly says more than one
    where no face appeared at all, so it is preferred -- reporting whichever
    happened to be last would discard the better diagnosis.
    """
    if not attempts:
        return result_builder.build_not_found(target)

    with_outcome = [a for a in attempts if a.outcome is not None]
    if not with_outcome:
        error = attempts[0].error
        if error is not None:
            raise error
        return result_builder.build_not_found(target)

    def informativeness(attempt: CandidateAttempt) -> tuple[int, int]:
        outcome = attempt.outcome
        assert outcome is not None
        return (outcome.frames_with_target_speaking, outcome.frames_with_any_speaker)

    best = max(with_outcome, key=informativeness)
    assert best.outcome is not None
    return result_builder.build_not_on_screen(best.candidate, best.outcome)
