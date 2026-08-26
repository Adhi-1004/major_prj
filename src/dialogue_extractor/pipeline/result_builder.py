"""Assembling the final result.

PURE: no I/O, no third-party imports.

The brief asks for a timestamp, a frame number, the extracted text, and the
frame image, plus a confidence value and an honest account of uncertainty. This
turns a verified frame into exactly that, and turns each way of failing into a
result that says which way it failed.
"""

from __future__ import annotations

from dialogue_extractor.domain.enums import AmbiguityReason, Status
from dialogue_extractor.domain.models import Candidate, ExtractionResult
from dialogue_extractor.pipeline import confidence as confidence_module
from dialogue_extractor.vision.frame_selector import SelectionOutcome


def build_success(
    candidate: Candidate,
    outcome: SelectionOutcome,
    *,
    image_name: str | None = None,
) -> ExtractionResult:
    """Build the result for a window where the speaker was found on screen.

    The status is not automatically `success`: a verified frame with weak
    supporting evidence is reported as ambiguous, with the term that dragged it
    down named. Reporting everything that produced a frame as a success would
    make the confidence value decorative.
    """
    hit = outcome.hit
    if hit is None:
        raise ValueError("build_success requires an outcome with a hit")

    breakdown = confidence_module.compute(
        match_score=candidate.match_score,
        asr_confidence=_mean_probability(candidate),
        face_confidence=hit.mean_face_confidence,
        run_length=hit.run_length,
        required_run=outcome.required_run,
    )

    status = confidence_module.classify(breakdown.score)

    # A caveat raised during selection -- an unverified speaker, a crowded frame,
    # simultaneous speakers -- outranks the score, because it describes something
    # the score cannot express.
    reason: AmbiguityReason | None = outcome.reason
    detail: str | None = outcome.detail or None

    if reason is not None and status is Status.SUCCESS:
        status = Status.AMBIGUOUS
    elif status is not Status.SUCCESS and reason is None:
        weakest = confidence_module.weakest_signal(breakdown)
        if weakest is not None:
            reason = weakest
            detail = _explain(weakest, breakdown.score)

    return ExtractionResult(
        status=status,
        text=candidate.text,
        timestamp=hit.timestamp,
        frame_number=hit.global_frame,
        speaker_id=candidate.speaker_id,
        confidence=breakdown.score,
        face_detection_confidence=hit.mean_face_confidence,
        image=image_name,
        ambiguity_reason=reason,
        detail=detail,
        breakdown=breakdown,
    )


def build_not_on_screen(
    candidate: Candidate,
    outcome: SelectionOutcome,
    *,
    with_timestamp: bool = False,
) -> ExtractionResult:
    """Build the result for a window where the speaker was not found on screen.

    Carries the selector's diagnosis rather than a bare status: "no face was
    visible", "someone else was speaking", and "the run was too short" are
    different answers, and collapsing them would discard the only information
    that makes the result actionable.

    `with_timestamp` records when the line was *spoken*, which is what discovery
    needs -- a list of off-screen lines is unusable without knowing where each
    one is. It stays off by default because there is no frame here, and Mode A
    reports a single answer where a timestamp with no frame beside it invites
    being read as one.
    """
    return ExtractionResult(
        status=Status.NOT_ON_SCREEN,
        text=candidate.text,
        timestamp=candidate.start if with_timestamp else None,
        speaker_id=candidate.speaker_id,
        confidence=0.0,
        ambiguity_reason=outcome.reason or AmbiguityReason.NO_FACE_DETECTED,
        detail=outcome.detail or "The speaker was not detected on screen.",
    )


def build_not_found(target: str) -> ExtractionResult:
    """Build the result for a phrase that is not in the transcript at all.

    Distinct from `not_on_screen`: one means the line was never said, the other
    that it was said off camera. Conflating them would send the user looking in
    the wrong place.
    """
    return ExtractionResult(
        status=Status.NOT_FOUND,
        text=target,
        confidence=0.0,
        detail=(
            "The phrase was not found in the transcript. It may be worded "
            "differently, spoken in another language, or absent from this video."
        ),
    )


def _mean_probability(candidate: Candidate) -> float:
    if not candidate.words:
        return 1.0
    return sum(w.probability for w in candidate.words) / len(candidate.words)


def _explain(reason: AmbiguityReason, score: float) -> str:
    messages = {
        AmbiguityReason.WEAK_TEXT_MATCH: (
            "The transcript match was loose enough that this may be a different line."
        ),
        AmbiguityReason.LOW_FACE_CONFIDENCE: (
            "The face was detected with low confidence -- the shot may be dark, "
            "blurred, or the face partly turned away."
        ),
        AmbiguityReason.SHORT_SPEAKING_RUN: (
            "The speaker was detected talking for barely longer than the minimum, "
            "so the detection has little temporal support."
        ),
        AmbiguityReason.LOW_CONFIDENCE: (
            "The combined evidence was not strong enough to report this as certain."
        ),
    }
    return f"{messages.get(reason, messages[AmbiguityReason.LOW_CONFIDENCE])} " \
           f"(confidence {score:.2f})"
