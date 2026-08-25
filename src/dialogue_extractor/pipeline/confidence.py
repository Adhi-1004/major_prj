"""The composite confidence score.

PURE: no I/O, no third-party imports.

The brief asks for results to carry a confidence value rather than a bare
true/false, and for uncertain ones to be reported as uncertain instead of
guessed at. An undefined number would satisfy neither, so the formula is fixed,
documented, and returns its per-term breakdown alongside the total.

Four terms, each measuring something the answer actually depends on:

    match      how closely the transcript matched the requested phrase
    asr        how confident the recognizer was about those words
    face       how confident the detector was that it saw a face
    stability  how much longer than the minimum the speaking run lasted

**Discovery mode omits `match` rather than scoring it 1.0.** There is no target
phrase in discovery mode, so the term is undefined, not perfect. Substituting
1.0 would hand every discovery result a free 0.40 for something never checked,
letting identical detection evidence clear a much lower bar in one mode than the
other. Instead the term is dropped and the remaining weights renormalize, so
both modes produce comparable numbers on the same 0-1 scale.
"""

from __future__ import annotations

from dialogue_extractor.config.constants import (
    CONFIDENCE_AMBIGUOUS_THRESHOLD,
    CONFIDENCE_SUCCESS_THRESHOLD,
    CONFIDENCE_WEIGHTS,
    LOW_FACE_CONFIDENCE_THRESHOLD,
    WEAK_MATCH_THRESHOLD,
)
from dialogue_extractor.domain.enums import AmbiguityReason, Status
from dialogue_extractor.domain.models import ConfidenceBreakdown


def stability_term(run_length: int, required_run: int) -> float:
    """How much temporal support the speaking run has, 0-1.

    Saturates at twice the required length: a run that is merely long does not
    keep getting more convincing, and letting it dominate would let a lingering
    shot outweigh a poor face detection.
    """
    if required_run <= 0:
        return 1.0
    return min(1.0, run_length / (2.0 * required_run))


def compute(
    *,
    match_score: float | None,
    asr_confidence: float,
    face_confidence: float,
    run_length: int,
    required_run: int,
    weights: dict[str, float] | None = None,
) -> ConfidenceBreakdown:
    """Combine the evidence into a single 0-1 score plus its breakdown.

    Args:
        match_score: rapidfuzz ratio 0-100, or None in discovery mode. None
            removes the term and renormalizes -- see the module docstring.
    """
    table = dict(weights or CONFIDENCE_WEIGHTS)

    terms: dict[str, float] = {
        "asr": _clamp(asr_confidence),
        "face": _clamp(face_confidence),
        "stability": _clamp(stability_term(run_length, required_run)),
    }
    if match_score is not None:
        terms["match"] = _clamp(match_score / 100.0)

    active = {name: weight for name, weight in table.items() if name in terms}
    total_weight = sum(active.values())
    if total_weight <= 0:
        return ConfidenceBreakdown(terms=terms, weights=active, score=0.0)

    score = sum(terms[name] * weight for name, weight in active.items()) / total_weight
    return ConfidenceBreakdown(
        terms={k: round(v, 4) for k, v in terms.items()},
        weights=active,
        score=round(score, 6),
    )


def classify(
    score: float,
    *,
    success_threshold: float = CONFIDENCE_SUCCESS_THRESHOLD,
    ambiguous_threshold: float = CONFIDENCE_AMBIGUOUS_THRESHOLD,
) -> Status:
    """Map a score to a reported status."""
    if score >= success_threshold:
        return Status.SUCCESS
    if score >= ambiguous_threshold:
        return Status.AMBIGUOUS
    return Status.NOT_ON_SCREEN


def weakest_signal(
    breakdown: ConfidenceBreakdown,
    *,
    face_threshold: float = LOW_FACE_CONFIDENCE_THRESHOLD,
    match_threshold: float = WEAK_MATCH_THRESHOLD,
) -> AmbiguityReason | None:
    """Name the term dragging a borderline result down.

    "Ambiguous" on its own tells the user nothing they can act on; knowing it
    was a poor face detection rather than a shaky text match does.
    """
    terms = breakdown.terms

    if "match" in terms and terms["match"] * 100.0 < match_threshold:
        return AmbiguityReason.WEAK_TEXT_MATCH
    if terms.get("face", 1.0) < face_threshold:
        return AmbiguityReason.LOW_FACE_CONFIDENCE
    if terms.get("stability", 1.0) < 0.5:
        return AmbiguityReason.SHORT_SPEAKING_RUN
    return AmbiguityReason.LOW_CONFIDENCE


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
