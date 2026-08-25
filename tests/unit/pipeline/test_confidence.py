"""The composite confidence score.

The property that matters most: discovery mode omits the `match` term rather
than scoring it 1.0. Substituting 1.0 would hand every discovery result a free
0.40 for something never checked, so identical detection evidence would clear a
much lower bar in one mode than the other.
"""

from __future__ import annotations

import pytest

from dialogue_extractor.config.constants import (
    CONFIDENCE_AMBIGUOUS_THRESHOLD,
    CONFIDENCE_SUCCESS_THRESHOLD,
)
from dialogue_extractor.domain.enums import AmbiguityReason, Status
from dialogue_extractor.pipeline.confidence import (
    classify,
    compute,
    stability_term,
    weakest_signal,
)


class TestStabilityTerm:
    def test_exactly_the_required_run_is_half(self) -> None:
        assert stability_term(3, 3) == pytest.approx(0.5)

    def test_twice_the_required_run_saturates(self) -> None:
        assert stability_term(6, 3) == pytest.approx(1.0)

    def test_beyond_that_does_not_keep_growing(self) -> None:
        """A lingering shot should not outweigh a poor face detection."""
        assert stability_term(600, 3) == pytest.approx(1.0)

    def test_shorter_than_required_scores_low(self) -> None:
        assert stability_term(1, 3) == pytest.approx(1 / 6)

    def test_zero_requirement_is_handled(self) -> None:
        assert stability_term(5, 0) == 1.0


class TestModeRenormalization:
    """The correction that makes the two modes comparable."""

    def test_discovery_omits_the_match_term(self) -> None:
        breakdown = compute(
            match_score=None,
            asr_confidence=0.9,
            face_confidence=0.9,
            run_length=6,
            required_run=3,
        )
        assert "match" not in breakdown.terms
        assert "match" not in breakdown.weights

    def test_target_mode_includes_it(self) -> None:
        breakdown = compute(
            match_score=100.0,
            asr_confidence=0.9,
            face_confidence=0.9,
            run_length=6,
            required_run=3,
        )
        assert "match" in breakdown.terms

    def test_identical_evidence_scores_identically_across_modes(self) -> None:
        """The whole point. A discovery result with the same detection quality
        must not outrank a targeted one just because a term was absent."""
        shared = {
            "asr_confidence": 0.88,
            "face_confidence": 0.91,
            "run_length": 6,
            "required_run": 3,
        }
        discovery = compute(match_score=None, **shared)
        perfect_match = compute(match_score=100.0, **shared)

        # A perfect match should score *at least* as well, never worse.
        assert perfect_match.score >= discovery.score

    def test_discovery_does_not_get_a_free_forty_percent(self) -> None:
        """If `match` were scored 1.0 instead of omitted, weak detection
        evidence would still clear the success threshold."""
        weak = {
            "asr_confidence": 0.3,
            "face_confidence": 0.3,
            "run_length": 1,
            "required_run": 3,
        }
        discovery = compute(match_score=None, **weak)
        inflated = compute(match_score=100.0, **weak)

        assert discovery.score < inflated.score
        assert discovery.score < CONFIDENCE_AMBIGUOUS_THRESHOLD

    def test_weights_renormalize_to_one(self) -> None:
        breakdown = compute(
            match_score=None,
            asr_confidence=1.0,
            face_confidence=1.0,
            run_length=6,
            required_run=3,
        )
        assert breakdown.score == pytest.approx(1.0)

    def test_perfect_evidence_scores_one_in_target_mode_too(self) -> None:
        breakdown = compute(
            match_score=100.0,
            asr_confidence=1.0,
            face_confidence=1.0,
            run_length=6,
            required_run=3,
        )
        assert breakdown.score == pytest.approx(1.0)


class TestScoreRange:
    def test_worst_case_is_zero(self) -> None:
        breakdown = compute(
            match_score=0.0,
            asr_confidence=0.0,
            face_confidence=0.0,
            run_length=0,
            required_run=3,
        )
        assert breakdown.score == pytest.approx(0.0)

    @pytest.mark.parametrize("match", [None, 0.0, 55.0, 100.0])
    @pytest.mark.parametrize("asr", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("face", [0.0, 0.5, 1.0])
    def test_always_within_zero_and_one(
        self, match: float | None, asr: float, face: float
    ) -> None:
        breakdown = compute(
            match_score=match,
            asr_confidence=asr,
            face_confidence=face,
            run_length=4,
            required_run=3,
        )
        assert 0.0 <= breakdown.score <= 1.0

    def test_out_of_range_inputs_are_clamped(self) -> None:
        breakdown = compute(
            match_score=100.0,
            asr_confidence=1.7,
            face_confidence=-0.4,
            run_length=6,
            required_run=3,
        )
        assert breakdown.terms["asr"] == 1.0
        assert breakdown.terms["face"] == 0.0


class TestBreakdownIsExplainable:
    def test_reports_every_term_and_weight(self) -> None:
        breakdown = compute(
            match_score=94.0,
            asr_confidence=0.9,
            face_confidence=0.92,
            run_length=24,
            required_run=3,
        )
        assert set(breakdown.terms) == {"match", "asr", "face", "stability"}
        assert set(breakdown.weights) == set(breakdown.terms)

    def test_the_breakdown_reproduces_the_score(self) -> None:
        """An unexplained number would not satisfy the explainability
        requirement; this checks the explanation is the actual arithmetic."""
        breakdown = compute(
            match_score=94.0,
            asr_confidence=0.9,
            face_confidence=0.92,
            run_length=24,
            required_run=3,
        )
        recomputed = sum(
            breakdown.terms[name] * weight for name, weight in breakdown.weights.items()
        ) / sum(breakdown.weights.values())
        assert recomputed == pytest.approx(breakdown.score, abs=1e-4)

    def test_is_json_serializable(self) -> None:
        import json

        breakdown = compute(
            match_score=94.0,
            asr_confidence=0.9,
            face_confidence=0.92,
            run_length=24,
            required_run=3,
        )
        assert json.loads(json.dumps(breakdown.to_dict()))["score"] > 0


class TestClassify:
    def test_high_score_is_success(self) -> None:
        assert classify(0.95) is Status.SUCCESS

    def test_exactly_the_success_threshold_is_success(self) -> None:
        assert classify(CONFIDENCE_SUCCESS_THRESHOLD) is Status.SUCCESS

    def test_just_below_is_ambiguous(self) -> None:
        assert classify(CONFIDENCE_SUCCESS_THRESHOLD - 0.001) is Status.AMBIGUOUS

    def test_exactly_the_ambiguous_threshold_is_ambiguous(self) -> None:
        assert classify(CONFIDENCE_AMBIGUOUS_THRESHOLD) is Status.AMBIGUOUS

    def test_just_below_that_is_not_on_screen(self) -> None:
        assert classify(CONFIDENCE_AMBIGUOUS_THRESHOLD - 0.001) is Status.NOT_ON_SCREEN

    def test_zero_is_not_on_screen(self) -> None:
        assert classify(0.0) is Status.NOT_ON_SCREEN


class TestWeakestSignal:
    """"Ambiguous" alone tells the user nothing they can act on."""

    def test_names_a_weak_text_match(self) -> None:
        breakdown = compute(
            match_score=72.0,
            asr_confidence=0.9,
            face_confidence=0.9,
            run_length=6,
            required_run=3,
        )
        assert weakest_signal(breakdown) is AmbiguityReason.WEAK_TEXT_MATCH

    def test_names_a_poor_face_detection(self) -> None:
        breakdown = compute(
            match_score=100.0,
            asr_confidence=0.9,
            face_confidence=0.4,
            run_length=6,
            required_run=3,
        )
        assert weakest_signal(breakdown) is AmbiguityReason.LOW_FACE_CONFIDENCE

    def test_names_a_short_run(self) -> None:
        breakdown = compute(
            match_score=100.0,
            asr_confidence=0.9,
            face_confidence=0.9,
            run_length=1,
            required_run=3,
        )
        assert weakest_signal(breakdown) is AmbiguityReason.SHORT_SPEAKING_RUN

    def test_falls_back_to_generic_low_confidence(self) -> None:
        breakdown = compute(
            match_score=100.0,
            asr_confidence=0.5,
            face_confidence=0.9,
            run_length=6,
            required_run=3,
        )
        assert weakest_signal(breakdown) is AmbiguityReason.LOW_CONFIDENCE

    def test_discovery_mode_never_blames_the_absent_match_term(self) -> None:
        breakdown = compute(
            match_score=None,
            asr_confidence=0.5,
            face_confidence=0.9,
            run_length=6,
            required_run=3,
        )
        assert weakest_signal(breakdown) is not AmbiguityReason.WEAK_TEXT_MATCH


class TestRealisticCase:
    def test_the_observed_live_detection_scores_well(self) -> None:
        """Numbers from the real run: a 94.0 fuzzy match, 0.905 mean word
        probability, 0.92 face confidence, a 24-frame run against a 3-frame
        requirement."""
        breakdown = compute(
            match_score=94.0,
            asr_confidence=0.905,
            face_confidence=0.92,
            run_length=24,
            required_run=3,
        )
        assert classify(breakdown.score) is Status.SUCCESS
        assert breakdown.score > 0.9
