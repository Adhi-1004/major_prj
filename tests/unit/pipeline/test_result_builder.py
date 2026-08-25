"""Assembling the final result.

The brief asks for a timestamp, frame number, text, and image, plus a confidence
value and an honest account of uncertainty. These pin the output shape and the
distinctions between the ways a search can come back empty.
"""

from __future__ import annotations

import json

import pytest

from dialogue_extractor.domain.enums import AmbiguityReason, Mode, Status
from dialogue_extractor.domain.models import (
    BoundingBox,
    Candidate,
    ExtractionResult,
    Face,
    FrameHit,
    Word,
)
from dialogue_extractor.pipeline.result_builder import (
    build_not_found,
    build_not_on_screen,
    build_success,
)
from dialogue_extractor.vision.frame_selector import SelectionOutcome


def candidate(match_score: float | None = 94.0, probability: float = 0.9) -> Candidate:
    return Candidate(
        start=37.41,
        end=42.31,
        speaker_id="SPEAKER_04",
        text="You know, in another life, you'd have made an excellent criminal.",
        mode=Mode.TARGET if match_score is not None else Mode.DISCOVERY,
        match_score=match_score,
        words=(Word("You", 38.9, 39.1, probability=probability),),
    )


def hit(*, frame: int = 933, run_length: int = 24, confidence: float = 0.92) -> FrameHit:
    return FrameHit(
        global_frame=frame,
        timestamp=frame / 23.976024,
        face=Face(1, BoundingBox(0.1, 0.1, 0.2, 0.3), True, confidence, 2),
        run_length=run_length,
        mean_face_confidence=confidence,
    )


def outcome(
    *,
    found: bool = True,
    reason: AmbiguityReason | None = None,
    detail: str = "",
    required_run: int = 3,
    run_length: int = 24,
    confidence: float = 0.92,
) -> SelectionOutcome:
    return SelectionOutcome(
        hit=hit(run_length=run_length, confidence=confidence) if found else None,
        reason=reason,
        detail=detail,
        required_run=required_run,
    )


class TestSuccessShape:
    def test_carries_everything_the_brief_asks_for(self) -> None:
        payload = build_success(candidate(), outcome(), image_name="frame_933.png").to_dict()

        assert payload["status"] == "success"
        assert payload["frame_number"] == 933
        assert payload["timestamp"].startswith("00:00:38.")
        assert "excellent criminal" in payload["text"]
        assert payload["image"] == "frame_933.png"
        assert 0.0 <= payload["confidence"] <= 1.0
        assert 0.0 <= payload["face_detection_confidence"] <= 1.0

    def test_includes_the_speaker(self) -> None:
        assert build_success(candidate(), outcome()).to_dict()["speaker"] == "SPEAKER_04"

    def test_includes_the_confidence_breakdown(self) -> None:
        payload = build_success(candidate(), outcome()).to_dict()
        assert set(payload["confidence_breakdown"]["terms"]) == {
            "match",
            "asr",
            "face",
            "stability",
        }

    def test_is_json_serializable(self) -> None:
        payload = build_success(candidate(), outcome(), image_name="f.png").to_dict()
        assert json.loads(json.dumps(payload))["frame_number"] == 933

    def test_discovery_mode_omits_the_match_term(self) -> None:
        payload = build_success(candidate(match_score=None), outcome()).to_dict()
        assert "match" not in payload["confidence_breakdown"]["terms"]

    def test_requires_a_hit(self) -> None:
        with pytest.raises(ValueError, match="requires an outcome with a hit"):
            build_success(candidate(), outcome(found=False))


class TestStatusIsNotAutomatic:
    """A verified frame with weak evidence is ambiguous, or the confidence value
    would be decorative."""

    def test_strong_evidence_is_success(self) -> None:
        assert build_success(candidate(94.0, 0.9), outcome()).status is Status.SUCCESS

    def test_weak_evidence_is_not_reported_as_certain(self) -> None:
        result = build_success(
            candidate(71.0, 0.3), outcome(run_length=3, confidence=0.35)
        )
        assert result.status is not Status.SUCCESS
        assert result.ambiguity_reason is not None

    def test_a_selection_caveat_downgrades_a_high_score(self) -> None:
        """An unverified speaker outranks the score, because it describes
        something the score cannot express."""
        result = build_success(
            candidate(100.0, 1.0),
            outcome(reason=AmbiguityReason.SPEAKER_MISMATCH, detail="unverified"),
        )
        assert result.status is Status.AMBIGUOUS
        assert result.ambiguity_reason is AmbiguityReason.SPEAKER_MISMATCH

    def test_uncertain_results_explain_themselves(self) -> None:
        result = build_success(
            candidate(71.0, 0.3), outcome(run_length=3, confidence=0.35)
        )
        assert result.detail


class TestNotOnScreen:
    def test_carries_the_selector_diagnosis(self) -> None:
        result = build_not_on_screen(
            candidate(),
            outcome(
                found=False,
                reason=AmbiguityReason.SPEAKER_MISMATCH,
                detail="Someone else was speaking.",
            ),
        )
        assert result.status is Status.NOT_ON_SCREEN
        assert result.ambiguity_reason is AmbiguityReason.SPEAKER_MISMATCH
        assert result.detail == "Someone else was speaking."

    def test_has_no_frame_or_image(self) -> None:
        payload = build_not_on_screen(candidate(), outcome(found=False)).to_dict()
        assert "frame_number" not in payload
        assert "image" not in payload

    def test_keeps_the_text_that_was_searched_for(self) -> None:
        result = build_not_on_screen(candidate(), outcome(found=False))
        assert "excellent criminal" in (result.text or "")

    def test_falls_back_to_a_reason_when_none_given(self) -> None:
        result = build_not_on_screen(candidate(), outcome(found=False))
        assert result.ambiguity_reason is not None
        assert result.detail


class TestNotFoundIsDistinct:
    """"Never said" and "said off camera" are different answers; conflating them
    sends the user looking in the wrong place."""

    def test_status_differs_from_not_on_screen(self) -> None:
        assert build_not_found("some line").status is Status.NOT_FOUND
        assert (
            build_not_on_screen(candidate(), outcome(found=False)).status
            is Status.NOT_ON_SCREEN
        )

    def test_explains_the_likely_causes(self) -> None:
        result = build_not_found("some line")
        assert result.detail is not None
        assert "worded differently" in result.detail

    def test_echoes_the_requested_phrase(self) -> None:
        assert build_not_found("my mind rebels").text == "my mind rebels"


class TestEveryStatusSerializes:
    @pytest.mark.parametrize(
        "result",
        [
            build_success(candidate(), outcome(), image_name="f.png"),
            build_not_on_screen(candidate(), outcome(found=False)),
            build_not_found("x"),
        ],
    )
    def test_round_trips_through_json(self, result: ExtractionResult) -> None:
        payload = result.to_dict()
        assert json.loads(json.dumps(payload))["status"] == payload["status"]
