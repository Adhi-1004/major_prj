"""Data contracts: immutability, derived fields, and output shape."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from dialogue_extractor.domain.enums import AmbiguityReason, Mode, Status
from dialogue_extractor.domain.models import (
    ASDFrame,
    ASDResult,
    BoundingBox,
    Candidate,
    ConfidenceBreakdown,
    DiscoveryResult,
    ExtractionResult,
    Face,
    SpeechSegment,
    Transcript,
    Word,
)


def make_face(
    *,
    face_id: int = 1,
    speaking: bool = True,
    conf: float = 0.9,
    speaker: str | None = "SPEAKER_00",
) -> Face:
    return Face(
        face_id=face_id,
        bbox=BoundingBox(0.1, 0.1, 0.2, 0.3),
        is_speaking=speaking,
        face_detection_confidence=conf,
        diarized_speaker_id=speaker,
    )


class TestImmutability:
    @pytest.mark.parametrize(
        "instance",
        [
            Word("hi", 0.0, 0.1),
            SpeechSegment(words=(), speaker_id=None),
            Candidate(0.0, 1.0, None, "x", Mode.TARGET),
            BoundingBox(0, 0, 1, 1),
            make_face(),
            ASDFrame(frame_id=0),
        ],
    )
    def test_cannot_reassign_fields(self, instance: object) -> None:
        field_name = dataclasses.fields(instance)[0].name  # type: ignore[arg-type]
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(instance, field_name, None)

    def test_sequences_are_tuples_not_lists(self) -> None:
        """Frozen alone would not stop `segment.words.append(...)`."""
        segment = SpeechSegment(words=(Word("a", 0.0, 0.1),), speaker_id="S0")
        assert isinstance(segment.words, tuple)
        assert not hasattr(segment.words, "append")


class TestSpeechSegment:
    def test_text_is_derived_from_words(self) -> None:
        words = (Word("My", 0.0, 0.2), Word("mind", 0.2, 0.5), Word("rebels", 0.5, 0.9))
        assert SpeechSegment(words=words, speaker_id="S0").text == "My mind rebels"

    def test_text_is_not_a_stored_field(self) -> None:
        """Storing it would let it drift from the words after any edit."""
        names = {f.name for f in dataclasses.fields(SpeechSegment)}
        assert "text" not in names

    def test_bounds_come_from_first_and_last_word(self) -> None:
        segment = SpeechSegment(
            words=(Word("a", 1.0, 1.5), Word("b", 1.5, 2.25)), speaker_id="S0"
        )
        assert segment.start == 1.0
        assert segment.end == 2.25

    def test_empty_segment_is_safe(self) -> None:
        empty = SpeechSegment(words=(), speaker_id=None)
        assert empty.text == ""
        assert empty.start == 0.0
        assert empty.end == 0.0
        assert empty.mean_probability == 0.0

    def test_mean_probability_averages_words(self) -> None:
        words = (Word("a", 0.0, 0.1, probability=0.8), Word("b", 0.1, 0.2, probability=0.6))
        assert SpeechSegment(words=words, speaker_id="S0").mean_probability == pytest.approx(0.7)


class TestWord:
    def test_speaker_is_none_before_alignment(self) -> None:
        assert Word("hello", 0.0, 0.4).speaker_id is None

    def test_duration(self) -> None:
        assert Word("hello", 1.0, 1.75).duration == pytest.approx(0.75)


class TestCandidate:
    def test_match_score_is_none_in_discovery_mode(self) -> None:
        """Not 1.0 -- there is no phrase to score, and 1.0 would inflate it."""
        assert Candidate(0.0, 1.0, "S0", "x", Mode.DISCOVERY).match_score is None

    def test_target_mode_carries_a_score(self) -> None:
        assert Candidate(0.0, 1.0, "S0", "x", Mode.TARGET, match_score=91.0).match_score == 91.0


class TestASDResult:
    def test_detects_when_diarization_was_dropped(self) -> None:
        """Every face null -> the service ignored our diarization payload."""
        frames = (
            ASDFrame(0, (make_face(speaker=None),)),
            ASDFrame(1, (make_face(speaker=None),)),
        )
        result = ASDResult(frames, clip_start_frame=0, fps=25.0, provider="nvidia")
        assert result.any_speaker_id_present is False

    def test_one_populated_speaker_id_is_enough(self) -> None:
        frames = (
            ASDFrame(0, (make_face(speaker=None),)),
            ASDFrame(1, (make_face(speaker="SPEAKER_01"),)),
        )
        result = ASDResult(frames, clip_start_frame=0, fps=25.0, provider="nvidia")
        assert result.any_speaker_id_present is True

    def test_no_faces_at_all(self) -> None:
        result = ASDResult(
            (ASDFrame(0), ASDFrame(1)), clip_start_frame=0, fps=25.0, provider="nvidia"
        )
        assert result.any_face_detected is False
        assert result.any_speaker_id_present is False

    def test_speaking_faces_filters(self) -> None:
        frame = ASDFrame(
            0, (make_face(face_id=1, speaking=True), make_face(face_id=2, speaking=False))
        )
        assert len(frame.speaking_faces) == 1
        assert frame.speaking_faces[0].face_id == 1


class TestExtractionResultSerialization:
    def test_matches_the_documented_success_shape(self) -> None:
        result = ExtractionResult(
            status=Status.SUCCESS,
            text="My mind rebels at stagnation",
            timestamp=102.35,
            frame_number=2456,
            confidence=0.92,
            face_detection_confidence=0.91,
            image="frame_2456.png",
        )
        assert result.to_dict() == {
            "status": "success",
            "timestamp": "00:01:42.350",
            "frame_number": 2456,
            "text": "My mind rebels at stagnation",
            "confidence": 0.92,
            "face_detection_confidence": 0.91,
            "image": "frame_2456.png",
        }

    def test_omits_absent_fields_rather_than_emitting_nulls(self) -> None:
        assert ExtractionResult(status=Status.NOT_FOUND).to_dict() == {"status": "not_found"}

    def test_not_on_screen_carries_a_reason(self) -> None:
        payload = ExtractionResult(
            status=Status.NOT_ON_SCREEN,
            text="a line",
            ambiguity_reason=AmbiguityReason.NO_FACE_DETECTED,
            detail="No face was detected in any candidate window.",
        ).to_dict()
        assert payload["status"] == "not_on_screen"
        assert payload["ambiguity_reason"] == "no_face_detected"
        assert "detail" in payload

    def test_includes_breakdown_when_present(self) -> None:
        breakdown = ConfidenceBreakdown(
            terms={"asr": 0.9, "face": 0.8, "stability": 1.0},
            weights={"asr": 0.25, "face": 0.25, "stability": 0.10},
            score=0.8583,
        )
        payload = ExtractionResult(status=Status.SUCCESS, breakdown=breakdown).to_dict()
        assert payload["confidence_breakdown"]["terms"]["face"] == 0.8

    def test_is_json_serializable(self) -> None:
        import json

        payload = ExtractionResult(
            status=Status.SUCCESS, timestamp=1.0, frame_number=24, text="x"
        ).to_dict()
        assert json.loads(json.dumps(payload))["frame_number"] == 24


class TestDiscoveryResultSerialization:
    def test_reports_truncation_honestly(self) -> None:
        payload = DiscoveryResult(
            results=(ExtractionResult(status=Status.SUCCESS, text="a"),),
            truncated=True,
            candidates_processed=10,
            candidates_total=1200,
        ).to_dict()
        assert payload["truncated"] is True
        assert payload["candidates_processed"] == 10
        assert payload["candidates_total"] == 1200

    def test_omits_truncated_when_complete(self) -> None:
        payload = DiscoveryResult(candidates_processed=5, candidates_total=5).to_dict()
        assert "truncated" not in payload

    def test_ambiguous_is_a_separate_bucket(self) -> None:
        payload = DiscoveryResult(
            results=(ExtractionResult(status=Status.SUCCESS, text="clear"),),
            ambiguous=(ExtractionResult(status=Status.AMBIGUOUS, text="unclear"),),
        ).to_dict()
        assert len(payload["results"]) == 1
        assert len(payload["ambiguous"]) == 1

    def test_empty_discovery_is_not_found(self) -> None:
        assert DiscoveryResult().to_dict()["status"] == "not_found"


class TestTranscript:
    def test_text_joins_words(self) -> None:
        t = Transcript(
            words=(Word("hello", 0.0, 0.3), Word("world", 0.3, 0.7)),
            language="en",
            duration=1.0,
        )
        assert t.text == "hello world"


class TestEnumValues:
    def test_status_values_are_the_documented_strings(self) -> None:
        assert Status.SUCCESS == "success"
        assert Status.NOT_ON_SCREEN == "not_on_screen"

    def test_enums_render_as_plain_strings_in_json(self) -> None:
        import json

        assert json.dumps({"s": Status.SUCCESS.value}) == '{"s": "success"}'


def test_media_manifest_resolution_property() -> None:
    from dialogue_extractor.domain.models import MediaManifest

    manifest = MediaManifest(
        video_path=Path("v.mp4"),
        audio_path=Path("a.wav"),
        fps=25.0,
        frame_count=250,
        duration=10.0,
        width=1280,
        height=720,
        cfr_forced=True,
        source_was_vfr=False,
        video_sha256="a" * 64,
        audio_sha256="b" * 64,
    )
    assert manifest.resolution == "1280x720"
    assert manifest.cfr_forced is True
