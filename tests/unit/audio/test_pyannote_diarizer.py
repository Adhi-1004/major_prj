"""Diarization: turn extraction, gated-access diagnosis, failure handling.

No pipeline is loaded and no network is touched. The behaviour worth pinning is
what happens around the model, not inside it -- especially the gated-access
path, which is the single most common way this stage fails on a fresh machine
and whose native error message says nothing actionable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from tests.conftest import write_silent_wav

from dialogue_extractor.audio.pyannote_diarizer import (
    PyannoteDiarizer,
    extract_turns,
    looks_like_gated_access,
    turns_from_dict,
    turns_to_dict,
    unwrap_annotation,
)
from dialogue_extractor.config import constants
from dialogue_extractor.domain.errors import DiarizationError, GatedModelError
from dialogue_extractor.domain.models import SpeakerTurn


class FakeSegment:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class FakeAnnotation:
    """Stands in for a pyannote Annotation."""

    def __init__(self, tracks: list[tuple[FakeSegment, str, str]]) -> None:
        self._tracks = tracks

    def itertracks(self, yield_label: bool = False) -> Any:
        return iter(self._tracks)


class FakePipeline:
    def __init__(self, annotation: FakeAnnotation) -> None:
        self._annotation = annotation
        self.last_kwargs: dict[str, Any] = {}
        self.calls = 0

    def __call__(self, audio: str, **kwargs: Any) -> FakeAnnotation:
        self.calls += 1
        self.last_kwargs = kwargs
        return self._annotation

    def to(self, device: Any) -> FakePipeline:
        return self


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    """A real, minimal PCM WAV.

    Diarization decodes the audio in-process now (to avoid torchcodec), so a
    placeholder byte string no longer suffices.
    """
    path = tmp_path / "audio.wav"
    write_silent_wav(path, seconds=0.1)
    return path


def annotation(*tracks: tuple[float, float, str]) -> FakeAnnotation:
    return FakeAnnotation([(FakeSegment(s, e), "_", label) for s, e, label in tracks])


class TestExtractTurns:
    def test_converts_tracks_to_turns(self) -> None:
        turns = extract_turns(annotation((0.0, 5.0, "SPEAKER_00"), (5.2, 9.0, "SPEAKER_01")))
        assert len(turns) == 2
        assert turns[0] == SpeakerTurn("SPEAKER_00", 0.0, 5.0)
        assert turns[1].speaker_id == "SPEAKER_01"

    def test_output_is_sorted_by_time(self) -> None:
        """Alignment walks these in order and breaks early on the first turn
        starting after the word ends."""
        turns = extract_turns(annotation((5.0, 9.0, "SPEAKER_01"), (0.0, 4.0, "SPEAKER_00")))
        assert [t.start for t in turns] == [0.0, 5.0]

    def test_drops_zero_length_turns(self) -> None:
        """They carry no information and only add noise to overlap scoring."""
        turns = extract_turns(annotation((3.0, 3.0, "SPEAKER_00"), (4.0, 5.0, "SPEAKER_01")))
        assert len(turns) == 1
        assert turns[0].speaker_id == "SPEAKER_01"

    def test_drops_inverted_turns(self) -> None:
        turns = extract_turns(annotation((9.0, 5.0, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_01")))
        assert [t.speaker_id for t in turns] == ["SPEAKER_01"]

    def test_drops_negative_turns(self) -> None:
        turns = extract_turns(annotation((-2.0, 1.0, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_01")))
        assert [t.speaker_id for t in turns] == ["SPEAKER_01"]

    def test_labels_are_coerced_to_strings(self) -> None:
        """A numeric label would break the string speaker_id contract the ASD
        payload depends on."""
        tracks = [(FakeSegment(0.0, 1.0), "_", 0)]
        turns = extract_turns(FakeAnnotation(tracks))  # type: ignore[arg-type]
        assert turns[0].speaker_id == "0"
        assert isinstance(turns[0].speaker_id, str)

    def test_empty_annotation(self) -> None:
        assert extract_turns(FakeAnnotation([])) == ()

    def test_survives_malformed_segments(self) -> None:
        class Broken:
            start = "not a number"
            end = 1.0

        tracks = [(Broken(), "_", "SPEAKER_00"), (FakeSegment(1.0, 2.0), "_", "SPEAKER_01")]
        turns = extract_turns(FakeAnnotation(tracks))  # type: ignore[arg-type]
        assert [t.speaker_id for t in turns] == ["SPEAKER_01"]


class TestGatedAccessDetection:
    @pytest.mark.parametrize(
        "message",
        [
            "Access to model pyannote/speaker-diarization-community-1 is restricted",
            "You must accept the conditions to access this model",
            "401 Client Error: Unauthorized for url",
            "Your request to access this repo is awaiting a review",
            "This repository is gated",
            "Authentication failed",
            "Cannot access gated repo",
        ],
    )
    def test_recognizes_gated_failures(self, message: str) -> None:
        assert looks_like_gated_access(RuntimeError(message)) is True

    @pytest.mark.parametrize(
        "message",
        ["CUDA out of memory", "audio file is corrupt", "connection reset by peer"],
    )
    def test_ignores_unrelated_failures(self, message: str) -> None:
        assert looks_like_gated_access(RuntimeError(message)) is False


class TestMissingToken:
    def test_no_token_fails_before_any_network_call(self, audio_file: Path) -> None:
        diarizer = PyannoteDiarizer(token=None)
        with pytest.raises(GatedModelError) as excinfo:
            diarizer.diarize(audio_file)
        assert excinfo.value.reason == "missing_hf_token"

    def test_remediation_names_both_required_steps(self, audio_file: Path) -> None:
        """A valid token alone is not enough -- the terms must also be accepted,
        and that is the failure people actually hit."""
        with pytest.raises(GatedModelError) as excinfo:
            PyannoteDiarizer(token=None).diarize(audio_file)

        remediation = excinfo.value.to_dict()["remediation"]
        assert constants.PYANNOTE_PIPELINE in remediation
        assert "HF_TOKEN" in remediation
        assert "conditions" in remediation.lower()


class TestDiarize:
    def test_returns_turns(self, audio_file: Path) -> None:
        pipeline = FakePipeline(annotation((0.0, 5.0, "SPEAKER_00"), (5.2, 9.0, "SPEAKER_01")))
        turns = PyannoteDiarizer(pipeline=pipeline).diarize(audio_file)

        assert len(turns) == 2
        assert {t.speaker_id for t in turns} == {"SPEAKER_00", "SPEAKER_01"}

    def test_missing_audio_raises(self, tmp_path: Path) -> None:
        pipeline = FakePipeline(annotation((0.0, 1.0, "SPEAKER_00")))
        with pytest.raises(DiarizationError) as excinfo:
            PyannoteDiarizer(pipeline=pipeline).diarize(tmp_path / "absent.wav")
        assert excinfo.value.reason == "audio_not_found"

    def test_no_speakers_raises_a_specific_error(self, audio_file: Path) -> None:
        """An empty result must say why, not silently produce zero candidates
        that later look like "the phrase is not in this video"."""
        with pytest.raises(DiarizationError) as excinfo:
            PyannoteDiarizer(pipeline=FakePipeline(FakeAnnotation([]))).diarize(audio_file)
        assert excinfo.value.reason == "no_speakers_detected"

    def test_pipeline_failure_is_wrapped(self, audio_file: Path) -> None:
        class ExplodingPipeline:
            def __call__(self, audio: str, **kwargs: Any) -> Any:
                raise RuntimeError("something went wrong inside pyannote")

        with pytest.raises(DiarizationError) as excinfo:
            PyannoteDiarizer(pipeline=ExplodingPipeline()).diarize(audio_file)
        assert excinfo.value.reason == "diarization_failed"
        assert excinfo.value.to_dict()["stage"] == "diarization"

    def test_gated_failure_at_inference_is_still_diagnosed(self, audio_file: Path) -> None:
        class GatedPipeline:
            def __call__(self, audio: str, **kwargs: Any) -> Any:
                raise RuntimeError("401 Client Error: Unauthorized")

        with pytest.raises(GatedModelError) as excinfo:
            PyannoteDiarizer(pipeline=GatedPipeline()).diarize(audio_file)
        assert excinfo.value.reason == "gated_model_access"


class TestSpeakerCountHints:
    def test_no_hints_by_default(self, audio_file: Path) -> None:
        """An arbitrary video has an unknown speaker count; let pyannote estimate."""
        pipeline = FakePipeline(annotation((0.0, 1.0, "SPEAKER_00")))
        PyannoteDiarizer(pipeline=pipeline).diarize(audio_file)
        assert pipeline.last_kwargs == {}

    def test_fixed_count_is_passed_through(self, audio_file: Path) -> None:
        pipeline = FakePipeline(annotation((0.0, 1.0, "SPEAKER_00")))
        PyannoteDiarizer(pipeline=pipeline, num_speakers=2).diarize(audio_file)
        assert pipeline.last_kwargs == {"num_speakers": 2}

    def test_bounds_are_passed_through(self, audio_file: Path) -> None:
        pipeline = FakePipeline(annotation((0.0, 1.0, "SPEAKER_00")))
        PyannoteDiarizer(pipeline=pipeline, min_speakers=2, max_speakers=5).diarize(audio_file)
        assert pipeline.last_kwargs == {"min_speakers": 2, "max_speakers": 5}


class TestSerialization:
    def test_round_trips(self) -> None:
        turns = (SpeakerTurn("SPEAKER_00", 0.0, 5.0), SpeakerTurn("SPEAKER_01", 5.2, 9.0))
        assert turns_from_dict(turns_to_dict(turns)) == turns

    def test_is_json_serializable(self) -> None:
        import json

        turns = (SpeakerTurn("SPEAKER_00", 0.0, 5.0),)
        assert json.loads(json.dumps(turns_to_dict(turns)))[0]["speaker_id"] == "SPEAKER_00"

    def test_empty(self) -> None:
        assert turns_from_dict(turns_to_dict(())) == ()


class FakeDiarizeOutput:
    """Mirrors pyannote.audio 4.x's DiarizeOutput.

    4.x returns this wrapper rather than the bare Annotation 3.x returned, and
    reading it as an Annotation raises AttributeError deep inside conversion.
    """

    def __init__(
        self,
        speaker_diarization: FakeAnnotation | None = None,
        exclusive: FakeAnnotation | None = None,
    ) -> None:
        if speaker_diarization is not None:
            self.speaker_diarization = speaker_diarization
        if exclusive is not None:
            self.exclusive_speaker_diarization = exclusive
        self.speaker_embeddings = None


class TestUnwrapAnnotation:
    def test_bare_annotation_passes_through(self) -> None:
        """3.x-style results, and the fakes used throughout these tests."""
        ann = annotation((0.0, 1.0, "SPEAKER_00"))
        assert unwrap_annotation(ann) is ann

    def test_unwraps_diarize_output(self) -> None:
        ann = annotation((0.0, 1.0, "SPEAKER_00"))
        assert unwrap_annotation(FakeDiarizeOutput(speaker_diarization=ann)) is ann

    def test_prefers_the_overlapping_annotation(self) -> None:
        """Alignment resolves overlaps by greatest overlap, so crosstalk is
        information rather than a problem."""
        primary = annotation((0.0, 1.0, "SPEAKER_00"))
        exclusive = annotation((0.0, 1.0, "SPEAKER_01"))
        result = FakeDiarizeOutput(speaker_diarization=primary, exclusive=exclusive)
        assert unwrap_annotation(result) is primary

    def test_falls_back_to_the_exclusive_annotation(self) -> None:
        exclusive = annotation((0.0, 1.0, "SPEAKER_01"))
        assert unwrap_annotation(FakeDiarizeOutput(exclusive=exclusive)) is exclusive

    def test_unreadable_result_raises_a_stage_error(self) -> None:
        class Mystery:
            pass

        with pytest.raises(DiarizationError) as excinfo:
            unwrap_annotation(Mystery())
        assert excinfo.value.reason == "unexpected_result_shape"

    def test_extract_turns_accepts_the_wrapper(self) -> None:
        result = FakeDiarizeOutput(
            speaker_diarization=annotation((0.0, 5.0, "SPEAKER_00"), (5.2, 9.0, "SPEAKER_01"))
        )
        turns = extract_turns(result)
        assert [t.speaker_id for t in turns] == ["SPEAKER_00", "SPEAKER_01"]


class TestErrorBoundaryCoversConversion:
    """A failure while reading the result must be a structured error.

    The first real run raised AttributeError from inside conversion and it
    escaped as a traceback, because conversion sat outside the try block.
    """

    def test_wrapper_shape_change_is_reported_not_raised(self, audio_file: Path) -> None:
        class Mystery:
            pass

        class MysteryPipeline:
            def __call__(self, audio: Any, **kwargs: Any) -> Any:
                return Mystery()

        with pytest.raises(DiarizationError) as excinfo:
            PyannoteDiarizer(pipeline=MysteryPipeline()).diarize(audio_file)

        assert excinfo.value.reason == "unexpected_result_shape"
        assert excinfo.value.to_dict()["stage"] == "diarization"

    def test_arbitrary_conversion_failure_is_wrapped(self, audio_file: Path) -> None:
        class ExplodingAnnotation:
            def itertracks(self, yield_label: bool = False) -> Any:
                raise ValueError("iteration blew up")

        class ExplodingPipeline:
            def __call__(self, audio: Any, **kwargs: Any) -> Any:
                return ExplodingAnnotation()

        with pytest.raises(DiarizationError) as excinfo:
            PyannoteDiarizer(pipeline=ExplodingPipeline()).diarize(audio_file)

        assert excinfo.value.reason == "unexpected_result_shape"
        assert excinfo.value.to_dict()["remediation"]

    def test_no_raw_exception_escapes_the_diarize_call(self, audio_file: Path) -> None:
        """Whatever goes wrong, the caller sees a StageError."""

        class ChaosPipeline:
            def __call__(self, audio: Any, **kwargs: Any) -> Any:
                raise KeyError("something entirely unexpected")

        with pytest.raises(DiarizationError):
            PyannoteDiarizer(pipeline=ChaosPipeline()).diarize(audio_file)


class TestShortTurnFiltering:
    """Diarization emits sub-syllable turns that are artifacts, not data.

    Left in, they split single utterances across speakers -- and since Phase 5
    cross-checks the matched phrase's speaker against the face the detector
    reports as speaking, a word misattributed here makes a correct on-screen
    match fail verification.
    """

    def test_drops_turns_below_the_minimum(self) -> None:
        turns = extract_turns(
            annotation(
                (0.0, 0.017, "SPEAKER_00"),  # 17ms -- observed in real output
                (1.0, 3.0, "SPEAKER_01"),
            ),
            min_duration=0.25,
        )
        assert [t.speaker_id for t in turns] == ["SPEAKER_01"]

    def test_keeps_turns_at_the_threshold(self) -> None:
        turns = extract_turns(annotation((0.0, 0.25, "SPEAKER_00")), min_duration=0.25)
        assert len(turns) == 1

    def test_uses_the_configured_default(self) -> None:
        turns = extract_turns(annotation((0.0, 0.1, "SPEAKER_00"), (1.0, 3.0, "SPEAKER_01")))
        assert [t.speaker_id for t in turns] == ["SPEAKER_01"]

    def test_filtering_can_be_disabled(self) -> None:
        turns = extract_turns(annotation((0.0, 0.017, "SPEAKER_00")), min_duration=0.0)
        assert len(turns) == 1

    def test_prevents_an_utterance_being_split_across_speakers(self) -> None:
        """The concrete regression from real output.

        Turn boundaries and word timings below are the actual values observed:
        "Until next time, Big Joe." came back as SPEAKER_05 / SPEAKER_03 /
        SPEAKER_05 because a 0.20s turn sat between two halves of one speaker's
        real turn and won the overlap contest for a single word.
        """
        from dialogue_extractor.audio.aligner import assign_speakers, build_segments
        from dialogue_extractor.domain.models import Word

        words = (
            Word("Until", 30.65, 30.95),
            Word("next", 30.95, 31.30),
            Word("time", 31.45, 31.75),
            Word("Big", 31.87, 32.10),
            Word("Joe", 32.10, 32.40),
        )
        # Consecutive, non-overlapping, exactly as pyannote emitted them.
        raw = annotation(
            (31.047, 31.570, "SPEAKER_05"),
            (31.570, 31.773, "SPEAKER_03"),  # 0.20s artifact
            (31.773, 32.515, "SPEAKER_05"),
            (30.500, 31.047, "SPEAKER_05"),
        )

        unfiltered = build_segments(
            assign_speakers(words, extract_turns(raw, min_duration=0.0))
        )
        filtered = build_segments(
            assign_speakers(words, extract_turns(raw, min_duration=0.25))
        )

        # Unfiltered: "time" is stolen by the artifact, splitting the utterance.
        assert [s.speaker_id for s in unfiltered] == [
            "SPEAKER_05",
            "SPEAKER_03",
            "SPEAKER_05",
        ]

        # Filtered: one utterance, one speaker.
        assert len(filtered) == 1
        assert filtered[0].speaker_id == "SPEAKER_05"
        assert filtered[0].text == "Until next time Big Joe"
