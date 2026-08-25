"""Transcription: word extraction, determinism, device selection, failure modes.

No model is loaded. faster-whisper's output shape is simple enough to fake, and
the logic worth testing -- what gets dropped, what order words come out in, what
decoding parameters are used -- is all independent of the weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dialogue_extractor.audio.whisper_transcriber import (
    WhisperTranscriber,
    extract_words,
    resolve_device,
)
from dialogue_extractor.config import constants
from dialogue_extractor.domain.errors import TranscriptionError


class FakeWord:
    def __init__(
        self,
        word: str,
        start: float | None,
        end: float | None,
        probability: float | None = 0.9,
    ) -> None:
        self.word = word
        self.start = start
        self.end = end
        self.probability = probability


class FakeSegment:
    def __init__(self, words: list[FakeWord] | None) -> None:
        self.words = words


class FakeInfo:
    def __init__(self, language: str = "en", duration: float = 10.0) -> None:
        self.language = language
        self.duration = duration


class FakeModel:
    """Stands in for faster_whisper.WhisperModel."""

    def __init__(self, segments: list[FakeSegment], info: FakeInfo | None = None) -> None:
        self._segments = segments
        self._info = info or FakeInfo()
        self.last_options: dict[str, Any] = {}
        self.calls = 0

    def transcribe(self, audio: str, **options: Any) -> tuple[list[FakeSegment], FakeInfo]:
        self.calls += 1
        self.last_options = options
        return self._segments, self._info


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "audio.wav"
    path.write_bytes(b"RIFF....WAVEfmt ")
    return path


class TestExtractWords:
    def test_flattens_segments_into_one_stream(self) -> None:
        segments = [
            FakeSegment([FakeWord("My", 0.0, 0.2), FakeWord("mind", 0.2, 0.5)]),
            FakeSegment([FakeWord("rebels", 0.5, 0.9)]),
        ]
        words = extract_words(segments)
        assert [w.text for w in words] == ["My", "mind", "rebels"]

    def test_strips_whitespace_from_word_text(self) -> None:
        """faster-whisper emits words with a leading space."""
        words = extract_words([FakeSegment([FakeWord(" mind", 0.2, 0.5)])])
        assert words[0].text == "mind"

    def test_captures_probability(self) -> None:
        words = extract_words([FakeSegment([FakeWord("mind", 0.2, 0.5, probability=0.83)])])
        assert words[0].probability == pytest.approx(0.83)

    def test_missing_probability_defaults_to_one(self) -> None:
        words = extract_words([FakeSegment([FakeWord("mind", 0.2, 0.5, probability=None)])])
        assert words[0].probability == 1.0

    def test_probability_is_clamped(self) -> None:
        words = extract_words(
            [FakeSegment([FakeWord("a", 0.0, 0.1, 1.7), FakeWord("b", 0.1, 0.2, -0.4)])]
        )
        assert words[0].probability == 1.0
        assert words[1].probability == 0.0

    @pytest.mark.parametrize(
        "bad",
        [
            FakeWord("x", None, 0.5),
            FakeWord("x", 0.2, None),
            FakeWord("", 0.2, 0.5),
            FakeWord("   ", 0.2, 0.5),
            FakeWord("x", 0.9, 0.2),  # end before start
            FakeWord("x", -1.0, 0.5),  # negative start
        ],
    )
    def test_drops_unusable_words(self, bad: FakeWord) -> None:
        """An unmatched or malformed timing cannot become a candidate window,
        and keeping it would corrupt the ordering matching relies on."""
        good = FakeWord("keep", 1.0, 1.2)
        words = extract_words([FakeSegment([bad, good])])
        assert [w.text for w in words] == ["keep"]

    def test_drops_non_numeric_timings(self) -> None:
        words = extract_words([FakeSegment([FakeWord("x", "abc", 0.5)])])  # type: ignore[arg-type]
        assert words == ()

    def test_handles_segment_with_no_words(self) -> None:
        assert extract_words([FakeSegment(None), FakeSegment([])]) == ()

    def test_handles_no_segments(self) -> None:
        assert extract_words([]) == ()

    def test_output_is_sorted_by_time(self) -> None:
        """Matching slides a window over this stream; order must be monotonic."""
        segments = [
            FakeSegment([FakeWord("second", 5.0, 5.4)]),
            FakeSegment([FakeWord("first", 1.0, 1.4)]),
        ]
        words = extract_words(segments)
        assert [w.text for w in words] == ["first", "second"]
        assert all(
            words[i].start <= words[i + 1].start for i in range(len(words) - 1)
        )

    def test_speaker_is_unset_before_alignment(self) -> None:
        words = extract_words([FakeSegment([FakeWord("x", 0.0, 0.2)])])
        assert words[0].speaker_id is None


class TestCrossSegmentPhrases:
    """A phrase split across faster-whisper segments must stay matchable.

    This is the "dialogue spans multiple transcription chunks" edge case. It is
    handled by *flattening* -- segment boundaries are discarded entirely, so
    matching slides over one continuous word stream and never sees an edge.
    """

    def test_phrase_split_across_segments_is_contiguous(self) -> None:
        segments = [
            FakeSegment([FakeWord("My", 10.0, 10.2), FakeWord("mind", 10.2, 10.5)]),
            FakeSegment(
                [
                    FakeWord("rebels", 10.5, 10.9),
                    FakeWord("at", 10.9, 11.0),
                    FakeWord("stagnation", 11.0, 11.6),
                ]
            ),
        ]
        words = extract_words(segments)
        assert " ".join(w.text for w in words) == "My mind rebels at stagnation"

    def test_no_gap_is_introduced_at_a_segment_boundary(self) -> None:
        segments = [
            FakeSegment([FakeWord("mind", 10.2, 10.5)]),
            FakeSegment([FakeWord("rebels", 10.5, 10.9)]),
        ]
        words = extract_words(segments)
        assert words[0].end == words[1].start


class TestDeterminism:
    """faster-whisper's default temperature ladder resamples on low confidence,
    so the same audio yields different transcripts run to run."""

    def test_temperature_is_pinned_to_zero(self, audio_file: Path) -> None:
        model = FakeModel([FakeSegment([FakeWord("x", 0.0, 0.2)])])
        WhisperTranscriber(model=model).transcribe(audio_file)
        assert model.last_options["temperature"] == 0.0

    def test_conditioning_on_previous_text_is_off(self, audio_file: Path) -> None:
        """Otherwise each window depends on prior output, so one hallucination
        cascades into later segments."""
        model = FakeModel([FakeSegment([FakeWord("x", 0.0, 0.2)])])
        WhisperTranscriber(model=model).transcribe(audio_file)
        assert model.last_options["condition_on_previous_text"] is False

    def test_word_timestamps_are_requested(self, audio_file: Path) -> None:
        model = FakeModel([FakeSegment([FakeWord("x", 0.0, 0.2)])])
        WhisperTranscriber(model=model).transcribe(audio_file)
        assert model.last_options["word_timestamps"] is True

    def test_beam_size_is_fixed(self, audio_file: Path) -> None:
        model = FakeModel([FakeSegment([FakeWord("x", 0.0, 0.2)])])
        WhisperTranscriber(model=model).transcribe(audio_file)
        assert model.last_options["beam_size"] == constants.WHISPER_BEAM_SIZE

    def test_language_is_passed_through(self, audio_file: Path) -> None:
        model = FakeModel([FakeSegment([FakeWord("x", 0.0, 0.2)])])
        WhisperTranscriber(model=model, language="en").transcribe(audio_file)
        assert model.last_options["language"] == "en"

    def test_language_can_be_auto_detected(self, audio_file: Path) -> None:
        model = FakeModel([FakeSegment([FakeWord("x", 0.0, 0.2)])])
        WhisperTranscriber(model=model, language=None).transcribe(audio_file)
        assert model.last_options["language"] is None

    def test_repeated_runs_use_identical_options(self, audio_file: Path) -> None:
        model = FakeModel([FakeSegment([FakeWord("x", 0.0, 0.2)])])
        transcriber = WhisperTranscriber(model=model)
        transcriber.transcribe(audio_file)
        first = dict(model.last_options)
        transcriber.transcribe(audio_file)
        assert model.last_options == first


class TestTranscribe:
    def test_returns_a_transcript(self, audio_file: Path) -> None:
        model = FakeModel(
            [FakeSegment([FakeWord("hello", 0.0, 0.4), FakeWord("world", 0.4, 0.8)])],
            FakeInfo(language="en", duration=1.0),
        )
        transcript = WhisperTranscriber(model=model).transcribe(audio_file)
        assert transcript.text == "hello world"
        assert transcript.language == "en"
        assert transcript.duration == pytest.approx(1.0)

    def test_missing_audio_raises(self, tmp_path: Path) -> None:
        model = FakeModel([])
        with pytest.raises(TranscriptionError) as excinfo:
            WhisperTranscriber(model=model).transcribe(tmp_path / "absent.wav")
        assert excinfo.value.reason == "audio_not_found"

    def test_silence_raises_a_specific_error(self, audio_file: Path) -> None:
        """An empty transcript must say why, not return an empty result that
        later looks like "phrase not found"."""
        model = FakeModel([FakeSegment([])])
        with pytest.raises(TranscriptionError) as excinfo:
            WhisperTranscriber(model=model).transcribe(audio_file)
        assert excinfo.value.reason == "no_speech_detected"

    def test_model_failure_is_wrapped(self, audio_file: Path) -> None:
        class ExplodingModel:
            def transcribe(self, audio: str, **options: Any) -> Any:
                raise RuntimeError("tokenizer file is corrupt")

        # device="cpu" on purpose: a CUDA-flavoured message here would take the
        # CPU-fallback path and load a real model. That path has its own tests
        # in test_cuda_fallback.py, with the model faked out.
        with pytest.raises(TranscriptionError) as excinfo:
            WhisperTranscriber(model=ExplodingModel(), device="cpu").transcribe(audio_file)
        assert excinfo.value.reason == "transcription_failed"
        assert "tokenizer file is corrupt" in excinfo.value.detail
        assert excinfo.value.to_dict()["stage"] == "transcription"


class TestDeviceResolution:
    def test_explicit_cpu(self) -> None:
        assert resolve_device("cpu") == ("cpu", "int8")

    def test_explicit_cuda(self) -> None:
        """int8_float16 keeps large-v3 inside 8 GB of VRAM."""
        assert resolve_device("cuda") == ("cuda", "int8_float16")

    def test_auto_uses_cuda_when_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import ctranslate2

        monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 1)
        assert resolve_device("auto") == ("cuda", "int8_float16")

    def test_auto_falls_back_to_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import ctranslate2

        monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 0)
        assert resolve_device("auto") == ("cpu", "int8")

    def test_probe_failure_is_not_fatal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A broken CUDA install must degrade to CPU, not crash the run."""
        import ctranslate2

        def explode() -> int:
            raise RuntimeError("cudnn missing")

        monkeypatch.setattr(ctranslate2, "get_cuda_device_count", explode)
        assert resolve_device("auto") == ("cpu", "int8")
