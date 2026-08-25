"""Transcription against real audio with the real model.

Marked slow: it downloads model weights on first run and takes real time. The
assertions are deliberately loose about *wording* and strict about *structure* --
ASR output varies between models and versions, so pinning an exact transcript
would make this a change-detector rather than a test. What must hold is that
words come back in order, with sane timings inside the clip, and with
probabilities the confidence formula can use.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from tests.conftest import FFMPEG, requires_ffmpeg

from dialogue_extractor.audio.whisper_transcriber import WhisperTranscriber
from dialogue_extractor.domain.errors import TranscriptionError

pytestmark = [requires_ffmpeg, pytest.mark.slow]

MODEL = "tiny"
"""Smallest model. These tests check plumbing, not accuracy."""


@pytest.fixture(scope="module")
def silent_audio(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("audio") / "silence.wav"
    subprocess.run(
        [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
            "-t", "3", "-c:a", "pcm_s16le", str(out),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return out


@pytest.fixture(scope="module")
def tone_audio(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tone, not speech. Must not be transcribed into confident words."""
    out = tmp_path_factory.mktemp("audio") / "tone.wav"
    subprocess.run(
        [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000",
            "-t", "3", "-ac", "1", "-c:a", "pcm_s16le", str(out),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return out


class TestRealAudio:
    def test_silence_reports_no_speech_rather_than_inventing_words(
        self, silent_audio: Path
    ) -> None:
        """The failure must be specific. An empty transcript returned as success
        would later be indistinguishable from "the phrase is not in this video"."""
        transcriber = WhisperTranscriber(MODEL, device="cpu", language="en")
        with pytest.raises(TranscriptionError) as excinfo:
            transcriber.transcribe(silent_audio)
        assert excinfo.value.reason == "no_speech_detected"

    def test_tone_does_not_produce_a_confident_transcript(self, tone_audio: Path) -> None:
        """VAD should reject a pure tone. If it does produce words, they must at
        least be flagged low-probability rather than passed off as dialogue."""
        transcriber = WhisperTranscriber(MODEL, device="cpu", language="en")
        try:
            transcript = transcriber.transcribe(tone_audio)
        except TranscriptionError as exc:
            assert exc.reason == "no_speech_detected"
            return
        mean_prob = sum(w.probability for w in transcript.words) / len(transcript.words)
        assert mean_prob < 0.85, f"non-speech transcribed confidently: {transcript.text!r}"


@pytest.mark.skipif(
    not Path("output/work/audio.wav").exists(),
    reason="needs output/work/audio.wav from a Phase 1 run",
)
class TestAgainstIngestedAudio:
    """Runs only when a real Phase 1 ingestion has been done locally.

    Skipped in a clean checkout, so it never blocks CI, but exercises the real
    end-to-end shape when the artefacts are present.
    """

    def test_produces_ordered_words_with_usable_timings(self) -> None:
        audio = Path("output/work/audio.wav")
        transcriber = WhisperTranscriber(MODEL, device="auto", language="en")
        transcript = transcriber.transcribe(audio)

        assert transcript.words, "expected speech in the ingested audio"

        starts = [w.start for w in transcript.words]
        assert starts == sorted(starts), "words must be in chronological order"

        assert all(w.end >= w.start for w in transcript.words)
        assert all(0.0 <= w.probability <= 1.0 for w in transcript.words)

        # Timings must fall inside the clip, with a little slack for the final word.
        assert transcript.words[0].start >= 0.0
        assert transcript.words[-1].end <= transcript.duration + 1.0

    def test_words_carry_varied_probabilities(self) -> None:
        """All-1.0 would mean probabilities were not really captured, which
        silently flattens one term of the confidence score."""
        transcriber = WhisperTranscriber(MODEL, device="auto", language="en")
        transcript = transcriber.transcribe(Path("output/work/audio.wav"))
        assert len({round(w.probability, 3) for w in transcript.words}) > 1
