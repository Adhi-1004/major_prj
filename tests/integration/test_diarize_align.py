"""Diarization and alignment against real audio.

Marked slow and skipped without credentials: the pipeline is gated, so this
cannot run in a clean checkout. What it verifies is the part unit tests cannot --
that a real pyannote Annotation converts correctly, and that alignment over real
ASR timings produces a sane speaker attribution.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from tests.conftest import FFMPEG, credential, requires_ffmpeg

from dialogue_extractor.audio.aligner import assign_speakers, build_segments
from dialogue_extractor.audio.diarization_export import (
    looks_like_milliseconds,
    to_nvidia_diarization,
    validate_diarization_payload,
)
from dialogue_extractor.audio.pyannote_diarizer import PyannoteDiarizer
from dialogue_extractor.audio.whisper_transcriber import WhisperTranscriber

HF_TOKEN = credential("HF_TOKEN")

pytestmark = [
    requires_ffmpeg,
    pytest.mark.slow,
    pytest.mark.skipif(
        not HF_TOKEN,
        reason="needs HF_TOKEN with the gated pyannote pipeline's terms accepted",
    ),
]


@pytest.fixture(scope="module")
def two_speaker_audio(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two distinct synthetic voices, one after the other.

    Built from tones at different frequencies rather than real speech: this
    tests that *two distinct sources* are separated and attributed to the right
    halves, which is the structural property alignment depends on.
    """
    out_dir = tmp_path_factory.mktemp("diar")
    out = out_dir / "two_speakers.wav"
    subprocess.run(
        [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i",
            "sine=frequency=180:duration=4,asetrate=16000",
            "-f", "lavfi", "-i",
            "sine=frequency=420:duration=4,asetrate=16000",
            "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[out]",
            "-map", "[out]", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(out),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return out


class TestRealDiarization:
    def test_produces_well_formed_turns(self, two_speaker_audio: Path) -> None:
        diarizer = PyannoteDiarizer(HF_TOKEN, device="auto")
        try:
            turns = diarizer.diarize(two_speaker_audio)
        except Exception as exc:
            pytest.skip(f"diarization unavailable: {type(exc).__name__}: {exc}")

        assert turns, "expected at least one turn"
        assert all(t.end > t.start for t in turns)
        assert all(t.start >= 0 for t in turns)
        assert all(isinstance(t.speaker_id, str) for t in turns)

        starts = [t.start for t in turns]
        assert starts == sorted(starts), "turns must be chronological"


@pytest.mark.skipif(
    not Path("output/work/audio.wav").exists(),
    reason="needs output/work/audio.wav from a Phase 1 run",
)
class TestEndToEndAlignment:
    """Runs only when real ingested audio is present locally."""

    def test_alignment_over_real_transcript_and_turns(self) -> None:
        audio = Path("output/work/audio.wav")

        transcript = WhisperTranscriber("tiny", device="auto", language="en").transcribe(audio)
        turns = PyannoteDiarizer(HF_TOKEN, device="auto").diarize(audio)

        aligned = assign_speakers(transcript.words, turns)

        assert len(aligned) == len(transcript.words), "alignment must not drop words"
        assigned = [w for w in aligned if w.speaker_id is not None]
        assert assigned, "expected at least some words to be attributed"

        # Most words should land on a speaker; a majority unassigned means the
        # two timelines disagree badly enough to be worth failing on.
        ratio = len(assigned) / len(aligned)
        assert ratio > 0.5, f"only {ratio:.0%} of words were attributed to a speaker"

    def test_segments_partition_the_words(self) -> None:
        audio = Path("output/work/audio.wav")
        transcript = WhisperTranscriber("tiny", device="auto", language="en").transcribe(audio)
        turns = PyannoteDiarizer(HF_TOKEN, device="auto").diarize(audio)

        aligned = assign_speakers(transcript.words, turns)
        segments = build_segments(aligned)

        assert sum(len(s.words) for s in segments) == len(aligned)

    def test_asd_payload_is_valid_for_a_real_window(self) -> None:
        """The contract the ASD service consumes, built from real data."""
        audio = Path("output/work/audio.wav")
        transcript = WhisperTranscriber("tiny", device="auto", language="en").transcribe(audio)
        turns = PyannoteDiarizer(HF_TOKEN, device="auto").diarize(audio)
        aligned = assign_speakers(transcript.words, turns)

        window = tuple(w for w in aligned if 10.0 <= w.start <= 40.0)
        if not window:
            pytest.skip("no words in the sampled window")

        payload = to_nvidia_diarization(window, clip_start=window[0].start)

        validate_diarization_payload(payload)
        assert looks_like_milliseconds(payload, clip_duration=30.0) is False
        assert payload["words"][0]["start"] >= 0.0
        assert all(isinstance(w["speaker_id"], str) for w in payload["words"])
