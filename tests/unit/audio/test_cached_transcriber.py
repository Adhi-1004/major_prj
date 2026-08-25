"""Transcript caching.

Two properties matter. A hit must return exactly what was stored -- a transcript
that loses word probabilities is worse than no cache, because the confidence
score silently changes. And a hit must never be served across a configuration
change: same audio decoded by a different model is a different transcript.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dialogue_extractor.audio.cached_transcriber import (
    CachedTranscriber,
    transcript_from_dict,
    transcript_to_dict,
    whisper_fingerprint,
)
from dialogue_extractor.domain.models import Transcript, Word
from dialogue_extractor.infra.disk_cache import DiskCache
from dialogue_extractor.interfaces.ports import Transcriber


class CountingTranscriber(Transcriber):
    def __init__(self, transcript: Transcript) -> None:
        self.transcript = transcript
        self.calls = 0

    def transcribe(self, audio_path: Path) -> Transcript:
        self.calls += 1
        return self.transcript


@pytest.fixture
def transcript() -> Transcript:
    return Transcript(
        words=(
            Word("My", 0.0, 0.2, probability=0.91),
            Word("mind", 0.2, 0.5, probability=0.88, speaker_id="SPEAKER_00"),
            Word("rebels", 0.5, 0.9, probability=0.95),
        ),
        language="en",
        duration=1.2,
    )


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "audio.wav"
    path.write_bytes(b"audio content")
    return path


class TestRoundTrip:
    def test_preserves_every_field(self, transcript: Transcript) -> None:
        restored = transcript_from_dict(transcript_to_dict(transcript))
        assert restored == transcript

    def test_preserves_probabilities(self, transcript: Transcript) -> None:
        """Losing these would silently change the confidence score."""
        restored = transcript_from_dict(transcript_to_dict(transcript))
        assert [w.probability for w in restored.words] == [0.91, 0.88, 0.95]

    def test_preserves_speaker_ids(self, transcript: Transcript) -> None:
        restored = transcript_from_dict(transcript_to_dict(transcript))
        assert restored.words[1].speaker_id == "SPEAKER_00"
        assert restored.words[0].speaker_id is None

    def test_is_json_serializable(self, transcript: Transcript) -> None:
        import json

        assert json.loads(json.dumps(transcript_to_dict(transcript)))["language"] == "en"


class TestCaching:
    def test_first_call_transcribes(
        self, tmp_path: Path, audio_file: Path, transcript: Transcript
    ) -> None:
        inner = CountingTranscriber(transcript)
        cached = CachedTranscriber(inner, DiskCache(tmp_path / "cache"))
        assert cached.transcribe(audio_file) == transcript
        assert inner.calls == 1

    def test_second_call_hits_the_cache(
        self, tmp_path: Path, audio_file: Path, transcript: Transcript
    ) -> None:
        inner = CountingTranscriber(transcript)
        cached = CachedTranscriber(inner, DiskCache(tmp_path / "cache"))
        cached.transcribe(audio_file)
        result = cached.transcribe(audio_file)

        assert inner.calls == 1
        assert result == transcript

    def test_cache_survives_a_new_instance(
        self, tmp_path: Path, audio_file: Path, transcript: Transcript
    ) -> None:
        """The point is surviving across process runs during development."""
        cache_dir = tmp_path / "cache"
        first = CountingTranscriber(transcript)
        CachedTranscriber(first, DiskCache(cache_dir)).transcribe(audio_file)

        second = CountingTranscriber(transcript)
        CachedTranscriber(second, DiskCache(cache_dir)).transcribe(audio_file)
        assert second.calls == 0

    def test_different_audio_is_a_miss(
        self, tmp_path: Path, audio_file: Path, transcript: Transcript
    ) -> None:
        other = tmp_path / "other.wav"
        other.write_bytes(b"completely different audio")

        inner = CountingTranscriber(transcript)
        cached = CachedTranscriber(inner, DiskCache(tmp_path / "cache"))
        cached.transcribe(audio_file)
        cached.transcribe(other)
        assert inner.calls == 2

    def test_identical_content_at_a_different_path_hits(
        self, tmp_path: Path, audio_file: Path, transcript: Transcript
    ) -> None:
        """Keyed by content, not filename."""
        copy = tmp_path / "renamed.wav"
        copy.write_bytes(audio_file.read_bytes())

        inner = CountingTranscriber(transcript)
        cached = CachedTranscriber(inner, DiskCache(tmp_path / "cache"))
        cached.transcribe(audio_file)
        cached.transcribe(copy)
        assert inner.calls == 1


class TestFingerprintInvalidation:
    def test_different_model_is_a_miss(
        self, tmp_path: Path, audio_file: Path, transcript: Transcript
    ) -> None:
        """A large-v3 transcript must never be served to a medium run."""
        cache_dir = tmp_path / "cache"
        first = CountingTranscriber(transcript)
        CachedTranscriber(
            first, DiskCache(cache_dir), fingerprint=whisper_fingerprint("large-v3", "en", 5)
        ).transcribe(audio_file)

        second = CountingTranscriber(transcript)
        CachedTranscriber(
            second, DiskCache(cache_dir), fingerprint=whisper_fingerprint("medium", "en", 5)
        ).transcribe(audio_file)
        assert second.calls == 1, "a model change must invalidate the cache"

    def test_different_language_is_a_miss(
        self, tmp_path: Path, audio_file: Path, transcript: Transcript
    ) -> None:
        cache_dir = tmp_path / "cache"
        a = CountingTranscriber(transcript)
        CachedTranscriber(
            a, DiskCache(cache_dir), fingerprint=whisper_fingerprint("large-v3", "en", 5)
        ).transcribe(audio_file)

        b = CountingTranscriber(transcript)
        CachedTranscriber(
            b, DiskCache(cache_dir), fingerprint=whisper_fingerprint("large-v3", "fr", 5)
        ).transcribe(audio_file)
        assert b.calls == 1

    def test_different_beam_size_is_a_miss(
        self, tmp_path: Path, audio_file: Path, transcript: Transcript
    ) -> None:
        cache_dir = tmp_path / "cache"
        a = CountingTranscriber(transcript)
        CachedTranscriber(
            a, DiskCache(cache_dir), fingerprint=whisper_fingerprint("large-v3", "en", 5)
        ).transcribe(audio_file)

        b = CountingTranscriber(transcript)
        CachedTranscriber(
            b, DiskCache(cache_dir), fingerprint=whisper_fingerprint("large-v3", "en", 1)
        ).transcribe(audio_file)
        assert b.calls == 1

    def test_same_fingerprint_hits(
        self, tmp_path: Path, audio_file: Path, transcript: Transcript
    ) -> None:
        cache_dir = tmp_path / "cache"
        fp = whisper_fingerprint("large-v3", "en", 5)
        a = CountingTranscriber(transcript)
        CachedTranscriber(a, DiskCache(cache_dir), fingerprint=fp).transcribe(audio_file)

        b = CountingTranscriber(transcript)
        CachedTranscriber(b, DiskCache(cache_dir), fingerprint=fp).transcribe(audio_file)
        assert b.calls == 0


class TestCorruptEntries:
    def test_unreadable_entry_falls_back_to_transcribing(
        self, tmp_path: Path, audio_file: Path, transcript: Transcript
    ) -> None:
        """A cache is an optimization; a bad entry must never fail the run."""
        cache = DiskCache(tmp_path / "cache")
        inner = CountingTranscriber(transcript)
        cached = CachedTranscriber(inner, cache)

        cache.put(cached.cache_key(audio_file), {"unexpected": "shape"})
        assert cached.transcribe(audio_file) == transcript
        assert inner.calls == 1

    @pytest.mark.parametrize(
        "payload",
        [
            {"words": [{"text": "x"}], "language": "en", "duration": 1.0},  # no timings
            {"language": "en", "duration": 1.0},  # no words key
            {"words": "not a list", "language": "en", "duration": 1.0},
        ],
    )
    def test_malformed_payloads_are_misses(
        self, tmp_path: Path, audio_file: Path, transcript: Transcript, payload: dict
    ) -> None:
        cache = DiskCache(tmp_path / "cache")
        inner = CountingTranscriber(transcript)
        cached = CachedTranscriber(inner, cache)

        cache.put(cached.cache_key(audio_file), payload)
        assert cached.transcribe(audio_file) == transcript


class TestDisabledCache:
    def test_null_cache_always_transcribes(
        self, tmp_path: Path, audio_file: Path, transcript: Transcript
    ) -> None:
        from dialogue_extractor.infra.disk_cache import NullCache

        inner = CountingTranscriber(transcript)
        cached = CachedTranscriber(inner, NullCache())
        cached.transcribe(audio_file)
        cached.transcribe(audio_file)
        assert inner.calls == 2


def test_satisfies_the_transcriber_port(tmp_path: Path, transcript: Transcript) -> None:
    """The pipeline must not be able to tell it is talking to a cache."""
    cached = CachedTranscriber(CountingTranscriber(transcript), DiskCache(tmp_path))
    assert isinstance(cached, Transcriber)
