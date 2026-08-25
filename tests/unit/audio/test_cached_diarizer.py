"""Diarization caching. Mirrors the transcript cache; same invariants."""

from __future__ import annotations

from pathlib import Path

import pytest

from dialogue_extractor.audio.cached_diarizer import CachedDiarizer, pyannote_fingerprint
from dialogue_extractor.domain.models import SpeakerTurn
from dialogue_extractor.infra.disk_cache import DiskCache, NullCache
from dialogue_extractor.interfaces.ports import Diarizer

TURNS = (
    SpeakerTurn("SPEAKER_00", 0.0, 5.0),
    SpeakerTurn("SPEAKER_01", 5.2, 9.0),
)


class CountingDiarizer(Diarizer):
    def __init__(self, turns: tuple[SpeakerTurn, ...] = TURNS) -> None:
        self.turns = turns
        self.calls = 0

    def diarize(self, audio_path: Path) -> tuple[SpeakerTurn, ...]:
        self.calls += 1
        return self.turns


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "audio.wav"
    path.write_bytes(b"audio content")
    return path


def test_first_call_diarizes(tmp_path: Path, audio_file: Path) -> None:
    inner = CountingDiarizer()
    cached = CachedDiarizer(inner, DiskCache(tmp_path / "cache"))
    assert cached.diarize(audio_file) == TURNS
    assert inner.calls == 1


def test_second_call_hits_the_cache(tmp_path: Path, audio_file: Path) -> None:
    inner = CountingDiarizer()
    cached = CachedDiarizer(inner, DiskCache(tmp_path / "cache"))
    cached.diarize(audio_file)
    assert cached.diarize(audio_file) == TURNS
    assert inner.calls == 1


def test_round_trip_preserves_turns(tmp_path: Path, audio_file: Path) -> None:
    cached = CachedDiarizer(CountingDiarizer(), DiskCache(tmp_path / "cache"))
    cached.diarize(audio_file)
    assert cached.diarize(audio_file) == TURNS


def test_speaker_count_hint_invalidates(tmp_path: Path, audio_file: Path) -> None:
    """Pinning the speaker count changes the segmentation, so a run that pinned
    it must not be served to one that let pyannote estimate."""
    cache_dir = tmp_path / "cache"
    a = CountingDiarizer()
    CachedDiarizer(
        a, DiskCache(cache_dir), fingerprint=pyannote_fingerprint("p", num_speakers=2)
    ).diarize(audio_file)

    b = CountingDiarizer()
    CachedDiarizer(
        b, DiskCache(cache_dir), fingerprint=pyannote_fingerprint("p", num_speakers=None)
    ).diarize(audio_file)
    assert b.calls == 1


def test_pipeline_change_invalidates(tmp_path: Path, audio_file: Path) -> None:
    cache_dir = tmp_path / "cache"
    a = CountingDiarizer()
    CachedDiarizer(a, DiskCache(cache_dir), fingerprint=pyannote_fingerprint("v1")).diarize(
        audio_file
    )
    b = CountingDiarizer()
    CachedDiarizer(b, DiskCache(cache_dir), fingerprint=pyannote_fingerprint("v2")).diarize(
        audio_file
    )
    assert b.calls == 1


def test_different_audio_is_a_miss(tmp_path: Path, audio_file: Path) -> None:
    other = tmp_path / "other.wav"
    other.write_bytes(b"different audio")
    inner = CountingDiarizer()
    cached = CachedDiarizer(inner, DiskCache(tmp_path / "cache"))
    cached.diarize(audio_file)
    cached.diarize(other)
    assert inner.calls == 2


def test_corrupt_entry_is_a_miss(tmp_path: Path, audio_file: Path) -> None:
    cache = DiskCache(tmp_path / "cache")
    inner = CountingDiarizer()
    cached = CachedDiarizer(inner, cache)
    cache.put(cached.cache_key(audio_file), [{"wrong": "shape"}])
    assert cached.diarize(audio_file) == TURNS
    assert inner.calls == 1


def test_empty_cached_result_is_not_served(tmp_path: Path, audio_file: Path) -> None:
    """An empty turn list is indistinguishable from a failed run; re-diarize."""
    cache = DiskCache(tmp_path / "cache")
    inner = CountingDiarizer()
    cached = CachedDiarizer(inner, cache)
    cache.put(cached.cache_key(audio_file), [])
    assert cached.diarize(audio_file) == TURNS


def test_null_cache_always_diarizes(tmp_path: Path, audio_file: Path) -> None:
    inner = CountingDiarizer()
    cached = CachedDiarizer(inner, NullCache())
    cached.diarize(audio_file)
    cached.diarize(audio_file)
    assert inner.calls == 2


def test_satisfies_the_diarizer_port(tmp_path: Path) -> None:
    assert isinstance(CachedDiarizer(CountingDiarizer(), DiskCache(tmp_path)), Diarizer)


class TestFingerprintCoversEveryOutputAffectingParameter:
    """A config change that alters the stored turns must invalidate the cache.

    The regression: the short-turn filter runs during extraction, so the cache
    stores *filtered* turns -- but the threshold was not in the key. On any
    machine that had already diarized, tuning it changed nothing, and the stage
    kept serving the old unfiltered attribution while reporting success.
    """

    def test_min_turn_seconds_is_part_of_the_key(self) -> None:
        assert pyannote_fingerprint("p", min_turn_seconds=0.25) != pyannote_fingerprint(
            "p", min_turn_seconds=0.40
        )

    def test_changing_the_filter_invalidates_a_cached_result(
        self, tmp_path: Path, audio_file: Path
    ) -> None:
        cache_dir = tmp_path / "cache"

        first = CountingDiarizer()
        CachedDiarizer(
            first, DiskCache(cache_dir), fingerprint=pyannote_fingerprint("p", min_turn_seconds=0.0)
        ).diarize(audio_file)

        second = CountingDiarizer()
        CachedDiarizer(
            second,
            DiskCache(cache_dir),
            fingerprint=pyannote_fingerprint("p", min_turn_seconds=0.25),
        ).diarize(audio_file)

        assert second.calls == 1, "a filter change must not be served from cache"

    def test_default_fingerprint_tracks_the_configured_constant(self) -> None:
        """Guards against the constant and the key drifting apart."""
        from dialogue_extractor.config import constants

        assert f"min_turn={constants.MIN_SPEAKER_TURN_SECONDS}" in pyannote_fingerprint("p")
