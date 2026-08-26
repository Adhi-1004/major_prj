"""Caching the download and normalization together.

Written after a real failure. Every run re-downloaded the video, and four or
five runs through the UI in quick succession was enough for YouTube to start
answering "Sign in to confirm you're not a bot" -- a block that then persists.
The pipeline was not at fault; asking for the same file over and over was.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dialogue_extractor.domain.models import MediaManifest
from dialogue_extractor.infra.disk_cache import DiskCache, NullCache
from dialogue_extractor.ingestion.media_cache import (
    canonical_url,
    lookup,
    media_fingerprint,
    media_key,
    store,
)

FINGERPRINT = media_fingerprint(max_height=1080, target_audio_rate=16000)
URL = "https://youtu.be/CoBS_v70xbE"


@pytest.fixture
def normalized(tmp_path: Path) -> MediaManifest:
    work = tmp_path / "work"
    work.mkdir()
    (work / "normalized.mp4").write_bytes(b"VIDEO")
    (work / "audio.wav").write_bytes(b"AUDIO")
    return MediaManifest(
        video_path=work / "normalized.mp4",
        audio_path=work / "audio.wav",
        fps=23.976024,
        frame_count=14636,
        duration=610.499,
        width=1920,
        height=1080,
        cfr_forced=True,
        source_was_vfr=False,
        video_sha256="v",
        audio_sha256="a",
    )


class TestRoundTrip:
    def test_a_cold_lookup_misses(self, tmp_path: Path) -> None:
        cache = DiskCache(tmp_path / "c")
        assert lookup(cache, URL, FINGERPRINT, tmp_path / "w") is None

    def test_a_stored_entry_is_found_from_a_fresh_directory(
        self, tmp_path: Path, normalized: MediaManifest
    ) -> None:
        """The second run has its own working directory and must still hit."""
        cache = DiskCache(tmp_path / "c")
        store(cache, URL, FINGERPRINT, normalized)

        hit = lookup(cache, URL, FINGERPRINT, tmp_path / "elsewhere")
        assert hit is not None
        assert hit.video_path.read_bytes() == b"VIDEO"
        assert hit.audio_path.read_bytes() == b"AUDIO"

    def test_every_measured_property_survives(
        self, tmp_path: Path, normalized: MediaManifest
    ) -> None:
        """Invariant 2 depends on fps and cfr_forced, so they must round-trip
        exactly rather than approximately."""
        cache = DiskCache(tmp_path / "c")
        store(cache, URL, FINGERPRINT, normalized)
        hit = lookup(cache, URL, FINGERPRINT, tmp_path / "w2")

        assert hit is not None
        assert hit.fps == normalized.fps
        assert hit.frame_count == normalized.frame_count
        assert hit.duration == normalized.duration
        assert hit.resolution == normalized.resolution
        assert hit.cfr_forced is True

    def test_files_land_where_every_later_stage_expects_them(
        self, tmp_path: Path, normalized: MediaManifest
    ) -> None:
        cache = DiskCache(tmp_path / "c")
        store(cache, URL, FINGERPRINT, normalized)
        work = tmp_path / "w2"

        hit = lookup(cache, URL, FINGERPRINT, work)
        assert hit is not None
        assert hit.video_path == work / "normalized.mp4"
        assert hit.audio_path == work / "audio.wav"


class TestKeying:
    @pytest.mark.parametrize(
        "spelling",
        [
            "https://youtu.be/CoBS_v70xbE",
            "https://www.youtube.com/watch?v=CoBS_v70xbE",
            "https://youtube.com/watch?v=CoBS_v70xbE",
            "https://youtu.be/CoBS_v70xbE?si=-oWWlTJ3g3rpw5El",
            "https://m.youtube.com/watch?v=CoBS_v70xbE&feature=share",
            "  https://youtu.be/CoBS_v70xbE  ",
        ],
    )
    def test_one_video_is_one_key_however_it_is_written(self, spelling: str) -> None:
        """Keying on the raw string stored a separate 286 MB copy per spelling,
        and made a pasted share link re-download a video already cached."""
        assert media_key(spelling, FINGERPRINT) == media_key(URL, FINGERPRINT)

    def test_a_different_video_is_a_different_key(self) -> None:
        assert media_key("https://youtu.be/OTHER", FINGERPRINT) != media_key(
            URL, FINGERPRINT
        )

    def test_different_normalization_settings_do_not_share_an_entry(self) -> None:
        """Media normalized to 720p is not the same media."""
        other = media_fingerprint(max_height=720, target_audio_rate=16000)
        assert media_key(URL, other) != media_key(URL, FINGERPRINT)

    @pytest.mark.parametrize(
        "uri", ["./clip.mp4", "C:/videos/clip.mp4", "/tmp/clip.mp4"]
    )
    def test_local_paths_pass_through_untouched(self, uri: str) -> None:
        assert canonical_url(uri) == uri

    def test_a_non_youtube_host_keeps_its_path(self) -> None:
        """Only YouTube's two host forms are unified; nothing else is guessed."""
        assert (
            canonical_url("https://ok.ru/video/248244667877")
            == "https://ok.ru/video/248244667877"
        )

    def test_two_different_ok_ru_videos_stay_distinct(self) -> None:
        a = canonical_url("https://ok.ru/video/111")
        b = canonical_url("https://ok.ru/video/222")
        assert a != b


class TestDegradesSafely:
    """A cache problem must never fail a run that could still be done slowly."""

    def test_metadata_without_media_is_a_miss(
        self, tmp_path: Path, normalized: MediaManifest
    ) -> None:
        cache = DiskCache(tmp_path / "c")
        store(cache, URL, FINGERPRINT, normalized)
        cache.path_for(media_key(URL, FINGERPRINT), ".mp4").unlink()

        assert lookup(cache, URL, FINGERPRINT, tmp_path / "w2") is None

    def test_missing_audio_is_a_miss(
        self, tmp_path: Path, normalized: MediaManifest
    ) -> None:
        cache = DiskCache(tmp_path / "c")
        store(cache, URL, FINGERPRINT, normalized)
        cache.path_for(media_key(URL, FINGERPRINT), ".wav").unlink()

        assert lookup(cache, URL, FINGERPRINT, tmp_path / "w2") is None

    def test_a_corrupt_entry_is_a_miss(self, tmp_path: Path) -> None:
        cache = DiskCache(tmp_path / "c")
        cache.put(media_key(URL, FINGERPRINT), "not a manifest")
        assert lookup(cache, URL, FINGERPRINT, tmp_path / "w2") is None

    def test_a_cache_that_keeps_nothing_is_harmless(
        self, tmp_path: Path, normalized: MediaManifest
    ) -> None:
        """--no-cache must still run, just without the shortcut."""
        cache = NullCache()
        store(cache, URL, FINGERPRINT, normalized)
        assert lookup(cache, URL, FINGERPRINT, tmp_path / "w2") is None
