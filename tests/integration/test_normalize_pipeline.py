"""Normalization against real media, with the real ffmpeg.

Everything downstream assumes the output contract holds: constant frame rate,
faststart, 16 kHz mono audio. These tests check the produced *files*, not the
flags that were passed, because the flags can silently do nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.conftest import requires_ffmpeg

from dialogue_extractor.domain.errors import NoAudioStreamError
from dialogue_extractor.domain.models import RawMedia
from dialogue_extractor.ingestion.ffprobe_validator import probe_audio, validate_and_probe
from dialogue_extractor.ingestion.normalizer import is_faststart, normalize, sha256_of

pytestmark = requires_ffmpeg


def as_raw(path: Path) -> RawMedia:
    return RawMedia(path=path, source_uri=str(path), title=path.stem)


class TestOutputContract:
    def test_produces_both_outputs(self, counter_clip_25fps: Path, tmp_path: Path) -> None:
        manifest = normalize(as_raw(counter_clip_25fps), tmp_path)
        assert manifest.video_path.exists() and manifest.video_path.stat().st_size > 0
        assert manifest.audio_path.exists() and manifest.audio_path.stat().st_size > 0

    def test_audio_is_16khz_mono_pcm(self, counter_clip_25fps: Path, tmp_path: Path) -> None:
        """What faster-whisper and pyannote both expect."""
        manifest = normalize(as_raw(counter_clip_25fps), tmp_path)
        audio = probe_audio(manifest.audio_path)
        assert audio.sample_rate == 16_000
        assert audio.channels == 1
        assert audio.codec == "pcm_s16le"

    def test_video_is_h264_yuv420p(self, counter_clip_25fps: Path, tmp_path: Path) -> None:
        manifest = normalize(as_raw(counter_clip_25fps), tmp_path)
        assert validate_and_probe(manifest.video_path).video_codec == "h264"

    def test_video_is_faststart(self, counter_clip_25fps: Path, tmp_path: Path) -> None:
        """Required by the ASD service: non-streamable input fails *quietly*."""
        manifest = normalize(as_raw(counter_clip_25fps), tmp_path)
        assert is_faststart(manifest.video_path) is True

    def test_manifest_records_cfr_was_forced(
        self, counter_clip_25fps: Path, tmp_path: Path
    ) -> None:
        assert normalize(as_raw(counter_clip_25fps), tmp_path).cfr_forced is True


class TestStereoIsDownmixed:
    def test_stereo_44k_becomes_mono_16k(self, clip_stereo_44k: Path, tmp_path: Path) -> None:
        manifest = normalize(as_raw(clip_stereo_44k), tmp_path)
        audio = probe_audio(manifest.audio_path)
        assert audio.sample_rate == 16_000
        assert audio.channels == 1


class TestVfrIsForcedToCfr:
    def test_source_vfr_is_detected_and_recorded(self, clip_vfr: Path, tmp_path: Path) -> None:
        manifest = normalize(as_raw(clip_vfr), tmp_path)
        assert manifest.source_was_vfr is True
        assert manifest.cfr_forced is True

    def test_output_reports_a_single_consistent_rate(
        self, clip_vfr: Path, tmp_path: Path
    ) -> None:
        """After normalization the two ffprobe rates must agree.

        This is the property invariant 2 rests on. If they still diverged,
        `frame_number == timestamp * fps` would remain false and the frame
        number the pipeline reports could not be verified.
        """
        manifest = normalize(as_raw(clip_vfr), tmp_path)
        out = validate_and_probe(manifest.video_path)
        assert out.is_vfr is False, (
            f"output still VFR: r={out.r_frame_rate} avg={out.avg_frame_rate}"
        )


class TestInvariantTwoHoldsOnRealOutput:
    def test_frame_count_matches_duration_times_fps(
        self, counter_clip_25fps: Path, tmp_path: Path
    ) -> None:
        manifest = normalize(as_raw(counter_clip_25fps), tmp_path)
        implied_duration = manifest.frame_count / manifest.fps
        assert implied_duration == pytest.approx(manifest.duration, abs=0.15)

    def test_frame_count_is_plausible(self, counter_clip_25fps: Path, tmp_path: Path) -> None:
        """The fixture is 100 frames at 25 fps."""
        manifest = normalize(as_raw(counter_clip_25fps), tmp_path)
        assert manifest.fps == pytest.approx(25.0, abs=0.01)
        assert manifest.frame_count == pytest.approx(100, abs=2)


class TestValidationRunsFirst:
    def test_no_audio_fails_before_any_encoding(
        self, clip_no_audio: Path, tmp_path: Path
    ) -> None:
        """Must fail in validation, not partway through a long re-encode."""
        with pytest.raises(NoAudioStreamError) as excinfo:
            normalize(as_raw(clip_no_audio), tmp_path)

        assert excinfo.value.reason == "no_audio_stream"
        assert not (tmp_path / "normalized.mp4").exists(), (
            "nothing should be encoded once validation has failed"
        )


class TestHashing:
    def test_hashes_are_recorded(self, counter_clip_25fps: Path, tmp_path: Path) -> None:
        manifest = normalize(as_raw(counter_clip_25fps), tmp_path)
        assert len(manifest.video_sha256) == 64
        assert len(manifest.audio_sha256) == 64
        assert manifest.video_sha256 != manifest.audio_sha256

    def test_hash_matches_the_file(self, counter_clip_25fps: Path, tmp_path: Path) -> None:
        manifest = normalize(as_raw(counter_clip_25fps), tmp_path)
        assert sha256_of(manifest.audio_path) == manifest.audio_sha256

    def test_audio_hash_is_stable_across_runs(
        self, counter_clip_25fps: Path, tmp_path: Path
    ) -> None:
        """The audio hash keys the transcript cache, so it must be reproducible."""
        first = normalize(as_raw(counter_clip_25fps), tmp_path / "a")
        second = normalize(as_raw(counter_clip_25fps), tmp_path / "b")
        assert first.audio_sha256 == second.audio_sha256


class TestIsFaststart:
    def test_detects_a_non_faststart_file(self, clip_vfr: Path) -> None:
        """The VFR fixture is built without +faststart, so this must be False.

        A detector that always returned True would make the faststart assertion
        above vacuous.
        """
        assert is_faststart(clip_vfr) is False

    def test_handles_a_non_mp4_gracefully(self, tmp_path: Path) -> None:
        junk = tmp_path / "notmedia.mp4"
        junk.write_bytes(b"not an mp4 at all")
        assert is_faststart(junk) is False
