"""Media validation, parsed from fixed ffprobe payloads.

Driving these from recorded JSON rather than real files keeps the awkward cases
-- VFR, NTSC fractional rates, a missing audio track -- cheap to express and
deterministic to assert.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dialogue_extractor.domain.errors import (
    CorruptMediaError,
    NoAudioStreamError,
    NoVideoStreamError,
)
from dialogue_extractor.ingestion import ffprobe_validator
from dialogue_extractor.ingestion.ffprobe_validator import parse_frame_rate, validate_and_probe


def video_stream(**overrides: Any) -> dict[str, Any]:
    base = {
        "codec_type": "video",
        "codec_name": "h264",
        "width": 1920,
        "height": 1080,
        "r_frame_rate": "25/1",
        "avg_frame_rate": "25/1",
        "duration": "120.0",
        "nb_frames": "3000",
    }
    base.update(overrides)
    return base


def audio_stream(**overrides: Any) -> dict[str, Any]:
    base = {
        "codec_type": "audio",
        "codec_name": "aac",
        "sample_rate": "48000",
        "channels": 2,
    }
    base.update(overrides)
    return base


def probe_payload(streams: list[dict[str, Any]], duration: str = "120.0") -> dict[str, Any]:
    return {"streams": streams, "format": {"duration": duration}}


@pytest.fixture
def media_file(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00" * 1024)
    return path


@pytest.fixture
def stub_probe(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Replace the ffprobe call with a canned payload."""

    def install(payload: dict[str, Any]) -> None:
        monkeypatch.setattr(
            ffprobe_validator, "probe_json", lambda path, binary=None: payload
        )

    return install


class TestParseFrameRate:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("25/1", 25.0),
            ("30/1", 30.0),
            ("24000/1001", 23.976023976),  # NTSC film
            ("30000/1001", 29.97002997),  # NTSC video
            ("60/1", 60.0),
            ("25", 25.0),
        ],
    )
    def test_parses_fractions_exactly(self, raw: str, expected: float) -> None:
        """Parsed as fractions, not pre-rounded decimals, so 23.976 stays exact."""
        assert parse_frame_rate(raw) == pytest.approx(expected, rel=1e-9)

    @pytest.mark.parametrize("raw", [None, "", "0/0", "0/1", "garbage", "N/A"])
    def test_returns_zero_for_unusable_values(self, raw: str | None) -> None:
        """Zero rather than raising, so the caller can fall back."""
        assert parse_frame_rate(raw) == 0.0


class TestStreamRequirements:
    def test_accepts_a_normal_file(self, media_file: Path, stub_probe) -> None:  # type: ignore[no-untyped-def]
        stub_probe(probe_payload([video_stream(), audio_stream()]))
        probe = validate_and_probe(media_file)
        assert probe.width == 1920
        assert probe.video_codec == "h264"
        assert probe.audio_codec == "aac"
        assert probe.audio_sample_rate == 48000
        assert probe.audio_channels == 2

    def test_rejects_audio_only(self, media_file: Path, stub_probe) -> None:  # type: ignore[no-untyped-def]
        stub_probe(probe_payload([audio_stream()]))
        with pytest.raises(NoVideoStreamError) as excinfo:
            validate_and_probe(media_file)
        assert excinfo.value.reason == "no_video_stream"

    def test_rejects_video_with_no_audio(self, media_file: Path, stub_probe) -> None:  # type: ignore[no-untyped-def]
        """Must fail here, before the transcriber is ever reached."""
        stub_probe(probe_payload([video_stream()]))
        with pytest.raises(NoAudioStreamError) as excinfo:
            validate_and_probe(media_file)
        assert excinfo.value.reason == "no_audio_stream"
        assert excinfo.value.to_dict()["stage"] == "validation"

    def test_audio_can_be_optional(self, media_file: Path, stub_probe) -> None:  # type: ignore[no-untyped-def]
        stub_probe(probe_payload([video_stream()]))
        probe = validate_and_probe(media_file, require_audio=False)
        assert probe.audio_codec == "none"

    def test_rejects_empty_stream_list(self, media_file: Path, stub_probe) -> None:  # type: ignore[no-untyped-def]
        stub_probe(probe_payload([]))
        with pytest.raises(CorruptMediaError) as excinfo:
            validate_and_probe(media_file)
        assert excinfo.value.reason == "no_streams"


class TestVfrDetection:
    def test_matching_rates_are_cfr(self, media_file: Path, stub_probe) -> None:  # type: ignore[no-untyped-def]
        stub_probe(probe_payload([video_stream(), audio_stream()]))
        assert validate_and_probe(media_file).is_vfr is False

    def test_diverging_rates_are_vfr(self, media_file: Path, stub_probe) -> None:  # type: ignore[no-untyped-def]
        """The whole reason normalization forces CFR rather than measuring."""
        stub_probe(
            probe_payload(
                [video_stream(r_frame_rate="30/1", avg_frame_rate="10/1"), audio_stream()]
            )
        )
        probe = validate_and_probe(media_file)
        assert probe.is_vfr is True
        # fps prefers the average -- what is actually present, not the nominal.
        assert probe.fps == pytest.approx(10.0)

    def test_ntsc_rounding_is_not_mistaken_for_vfr(
        self, media_file: Path, stub_probe
    ) -> None:  # type: ignore[no-untyped-def]
        """23.976 vs 23.98 differ by rounding, not by being variable."""
        stub_probe(
            probe_payload(
                [
                    video_stream(r_frame_rate="24000/1001", avg_frame_rate="23.976"),
                    audio_stream(),
                ]
            )
        )
        assert validate_and_probe(media_file).is_vfr is False

    def test_missing_rate_is_not_reported_as_vfr(
        self, media_file: Path, stub_probe
    ) -> None:  # type: ignore[no-untyped-def]
        stub_probe(
            probe_payload(
                [video_stream(r_frame_rate="25/1", avg_frame_rate="0/0"), audio_stream()]
            )
        )
        probe = validate_and_probe(media_file)
        assert probe.is_vfr is False
        assert probe.fps == pytest.approx(25.0)


class TestDuration:
    def test_prefers_container_duration(self, media_file: Path, stub_probe) -> None:  # type: ignore[no-untyped-def]
        payload = probe_payload([video_stream(duration="99.0"), audio_stream()], duration="120.0")
        stub_probe(payload)
        assert validate_and_probe(media_file).duration == pytest.approx(120.0)

    def test_falls_back_to_stream_duration(self, media_file: Path, stub_probe) -> None:  # type: ignore[no-untyped-def]
        payload = {"streams": [video_stream(duration="88.5"), audio_stream()], "format": {}}
        stub_probe(payload)
        assert validate_and_probe(media_file).duration == pytest.approx(88.5)

    def test_rejects_unknown_duration(self, media_file: Path, stub_probe) -> None:  # type: ignore[no-untyped-def]
        payload = {
            "streams": [video_stream(duration="N/A"), audio_stream()],
            "format": {"duration": "N/A"},
        }
        stub_probe(payload)
        with pytest.raises(CorruptMediaError) as excinfo:
            validate_and_probe(media_file)
        assert excinfo.value.reason == "unknown_duration"


class TestResolution:
    def test_rejects_missing_dimensions(self, media_file: Path, stub_probe) -> None:  # type: ignore[no-untyped-def]
        stub_probe(probe_payload([video_stream(width=0, height=0), audio_stream()]))
        with pytest.raises(CorruptMediaError) as excinfo:
            validate_and_probe(media_file)
        assert excinfo.value.reason == "unknown_resolution"


class TestNbFrames:
    def test_parses_when_present(self, media_file: Path, stub_probe) -> None:  # type: ignore[no-untyped-def]
        stub_probe(probe_payload([video_stream(nb_frames="3000"), audio_stream()]))
        assert validate_and_probe(media_file).nb_frames == 3000

    @pytest.mark.parametrize("value", ["N/A", "", None])
    def test_is_none_when_unusable(
        self, media_file: Path, stub_probe, value: str | None
    ) -> None:  # type: ignore[no-untyped-def]
        stub_probe(probe_payload([video_stream(nb_frames=value), audio_stream()]))
        assert validate_and_probe(media_file).nb_frames is None


class TestFileLevelChecks:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(CorruptMediaError) as excinfo:
            validate_and_probe(tmp_path / "absent.mp4")
        assert excinfo.value.reason == "file_not_found"

    def test_empty_file_is_a_truncated_download(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.mp4"
        path.write_bytes(b"")
        with pytest.raises(CorruptMediaError) as excinfo:
            validate_and_probe(path)
        assert excinfo.value.reason == "empty_file"
        assert "truncated" in excinfo.value.to_dict()["remediation"].lower()


class TestProbeAudio:
    """The normalized WAV has no video track by design, so the video validator
    would (correctly) reject it. Audio-only files get their own probe."""

    def test_reads_audio_properties(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dialogue_extractor.ingestion.ffprobe_validator import probe_audio

        path = tmp_path / "audio.wav"
        path.write_bytes(b"\x00" * 1024)
        monkeypatch.setattr(
            ffprobe_validator,
            "probe_json",
            lambda p, binary=None: {
                "streams": [audio_stream(codec_name="pcm_s16le", sample_rate="16000", channels=1)],
                "format": {"duration": "42.5"},
            },
        )

        probe = probe_audio(path)
        assert probe.sample_rate == 16_000
        assert probe.channels == 1
        assert probe.codec == "pcm_s16le"
        assert probe.duration == pytest.approx(42.5)

    def test_rejects_a_file_with_no_audio(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dialogue_extractor.ingestion.ffprobe_validator import probe_audio

        path = tmp_path / "audio.wav"
        path.write_bytes(b"\x00" * 1024)
        monkeypatch.setattr(
            ffprobe_validator,
            "probe_json",
            lambda p, binary=None: {"streams": [video_stream()], "format": {}},
        )
        with pytest.raises(NoAudioStreamError):
            probe_audio(path)

    def test_rejects_missing_file(self, tmp_path: Path) -> None:
        from dialogue_extractor.ingestion.ffprobe_validator import probe_audio

        with pytest.raises(CorruptMediaError):
            probe_audio(tmp_path / "absent.wav")
