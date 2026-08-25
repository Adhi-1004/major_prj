"""Media validation and measurement via ffprobe.

Runs before any expensive stage so that a file with no audio track, or one
ffprobe cannot parse at all, fails immediately with a specific reason instead
of surfacing as a confusing error inside the transcriber.

Also the place variable frame rate is detected. ffprobe reports two rates:

  * r_frame_rate   -- the container's nominal/base rate
  * avg_frame_rate -- frames actually present divided by duration

They agree on constant-frame-rate media and diverge on VFR. That divergence is
why normalization *forces* CFR rather than just recording the measured rate: on
a VFR source `frame_number == timestamp * fps` is simply false, and a frame
number computed from it is unverifiable.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from dialogue_extractor.config.constants import FALLBACK_FPS
from dialogue_extractor.domain.errors import (
    CorruptMediaError,
    NoAudioStreamError,
    NoVideoStreamError,
)
from dialogue_extractor.infra.ffmpeg_runner import probe_json
from dialogue_extractor.infra.logging import get_logger

log = get_logger(__name__)

# Beyond this relative difference, the two reported rates are treated as
# genuinely disagreeing rather than differing by rounding.
_VFR_RELATIVE_TOLERANCE = 0.01


@dataclass(frozen=True, slots=True)
class MediaProbe:
    """What ffprobe reports about a media file, parsed and typed."""

    path: Path
    duration: float
    width: int
    height: int
    video_codec: str
    audio_codec: str
    r_frame_rate: float
    avg_frame_rate: float
    is_vfr: bool
    nb_frames: int | None
    audio_sample_rate: int | None
    audio_channels: int | None

    @property
    def fps(self) -> float:
        """Best available frame rate.

        Prefers avg_frame_rate: on VFR input it reflects the frames actually
        present, which is what a CFR re-encode should target.
        """
        return self.avg_frame_rate or self.r_frame_rate or FALLBACK_FPS


def parse_frame_rate(raw: str | None) -> float:
    """Parse ffprobe's "num/den" rate string.

    NTSC rates arrive as exact fractions ("24000/1001"); parsing them as
    fractions rather than pre-rounded decimals keeps 23.976 from drifting.
    Returns 0.0 for absent, malformed, or zero-denominator values so callers
    can fall back rather than crash.
    """
    if not raw:
        return 0.0
    try:
        value = float(Fraction(raw))
    except (ValueError, ZeroDivisionError):
        return 0.0
    return value if value > 0 else 0.0


def _first_stream(streams: list[dict[str, Any]], codec_type: str) -> dict[str, Any] | None:
    for stream in streams:
        if stream.get("codec_type") == codec_type:
            return stream
    return None


def _duration_of(probe: dict[str, Any], video: dict[str, Any]) -> float:
    """Duration, preferring the container's value and falling back per-stream."""
    for source in (probe.get("format", {}), video):
        raw = source.get("duration")
        if raw:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
    return 0.0


def validate_and_probe(
    media_path: Path,
    *,
    require_audio: bool = True,
    ffprobe_binary: str | None = None,
) -> MediaProbe:
    """Validate `media_path` and return its measured properties.

    Raises NoVideoStreamError / NoAudioStreamError / CorruptMediaError.
    """
    if not media_path.exists():
        raise CorruptMediaError(
            "file_not_found", f"Media file does not exist: {media_path}"
        )
    if media_path.stat().st_size == 0:
        raise CorruptMediaError(
            "empty_file",
            f"Media file is empty: {media_path.name}",
            remediation="The download may have been truncated; try again.",
        )

    probe = probe_json(media_path, binary=ffprobe_binary)
    streams: list[dict[str, Any]] = probe.get("streams", [])

    if not streams:
        raise CorruptMediaError(
            "no_streams",
            f"ffprobe found no streams in {media_path.name}; the file is not usable media.",
            remediation="Check that the download completed and the file is a video.",
        )

    video = _first_stream(streams, "video")
    if video is None:
        raise NoVideoStreamError(
            "no_video_stream",
            f"{media_path.name} has no video track, so there are no frames to return.",
            remediation="Provide a video, not an audio-only file.",
        )

    audio = _first_stream(streams, "audio")
    if audio is None and require_audio:
        raise NoAudioStreamError(
            "no_audio_stream",
            f"{media_path.name} has no audio track. Every speech stage needs one.",
            remediation="Provide a video that contains spoken audio.",
        )

    duration = _duration_of(probe, video)
    if duration <= 0:
        raise CorruptMediaError(
            "unknown_duration",
            f"Could not determine the duration of {media_path.name}.",
            remediation="The file may be truncated or corrupt.",
        )

    r_rate = parse_frame_rate(video.get("r_frame_rate"))
    avg_rate = parse_frame_rate(video.get("avg_frame_rate"))
    is_vfr = _detect_vfr(r_rate, avg_rate)

    try:
        width = int(video.get("width") or 0)
        height = int(video.get("height") or 0)
    except (TypeError, ValueError):
        width = height = 0
    if width <= 0 or height <= 0:
        raise CorruptMediaError(
            "unknown_resolution",
            f"Could not determine the resolution of {media_path.name}.",
        )

    nb_frames_raw = video.get("nb_frames")
    try:
        nb_frames = int(nb_frames_raw) if nb_frames_raw else None
    except (TypeError, ValueError):
        nb_frames = None

    result = MediaProbe(
        path=media_path,
        duration=duration,
        width=width,
        height=height,
        video_codec=str(video.get("codec_name") or "unknown"),
        audio_codec=str(audio.get("codec_name") or "unknown") if audio else "none",
        r_frame_rate=r_rate,
        avg_frame_rate=avg_rate,
        is_vfr=is_vfr,
        nb_frames=nb_frames,
        audio_sample_rate=_int_or_none(audio.get("sample_rate")) if audio else None,
        audio_channels=_int_or_none(audio.get("channels")) if audio else None,
    )

    log.info(
        "probe.ok",
        file=media_path.name,
        duration_s=round(duration, 3),
        resolution=f"{width}x{height}",
        video_codec=result.video_codec,
        audio_codec=result.audio_codec,
        r_frame_rate=round(r_rate, 4),
        avg_frame_rate=round(avg_rate, 4),
        is_vfr=is_vfr,
    )
    return result


@dataclass(frozen=True, slots=True)
class AudioProbe:
    """Properties of an audio-only file.

    Separate from MediaProbe rather than a MediaProbe with empty video fields:
    the normalized WAV has no video track by design, and asking the video
    validator about it would (correctly) reject it.
    """

    path: Path
    duration: float
    codec: str
    sample_rate: int | None
    channels: int | None


def probe_audio(audio_path: Path, *, ffprobe_binary: str | None = None) -> AudioProbe:
    """Measure an audio-only file, e.g. the normalized 16 kHz WAV."""
    if not audio_path.exists():
        raise CorruptMediaError("file_not_found", f"Audio file does not exist: {audio_path}")
    if audio_path.stat().st_size == 0:
        raise CorruptMediaError("empty_file", f"Audio file is empty: {audio_path.name}")

    probe = probe_json(audio_path, binary=ffprobe_binary)
    streams: list[dict[str, Any]] = probe.get("streams", [])
    audio = _first_stream(streams, "audio")
    if audio is None:
        raise NoAudioStreamError(
            "no_audio_stream",
            f"{audio_path.name} contains no audio stream.",
        )

    duration = 0.0
    for source in (probe.get("format", {}), audio):
        raw = source.get("duration")
        if raw:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                duration = value
                break

    return AudioProbe(
        path=audio_path,
        duration=duration,
        codec=str(audio.get("codec_name") or "unknown"),
        sample_rate=_int_or_none(audio.get("sample_rate")),
        channels=_int_or_none(audio.get("channels")),
    )


def _detect_vfr(r_rate: float, avg_rate: float) -> bool:
    """True when the two reported rates disagree by more than rounding."""
    if r_rate <= 0 or avg_rate <= 0:
        return False
    return abs(r_rate - avg_rate) / max(r_rate, avg_rate) > _VFR_RELATIVE_TOLERANCE


def _int_or_none(raw: Any) -> int | None:
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None
