"""Trimming a candidate window into a clip the ASD service will accept.

Three properties are load-bearing:

**`+faststart`.** The service's documentation states plainly that non-streamable
MP4 files "will fail to process correctly". It is also what makes the service
select its streaming path. Verified on the produced file rather than trusted to
the flag, because the flag silently does nothing on some inputs.

**The frame offset.** The service numbers frames from the start of the clip it
was given, not from the start of the video. Every returned frame id has to have
`clip_start_frame` added back, or the answer is a real frame from the right
video at the wrong moment -- which looks entirely plausible.

**Constant frame rate, preserved.** Ingestion already forced CFR; re-encoding
here has to keep it, or `timestamp == frame / fps` stops holding for the clip.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dialogue_extractor.config.constants import MAX_CLIP_HEIGHT
from dialogue_extractor.domain.errors import ClipBuildError
from dialogue_extractor.domain.models import Candidate, MediaManifest
from dialogue_extractor.domain.timing import first_frame_at_or_after
from dialogue_extractor.infra.ffmpeg_runner import run_ffmpeg
from dialogue_extractor.infra.logging import get_logger
from dialogue_extractor.ingestion.normalizer import is_faststart

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Clip:
    """A trimmed clip, plus everything needed to map its results back."""

    path: Path
    audio_path: Path | None
    start: float
    end: float
    clip_start_frame: int
    fps: float
    size_bytes: int

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1_048_576


def build_clip(
    candidate: Candidate,
    manifest: MediaManifest,
    dest_dir: Path,
    *,
    name: str = "clip",
    max_height: int = MAX_CLIP_HEIGHT,
    extract_audio: bool = True,
    ffmpeg_binary: str | None = None,
) -> Clip:
    """Trim `candidate`'s window out of the normalized video.

    Trimming is done by *re-encoding* rather than stream-copying. Stream-copying
    is faster but can only cut on a keyframe, so the clip would silently start
    somewhere other than asked -- and every frame index derived from it would be
    off by that unknown amount.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    video_out = dest_dir / f"{name}.mp4"
    audio_out = dest_dir / f"{name}.wav" if extract_audio else None

    if candidate.end <= candidate.start:
        raise ClipBuildError(
            "empty_window",
            f"Candidate window is empty: {candidate.start} -> {candidate.end}",
        )

    clip_start_frame = first_frame_at_or_after(candidate.start, manifest.fps)
    duration = candidate.end - candidate.start

    args: list[str] = [
        # -ss before -i seeks quickly; the re-encode below makes the cut exact.
        "-ss", f"{candidate.start:.6f}",
        "-t", f"{duration:.6f}",
        "-i", str(manifest.video_path),
        "-map", "0:v:0",
    ]

    if manifest.height > max_height:
        args += ["-vf", f"scale=-2:{max_height}"]

    args += [
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        # Preserve CFR: the clip's own frame indexing must stay linear.
        "-fps_mode", "cfr",
        "-r", f"{manifest.fps:.6f}",
        # Mandatory, not an optimization -- see the module docstring.
        "-movflags", "+faststart",
        "-map", "0:a:0?",
        "-c:a", "aac",
        "-b:a", "128k",
        str(video_out),
    ]

    if audio_out is not None:
        args += [
            "-ss", f"{candidate.start:.6f}",
            "-t", f"{duration:.6f}",
            "-i", str(manifest.audio_path),
            "-map", "1:a:0",
            "-acodec", "pcm_s16le",
            str(audio_out),
        ]

    run_ffmpeg(args, binary=ffmpeg_binary)

    if not video_out.exists() or video_out.stat().st_size == 0:
        raise ClipBuildError(
            "empty_clip",
            f"Clip extraction produced no output for window "
            f"{candidate.start:.2f}-{candidate.end:.2f}.",
        )

    if not is_faststart(video_out):
        raise ClipBuildError(
            "not_streamable",
            "The clip is not streamable (moov atom is not at the front). The "
            "detection service fails quietly on such files rather than "
            "reporting an error.",
            remediation="Check that this ffmpeg build supports -movflags +faststart.",
        )

    clip = Clip(
        path=video_out,
        audio_path=audio_out if audio_out and audio_out.exists() else None,
        start=candidate.start,
        end=candidate.end,
        clip_start_frame=clip_start_frame,
        fps=manifest.fps,
        size_bytes=video_out.stat().st_size,
    )

    log.debug(
        "clip.built",
        window=f"{candidate.start:.2f}-{candidate.end:.2f}",
        duration_s=round(clip.duration, 2),
        size_mb=round(clip.size_mb, 2),
        clip_start_frame=clip_start_frame,
    )
    return clip


def halve(candidate: Candidate) -> Candidate:
    """Halve a candidate's window, keeping its start.

    Used when the service rejects a request as too large. The onset is what the
    answer depends on, so the tail is what gets dropped.
    """
    from dataclasses import replace

    midpoint = candidate.start + (candidate.end - candidate.start) / 2.0
    if midpoint <= candidate.start:
        raise ClipBuildError(
            "cannot_shrink",
            "Candidate window is already too short to halve.",
        )
    return replace(candidate, end=midpoint)
