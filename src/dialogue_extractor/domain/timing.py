"""Time <-> frame arithmetic.

This module is the single home of the two invariants the whole result depends
on (see the implementation plan, section 4):

  1. global_frame = clip_start_frame + asd_frame_id
     The ASD service indexes frames within the *clip we sent*, not the source.

  2. timestamp = global_frame / fps
     Valid only because ingestion forces constant frame rate. On a VFR source
     this relationship does not hold and a returned frame number is
     unverifiable, which is why normalization forces CFR rather than merely
     measuring the frame rate.

Three distinct seconds->frame conversions are provided because they are *not*
interchangeable, and picking the wrong one is an off-by-one that survives all
the way to the extracted image:

  * frame_containing        -- which frame is on screen at time t      (floor)
  * nearest_frame           -- which frame is closest to time t        (round)
  * first_frame_at_or_after -- first frame ffmpeg keeps trimming at t  (ceil)

Pure: no I/O, no third-party imports.
"""

from __future__ import annotations

import math

from dialogue_extractor.domain.enums import StageName
from dialogue_extractor.domain.errors import StageError

# Absorbs binary floating-point error so that, e.g., 3.0 seconds at 25 fps
# lands on frame 75 rather than 74 via 74.99999999999999.
_EPS = 1e-9


def _validate_fps(fps: float) -> None:
    if not math.isfinite(fps) or fps <= 0:
        raise StageError(
            "invalid_fps",
            f"Frame rate must be a positive finite number, got {fps!r}.",
            stage=StageName.VALIDATION,
        )


def _validate_seconds(seconds: float, *, name: str = "seconds") -> None:
    if not math.isfinite(seconds):
        raise StageError(
            "invalid_timestamp",
            f"{name} must be finite, got {seconds!r}.",
            stage=StageName.VALIDATION,
        )
    if seconds < 0:
        raise StageError(
            "negative_timestamp",
            f"{name} must be non-negative, got {seconds!r}.",
            stage=StageName.VALIDATION,
        )


def frame_containing(seconds: float, fps: float) -> int:
    """Index of the frame displayed at `seconds`.

    Frame N covers the half-open interval [N/fps, (N+1)/fps), so this floors.
    Use when asking "what is on screen right now".
    """
    _validate_fps(fps)
    _validate_seconds(seconds)
    return math.floor(seconds * fps + _EPS)


def nearest_frame(seconds: float, fps: float) -> int:
    """Index of the frame whose start time is closest to `seconds`.

    Rounds half **up** deliberately. Python's built-in round() is
    banker's rounding -- round(0.5) == 0 -- which would make the result depend
    on the parity of the frame index and break reproducibility.
    """
    _validate_fps(fps)
    _validate_seconds(seconds)
    return math.floor(seconds * fps + 0.5 + _EPS)


def first_frame_at_or_after(seconds: float, fps: float) -> int:
    """Index of the first frame at or after `seconds`.

    This is what ffmpeg keeps when trimming a clip starting at `seconds`, so it
    is the correct value for `clip_start_frame` in invariant 1.
    """
    _validate_fps(fps)
    _validate_seconds(seconds)
    return math.ceil(seconds * fps - _EPS)


def frame_to_seconds(frame: int, fps: float) -> float:
    """Presentation time of `frame` -- invariant 2."""
    _validate_fps(fps)
    if frame < 0:
        raise StageError(
            "negative_frame",
            f"Frame index must be non-negative, got {frame!r}.",
            stage=StageName.VALIDATION,
        )
    return frame / fps


def clip_frame_to_global(clip_frame: int, clip_start_frame: int) -> int:
    """Map a clip-relative frame index back to the source video -- invariant 1.

    The ASD response numbers frames from the start of the clip we uploaded.
    Forgetting this offset is the "Trimmed-clip frame offset" edge case in
    docs/approach.md, and it fails silently: you get a real frame, just the
    wrong one.
    """
    if clip_frame < 0:
        raise StageError(
            "negative_frame",
            f"Clip frame index must be non-negative, got {clip_frame!r}.",
            stage=StageName.VALIDATION,
        )
    if clip_start_frame < 0:
        raise StageError(
            "negative_frame",
            f"Clip start frame must be non-negative, got {clip_start_frame!r}.",
            stage=StageName.VALIDATION,
        )
    return clip_start_frame + clip_frame


def k_frames_for(fps: float, min_speaking_ms: float) -> int:
    """Frames needed to cover `min_speaking_ms` of speech at `fps`.

    The temporal-smoothing rule must be expressed as a *duration*, not a fixed
    frame count: a hardcoded K=3 is 125 ms at 24 fps but only 50 ms at 60 fps,
    so the detector would silently get stricter or looser with the source's
    frame rate -- on exactly the varying footage the design expects. Always at
    least 1, so the rule can never accept a zero-length run.
    """
    _validate_fps(fps)
    if not math.isfinite(min_speaking_ms) or min_speaking_ms < 0:
        raise StageError(
            "invalid_duration",
            f"min_speaking_ms must be non-negative and finite, got {min_speaking_ms!r}.",
            stage=StageName.VALIDATION,
        )
    return max(1, math.floor(min_speaking_ms / 1000.0 * fps + 0.5 + _EPS))


def pad_window(
    start: float,
    end: float,
    pad: float,
    duration: float,
) -> tuple[float, float]:
    """Widen [start, end] by `pad` on both sides, clamped to [0, duration].

    The padding absorbs ASR word-timestamp jitter so the candidate clip
    reliably contains the utterance. Clamping matters at the boundaries: a line
    delivered in the opening or closing seconds would otherwise produce a
    negative start or an end past EOF.
    """
    _validate_seconds(start, name="start")
    _validate_seconds(end, name="end")
    _validate_seconds(duration, name="duration")
    if end < start:
        raise StageError(
            "inverted_window",
            f"Window end ({end}) precedes start ({start}).",
            stage=StageName.VALIDATION,
        )
    if pad < 0 or not math.isfinite(pad):
        raise StageError(
            "invalid_padding",
            f"Padding must be non-negative and finite, got {pad!r}.",
            stage=StageName.VALIDATION,
        )
    return (max(0.0, start - pad), min(duration, end + pad))


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Length of the intersection of two intervals; 0.0 when disjoint.

    Used by the word/speaker aligner to pick the greatest-overlap speaker turn.
    """
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def format_timestamp(seconds: float) -> str:
    """Render seconds as HH:MM:SS.sss, the output format the brief specifies."""
    _validate_seconds(seconds)
    total_ms = math.floor(seconds * 1000 + 0.5 + _EPS)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    return f"{total_s // 3600:02d}:{(total_s // 60) % 60:02d}:{total_s % 60:02d}.{ms:03d}"
