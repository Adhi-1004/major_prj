"""Finding shot cuts inside a candidate window.

Two edge cases in the design document depend on this:

  * "Dialogue starts before a camera cut" -- return the first frame *after* the
    cut where the speaker is actively speaking.
  * "A line is split across a cut mid-sentence" -- check sync per sub-segment,
    because a partial match would otherwise mislabel the whole line.

Both reduce to the same rule: a run of "this person is speaking" must not be
credited across a cut. The person on screen after a cut may not be the person
who was on screen before it, and a detector tracking faces frame to frame has no
way to know that a shot changed.

Cuts are found with ffmpeg's own scene-score filter rather than a separate
dependency -- ffmpeg is already required, and the score is exactly the frame
difference metric a purpose-built library would compute.
"""

from __future__ import annotations

import re

from dialogue_extractor.config.constants import SCENE_CUT_THRESHOLD
from dialogue_extractor.infra.ffmpeg_runner import run_ffmpeg
from dialogue_extractor.infra.logging import get_logger

log = get_logger(__name__)

# showinfo writes one line per surviving frame; pts_time is its timestamp.
_PTS_TIME = re.compile(r"pts_time:(\d+(?:\.\d+)?)")


def parse_cut_times(ffmpeg_stderr: str) -> tuple[float, ...]:
    """Extract cut timestamps from ffmpeg's showinfo output.

    Kept separate so it can be tested against recorded output without invoking
    ffmpeg.
    """
    times = sorted({float(match) for match in _PTS_TIME.findall(ffmpeg_stderr)})
    return tuple(times)


def detect_cuts(
    video_path: str,
    *,
    threshold: float = SCENE_CUT_THRESHOLD,
    ffmpeg_binary: str | None = None,
) -> tuple[float, ...]:
    """Cut timestamps within `video_path`, in seconds from its start.

    Returns clip-relative times, because it is run on the trimmed clip. An
    empty result means one continuous shot, which is the common case for a few
    seconds of dialogue.

    Failure is non-fatal: cut detection refines the answer rather than
    producing it, so a broken probe degrades to "no cuts found" rather than
    failing the run.
    """
    try:
        stderr = run_ffmpeg(
            [
                "-i", str(video_path),
                "-vf", f"select='gt(scene,{threshold})',showinfo",
                "-fps_mode", "passthrough",
                "-f", "null", "-",
            ],
            binary=ffmpeg_binary,
            # showinfo reports at INFO level; the default "error" would discard
            # every line this function exists to read.
            loglevel="info",
        )
    except Exception as exc:
        log.warning("scene.detection_failed", error=type(exc).__name__, detail=str(exc)[:200])
        return ()

    cuts = parse_cut_times(stderr)
    if cuts:
        log.debug("scene.cuts_found", count=len(cuts), times=[round(c, 3) for c in cuts])
    return cuts


def cut_frames(cuts: tuple[float, ...], fps: float) -> frozenset[int]:
    """Convert cut timestamps to the clip-relative frame indices they land on.

    The frame *containing* the cut is the first frame of the new shot, so a run
    may not span it.
    """
    from dialogue_extractor.domain.timing import frame_containing

    if fps <= 0:
        return frozenset()
    return frozenset(frame_containing(cut, fps) for cut in cuts if cut >= 0)
