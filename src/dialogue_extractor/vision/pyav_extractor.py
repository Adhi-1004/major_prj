"""Pulling one exact frame out of the video.

The frame number is half the answer; the image is the other half, and it has to
be *that* frame, not one nearby.

Decoding is done by seeking to the preceding keyframe and stepping forward,
matching on presentation timestamp. The obvious alternative -- asking ffmpeg for
`-ss <t> -frames:v 1` -- is only as accurate as the seek, and a seek lands on a
keyframe. On a GOP of 250 frames that is up to ten seconds away, and the
returned image would look entirely plausible while showing the wrong moment.

Matching on PTS rather than counting decoded frames matters for the same reason:
a count assumes decoding started exactly where the seek claimed, which is the
thing that cannot be assumed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dialogue_extractor.domain.errors import FrameExtractionError
from dialogue_extractor.infra.logging import get_logger
from dialogue_extractor.interfaces.ports import FrameExtractor

log = get_logger(__name__)


class PyAVFrameExtractor(FrameExtractor):
    """Frame-exact extraction via PyAV."""

    def __init__(self, *, tolerance_frames: float = 0.5) -> None:
        """
        Args:
            tolerance_frames: How close a decoded frame's timestamp must be to
                the target, in frames. Half a frame means "this frame and no
                other" while absorbing floating-point error in the timebase
                conversion.
        """
        self._tolerance_frames = tolerance_frames

    def extract(self, video_path: Path, frame_number: int, dest: Path) -> Path:
        """Save frame `frame_number` of `video_path` to `dest` as a PNG."""
        try:
            import av
        except ImportError as exc:
            raise FrameExtractionError(
                "pyav_missing",
                "PyAV is not installed.",
                remediation="pip install -r requirements.txt",
            ) from exc

        if not video_path.exists():
            raise FrameExtractionError(
                "video_missing", f"Video does not exist: {video_path}"
            )
        if frame_number < 0:
            raise FrameExtractionError(
                "negative_frame", f"Frame number must be non-negative, got {frame_number}."
            )

        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            with av.open(str(video_path)) as container:
                stream = container.streams.video[0]
                # Decoding is deterministic and single-threaded here: frame-exact
                # extraction is a one-shot operation, and the threading modes
                # that speed up bulk decode reorder output.
                stream.thread_type = "NONE"

                image = self._decode_frame(container, stream, frame_number)
                image.to_image().save(dest)

        except FrameExtractionError:
            raise
        except Exception as exc:
            raise FrameExtractionError(
                "decode_failed",
                f"Could not extract frame {frame_number}: {type(exc).__name__}: "
                f"{str(exc)[:300]}",
            ) from exc

        log.debug("frame.extracted", frame=frame_number, path=str(dest))
        return dest

    def _decode_frame(self, container: Any, stream: Any, frame_number: int) -> Any:
        """Seek to the keyframe before the target, then step forward to it."""
        time_base = stream.time_base
        average_rate = stream.average_rate
        if time_base is None or average_rate is None or float(average_rate) <= 0:
            raise FrameExtractionError(
                "unknown_frame_rate",
                "The video does not report a usable frame rate, so a frame number "
                "cannot be converted to a timestamp.",
            )

        fps = float(average_rate)
        target_seconds = frame_number / fps
        tolerance_seconds = self._tolerance_frames / fps

        # Seek backwards to the nearest keyframe at or before the target.
        target_pts = int(target_seconds / float(time_base))
        try:
            container.seek(target_pts, stream=stream, backward=True, any_frame=False)
        except Exception:
            # A stream that cannot be seeked is still decodable from the start,
            # just more slowly.
            container.seek(0)

        best: Any = None
        best_delta = float("inf")

        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_seconds = float(frame.pts * time_base)
            delta = abs(frame_seconds - target_seconds)

            if delta < best_delta:
                best, best_delta = frame, delta

            if frame_seconds >= target_seconds - tolerance_seconds and delta <= tolerance_seconds:
                return frame
            # Decoding past the target with nothing close enough means the
            # exact frame does not exist; keep the nearest and stop.
            if frame_seconds > target_seconds + tolerance_seconds:
                break

        if best is None:
            raise FrameExtractionError(
                "frame_not_found",
                f"Frame {frame_number} could not be decoded; the video may be shorter "
                "than that.",
            )

        if best_delta > tolerance_seconds:
            log.warning(
                "frame.approximate",
                requested=frame_number,
                off_by_frames=round(best_delta * fps, 3),
            )
        return best


def frame_filename(frame_number: int, prefix: str = "frame") -> str:
    """Output filename for a frame, matching the shape in the design document."""
    return f"{prefix}_{frame_number}.png"
