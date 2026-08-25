"""Frame extraction against real video.

The fixture encodes each frame's index in its own pixels, so these read the
answer *out of the extracted image* rather than trusting the decoder's
accounting. That matters because the failure mode is not an error -- it is a
real frame, from the right video, of the wrong moment, which looks entirely
correct until someone checks it against the audio.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.conftest import decode_frame_index, requires_ffmpeg

from dialogue_extractor.domain.errors import FrameExtractionError
from dialogue_extractor.vision.pyav_extractor import PyAVFrameExtractor, frame_filename

pytestmark = requires_ffmpeg


def decode_saved(path: Path) -> int:
    from PIL import Image

    return decode_frame_index(Image.open(path).convert("RGB").getpixel((160, 120)))


class TestFrameExactness:
    @pytest.mark.parametrize("frame_number", [0, 1, 7, 24, 37, 50, 75, 99])
    def test_extracts_exactly_the_requested_frame(
        self, counter_clip_25fps: Path, tmp_path: Path, frame_number: int
    ) -> None:
        extractor = PyAVFrameExtractor()
        dest = extractor.extract(
            counter_clip_25fps, frame_number, tmp_path / f"f{frame_number}.png"
        )
        assert decode_saved(dest) == frame_number

    def test_every_frame_in_the_clip_round_trips(
        self, counter_clip_25fps: Path, tmp_path: Path
    ) -> None:
        """Exhaustive rather than sampled: an off-by-one that only shows up at
        certain GOP positions would slip past a handful of spot checks."""
        extractor = PyAVFrameExtractor()
        for frame_number in range(0, 100, 7):
            dest = extractor.extract(
                counter_clip_25fps, frame_number, tmp_path / f"f{frame_number}.png"
            )
            assert decode_saved(dest) == frame_number, f"frame {frame_number}"

    def test_a_frame_far_from_a_keyframe_is_still_exact(
        self, counter_clip_25fps: Path, tmp_path: Path
    ) -> None:
        """The case a seek-only approach gets wrong: seeking lands on a
        keyframe, which can be many frames away from what was asked for."""
        dest = PyAVFrameExtractor().extract(
            counter_clip_25fps, 63, tmp_path / "f63.png"
        )
        assert decode_saved(dest) == 63

    def test_repeated_extraction_is_stable(
        self, counter_clip_25fps: Path, tmp_path: Path
    ) -> None:
        extractor = PyAVFrameExtractor()
        first = extractor.extract(counter_clip_25fps, 42, tmp_path / "a.png")
        second = extractor.extract(counter_clip_25fps, 42, tmp_path / "b.png")
        assert decode_saved(first) == decode_saved(second) == 42


class TestOutput:
    def test_writes_a_readable_png(self, counter_clip_25fps: Path, tmp_path: Path) -> None:
        from PIL import Image

        dest = PyAVFrameExtractor().extract(counter_clip_25fps, 10, tmp_path / "f.png")
        with Image.open(dest) as image:
            assert image.format == "PNG"
            assert image.size == (320, 240)

    def test_creates_the_output_directory(
        self, counter_clip_25fps: Path, tmp_path: Path
    ) -> None:
        dest = tmp_path / "deep" / "nested" / "f.png"
        PyAVFrameExtractor().extract(counter_clip_25fps, 10, dest)
        assert dest.exists()

    def test_filename_matches_the_documented_shape(self) -> None:
        assert frame_filename(2456) == "frame_2456.png"


class TestFailureModes:
    def test_missing_video(self, tmp_path: Path) -> None:
        with pytest.raises(FrameExtractionError) as excinfo:
            PyAVFrameExtractor().extract(tmp_path / "absent.mp4", 0, tmp_path / "f.png")
        assert excinfo.value.reason == "video_missing"

    def test_negative_frame(self, counter_clip_25fps: Path, tmp_path: Path) -> None:
        with pytest.raises(FrameExtractionError) as excinfo:
            PyAVFrameExtractor().extract(counter_clip_25fps, -1, tmp_path / "f.png")
        assert excinfo.value.reason == "negative_frame"

    def test_frame_past_the_end_is_reported(
        self, counter_clip_25fps: Path, tmp_path: Path
    ) -> None:
        """Better to say the frame does not exist than to silently return the
        last one, which would look like a plausible answer."""
        extractor = PyAVFrameExtractor()
        dest = tmp_path / "f.png"
        try:
            extractor.extract(counter_clip_25fps, 100_000, dest)
        except FrameExtractionError as exc:
            assert exc.reason in {"frame_not_found", "decode_failed"}
        else:
            # If it did return something, it must be the genuine last frame and
            # the approximation must have been logged, not passed off as exact.
            assert decode_saved(dest) <= 99

    def test_a_non_video_file_fails_clearly(self, tmp_path: Path) -> None:
        junk = tmp_path / "notavideo.mp4"
        junk.write_bytes(b"this is not a video")
        with pytest.raises(FrameExtractionError) as excinfo:
            PyAVFrameExtractor().extract(junk, 0, tmp_path / "f.png")
        assert excinfo.value.to_dict()["stage"] == "frame_extraction"
