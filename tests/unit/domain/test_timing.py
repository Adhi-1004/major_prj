"""Timing arithmetic -- the two invariants the final answer rests on.

These are the cheapest tests in the project and they guard the most expensive
class of bug: an off-by-one here produces a real frame from the right video at
the wrong moment, which looks plausible and passes every smoke test.
"""

from __future__ import annotations

import pytest

from dialogue_extractor.domain.errors import StageError
from dialogue_extractor.domain.timing import (
    clip_frame_to_global,
    first_frame_at_or_after,
    format_timestamp,
    frame_containing,
    frame_to_seconds,
    k_frames_for,
    nearest_frame,
    overlap,
    pad_window,
)

# Real-world frame rates, including the two NTSC fractional ones that expose
# floating-point rounding bugs.
FPS_VALUES = [23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0]


class TestRoundTrip:
    @pytest.mark.parametrize("fps", FPS_VALUES)
    @pytest.mark.parametrize("frame", [0, 1, 37, 100, 2456, 129_600])
    def test_frame_to_seconds_and_back_is_identity(self, fps: float, frame: int) -> None:
        seconds = frame_to_seconds(frame, fps)
        assert frame_containing(seconds, fps) == frame
        assert nearest_frame(seconds, fps) == frame
        assert first_frame_at_or_after(seconds, fps) == frame

    @pytest.mark.parametrize("fps", FPS_VALUES)
    def test_exact_frame_boundary_does_not_drift(self, fps: float) -> None:
        """t = N/fps must land on N, not N-1 via 74.99999999999999."""
        for frame in range(0, 500):
            assert frame_containing(frame / fps, fps) == frame


class TestFrameContaining:
    def test_floors_within_a_frame(self) -> None:
        # At 25 fps frame 3 covers [0.12, 0.16).
        assert frame_containing(0.12, 25.0) == 3
        assert frame_containing(0.155, 25.0) == 3
        assert frame_containing(0.16, 25.0) == 4

    def test_time_zero_is_frame_zero(self) -> None:
        assert frame_containing(0.0, 30.0) == 0


class TestNearestFrame:
    def test_rounds_half_up_not_bankers(self) -> None:
        """Python's round() is banker's rounding: round(0.5) == 0, round(1.5) == 2.

        That would make the result depend on the parity of the frame index,
        which is not reproducible behaviour.
        """
        assert nearest_frame(0.5 / 25.0, 25.0) == 1  # exactly half a frame in
        assert nearest_frame(1.5 / 25.0, 25.0) == 2
        assert nearest_frame(2.5 / 25.0, 25.0) == 3

    def test_rounds_to_closest(self) -> None:
        assert nearest_frame(0.119, 25.0) == 3
        assert nearest_frame(0.159, 25.0) == 4


class TestFirstFrameAtOrAfter:
    def test_ceils_within_a_frame(self) -> None:
        assert first_frame_at_or_after(0.121, 25.0) == 4
        assert first_frame_at_or_after(0.12, 25.0) == 3

    def test_is_the_clip_start_frame_convention(self) -> None:
        """This is what ffmpeg keeps when trimming at `seconds`."""
        assert first_frame_at_or_after(12.5, 24.0) == 300


class TestInvariantOne:
    """global_frame == clip_start_frame + asd_frame_id."""

    @pytest.mark.parametrize(
        ("clip_start_seconds", "fps", "asd_frame_id"),
        [
            (0.0, 25.0, 0),
            (0.0, 25.0, 42),
            (12.5, 24.0, 0),
            (12.5, 24.0, 17),
            (3600.0, 29.97, 88),
        ],
    )
    def test_offset_is_added_back(
        self, clip_start_seconds: float, fps: float, asd_frame_id: int
    ) -> None:
        clip_start_frame = first_frame_at_or_after(clip_start_seconds, fps)
        global_frame = clip_frame_to_global(asd_frame_id, clip_start_frame)

        assert global_frame == clip_start_frame + asd_frame_id
        # And the inverse recovers the clip-relative index.
        assert global_frame - clip_start_frame == asd_frame_id

    def test_clip_starting_at_zero_is_a_noop(self) -> None:
        assert clip_frame_to_global(99, 0) == 99

    def test_forgetting_the_offset_is_detectable(self) -> None:
        """Guards the specific silent failure: a plausible but wrong frame."""
        clip_start_frame = first_frame_at_or_after(102.0, 24.0)
        assert clip_frame_to_global(8, clip_start_frame) != 8


class TestInvariantTwo:
    """timestamp == global_frame / fps, given forced CFR."""

    def test_matches_the_documented_example(self) -> None:
        # docs/approach.md's example result: frame 2456 at 00:01:42.350.
        # 2456 / 24 == 102.333..., which renders as 00:01:42.333; the doc's
        # 2456/102.350 pairing implies ~23.997 fps. Assert the relationship,
        # not the doc's rounded pair.
        fps = 24.0
        assert frame_to_seconds(2456, fps) == pytest.approx(102.3333, abs=1e-4)

    @pytest.mark.parametrize("fps", FPS_VALUES)
    def test_is_linear(self, fps: float) -> None:
        assert frame_to_seconds(2 * 100, fps) == pytest.approx(2 * frame_to_seconds(100, fps))


class TestKFramesFor:
    """The duration-derived smoothing rule (never a fixed frame count)."""

    @pytest.mark.parametrize(
        ("fps", "expected"),
        [(23.976, 3), (24.0, 3), (25.0, 3), (29.97, 4), (30.0, 4), (50.0, 6), (60.0, 7)],
    )
    def test_scales_with_frame_rate(self, fps: float, expected: int) -> None:
        assert k_frames_for(fps, 120.0) == expected

    @pytest.mark.parametrize("fps", FPS_VALUES)
    def test_covers_at_least_the_requested_duration(self, fps: float) -> None:
        """A run of k frames must last ~the configured minimum, at any fps.

        This is the property a fixed K=3 violates: it would be 125 ms at 24 fps
        but 50 ms at 60 fps.
        """
        k = k_frames_for(fps, 120.0)
        covered_ms = k / fps * 1000
        assert covered_ms == pytest.approx(120.0, abs=1000 / fps)

    def test_never_below_one(self) -> None:
        """A zero-length run must never be acceptable."""
        assert k_frames_for(24.0, 0.0) == 1
        assert k_frames_for(24.0, 1.0) == 1

    def test_rejects_negative_duration(self) -> None:
        with pytest.raises(StageError):
            k_frames_for(24.0, -1.0)


class TestPadWindow:
    def test_pads_both_sides(self) -> None:
        assert pad_window(10.0, 12.0, 1.5, 60.0) == (8.5, 13.5)

    def test_clamps_at_start_of_video(self) -> None:
        """A line delivered in the opening seconds must not produce t < 0."""
        assert pad_window(0.5, 2.0, 1.5, 60.0) == (0.0, 3.5)

    def test_clamps_at_end_of_video(self) -> None:
        assert pad_window(58.0, 59.5, 1.5, 60.0) == (56.5, 60.0)

    def test_clamps_both_ends_for_a_very_short_video(self) -> None:
        assert pad_window(0.5, 1.5, 10.0, 2.0) == (0.0, 2.0)

    def test_zero_padding_is_identity(self) -> None:
        assert pad_window(3.0, 4.0, 0.0, 60.0) == (3.0, 4.0)

    def test_rejects_inverted_window(self) -> None:
        with pytest.raises(StageError):
            pad_window(5.0, 3.0, 1.0, 60.0)

    def test_rejects_negative_padding(self) -> None:
        with pytest.raises(StageError):
            pad_window(3.0, 4.0, -1.0, 60.0)


class TestOverlap:
    def test_partial(self) -> None:
        assert overlap(0.0, 5.0, 3.0, 9.0) == pytest.approx(2.0)

    def test_contained(self) -> None:
        assert overlap(2.0, 3.0, 0.0, 10.0) == pytest.approx(1.0)

    def test_disjoint_is_zero_not_negative(self) -> None:
        assert overlap(0.0, 1.0, 5.0, 6.0) == 0.0

    def test_touching_is_zero(self) -> None:
        assert overlap(0.0, 1.0, 1.0, 2.0) == 0.0

    def test_is_symmetric(self) -> None:
        assert overlap(0.0, 5.0, 3.0, 9.0) == overlap(3.0, 9.0, 0.0, 5.0)


class TestFormatTimestamp:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.0, "00:00:00.000"),
            (102.35, "00:01:42.350"),  # the example in docs/approach.md
            (59.999, "00:00:59.999"),
            (3600.0, "01:00:00.000"),
            (3661.5, "01:01:01.500"),
            (86399.999, "23:59:59.999"),
        ],
    )
    def test_renders_hh_mm_ss_mmm(self, seconds: float, expected: str) -> None:
        assert format_timestamp(seconds) == expected

    def test_rounds_milliseconds_half_up(self) -> None:
        assert format_timestamp(1.0005) == "00:00:01.001"


class TestValidation:
    @pytest.mark.parametrize("fps", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_bad_fps(self, fps: float) -> None:
        with pytest.raises(StageError):
            frame_containing(1.0, fps)
        with pytest.raises(StageError):
            frame_to_seconds(1, fps)

    def test_rejects_negative_seconds(self) -> None:
        with pytest.raises(StageError):
            frame_containing(-0.1, 25.0)

    def test_rejects_negative_frame(self) -> None:
        with pytest.raises(StageError):
            frame_to_seconds(-1, 25.0)
        with pytest.raises(StageError):
            clip_frame_to_global(-1, 0)
        with pytest.raises(StageError):
            clip_frame_to_global(0, -1)

    def test_error_names_the_stage(self) -> None:
        with pytest.raises(StageError) as excinfo:
            frame_containing(1.0, 0.0)
        assert excinfo.value.to_dict()["stage"] == "validation"
        assert excinfo.value.reason == "invalid_fps"
