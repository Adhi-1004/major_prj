"""Shot-cut detection.

Two edge cases in the design document reduce to the same rule: a run of "this
person is speaking" must not be credited across a cut, because the person in
frame afterwards may not be the person before it.

Parsing is tested against recorded ffmpeg output rather than by invoking it, so
the awkward cases -- no cuts, malformed lines, duplicate timestamps -- are cheap
to express.
"""

from __future__ import annotations

import pytest

from dialogue_extractor.vision.scene_detector import cut_frames, detect_cuts, parse_cut_times

# Real showinfo output: one line per frame surviving the scene filter.
SHOWINFO = """\
[Parsed_showinfo_1 @ 000001] config in time_base: 1/12800, frame_rate: 25/1
[Parsed_showinfo_1 @ 000001] n:   0 pts:  30720 pts_time:2.4 duration:512 \
duration_time:0.04 fmt:yuv420p sar:1/1 s:320x240
[Parsed_showinfo_1 @ 000001] n:   1 pts:  61440 pts_time:4.8 duration:512 \
duration_time:0.04 fmt:yuv420p sar:1/1 s:320x240
"""

NO_CUTS = """\
[Parsed_showinfo_1 @ 000001] config in time_base: 1/12800, frame_rate: 25/1
frame=    0 fps=0.0 q=-0.0 Lsize=N/A time=00:00:04.00 bitrate=N/A speed=  50x
"""


class TestParseCutTimes:
    def test_extracts_timestamps(self) -> None:
        assert parse_cut_times(SHOWINFO) == (2.4, 4.8)

    def test_no_cuts_is_empty(self) -> None:
        """One continuous shot is the common case for a few seconds of dialogue."""
        assert parse_cut_times(NO_CUTS) == ()

    def test_empty_input(self) -> None:
        assert parse_cut_times("") == ()

    def test_results_are_sorted(self) -> None:
        jumbled = "pts_time:9.0 x\npts_time:1.5 y\npts_time:4.2 z"
        assert parse_cut_times(jumbled) == (1.5, 4.2, 9.0)

    def test_duplicates_are_collapsed(self) -> None:
        assert parse_cut_times("pts_time:2.4\npts_time:2.4") == (2.4,)

    def test_integer_timestamps(self) -> None:
        assert parse_cut_times("pts_time:3") == (3.0,)

    def test_zero(self) -> None:
        assert parse_cut_times("pts_time:0") == (0.0,)

    def test_unrelated_numbers_are_not_picked_up(self) -> None:
        """`duration_time` and `n:` appear on the same lines."""
        line = "n: 12 pts: 30720 pts_time:2.4 duration:512 duration_time:0.04"
        assert parse_cut_times(line) == (2.4,)


class TestCutFrames:
    def test_converts_seconds_to_frame_indices(self) -> None:
        assert cut_frames((2.4,), 25.0) == frozenset({60})

    def test_several_cuts(self) -> None:
        assert cut_frames((0.4, 2.4, 4.0), 25.0) == frozenset({10, 60, 100})

    def test_the_frame_containing_the_cut_is_the_new_shot(self) -> None:
        """Floors rather than rounds: the cut lands *within* that frame, and
        that frame is the first of the new shot."""
        assert cut_frames((2.45,), 25.0) == frozenset({61})

    def test_fractional_frame_rate(self) -> None:
        assert cut_frames((1.0,), 23.976023976) == frozenset({23})

    def test_no_cuts(self) -> None:
        assert cut_frames((), 25.0) == frozenset()

    def test_invalid_frame_rate_is_survivable(self) -> None:
        """Cut detection refines the answer; it must never fail the run."""
        assert cut_frames((2.4,), 0.0) == frozenset()

    def test_negative_timestamps_are_ignored(self) -> None:
        assert cut_frames((-1.0, 2.4), 25.0) == frozenset({60})


class TestFailureIsNonFatal:
    def test_a_broken_probe_degrades_to_no_cuts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cut detection sharpens the answer rather than producing it, so a
        failure must not take the run down with it."""

        def explode(*args: object, **kwargs: object) -> str:
            raise RuntimeError("ffmpeg fell over")

        monkeypatch.setattr(
            "dialogue_extractor.vision.scene_detector.run_ffmpeg", explode
        )
        assert detect_cuts("anything.mp4") == ()

    def test_a_missing_file_degrades_to_no_cuts(self, tmp_path: object) -> None:
        assert detect_cuts(str(tmp_path) + "/absent.mp4") == ()
