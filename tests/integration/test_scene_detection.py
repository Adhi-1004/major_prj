"""Shot-cut detection against real video with a real cut.

Parsing is unit tested; this checks the thing that cannot be faked -- that
ffmpeg's scene score actually fires on a hard cut and stays quiet on continuous
footage. A detector that never fires would make the cut rule silently inert, and
runs would keep being credited across shot changes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from tests.conftest import FFMPEG, requires_ffmpeg

from dialogue_extractor.vision.scene_detector import cut_frames, detect_cuts

pytestmark = requires_ffmpeg


@pytest.fixture(scope="module")
def clip_with_a_cut(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two visually unrelated shots spliced together at t=2.0."""
    directory = tmp_path_factory.mktemp("cuts")
    out = directory / "cut.mp4"
    subprocess.run(
        [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=320x240:r=25:d=2",
            "-f", "lavfi", "-i", "color=c=white:s=320x240:r=25:d=2",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[out]",
            "-map", "[out]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-fps_mode", "cfr", "-r", "25", str(out),
        ],
        check=True, capture_output=True, timeout=120,
    )
    return out


@pytest.fixture(scope="module")
def continuous_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One unbroken shot -- the common case for a few seconds of dialogue."""
    directory = tmp_path_factory.mktemp("nocuts")
    out = directory / "continuous.mp4"
    subprocess.run(
        [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=320x240:r=25:d=4",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-fps_mode", "cfr", "-r", "25", str(out),
        ],
        check=True, capture_output=True, timeout=120,
    )
    return out


def test_a_hard_cut_is_detected(clip_with_a_cut: Path) -> None:
    cuts = detect_cuts(str(clip_with_a_cut))
    assert cuts, "black-to-white splice produced no cut"
    assert any(abs(c - 2.0) < 0.3 for c in cuts), f"cut not near t=2.0: {cuts}"


def test_continuous_footage_reports_no_cut(continuous_clip: Path) -> None:
    """A detector that fired on everything would break every run."""
    assert detect_cuts(str(continuous_clip)) == ()


def test_the_cut_maps_to_a_plausible_frame(clip_with_a_cut: Path) -> None:
    frames = cut_frames(detect_cuts(str(clip_with_a_cut)), 25.0)
    assert frames
    assert any(abs(f - 50) <= 5 for f in frames), f"expected near frame 50, got {frames}"


def test_the_threshold_is_a_real_control(clip_with_a_cut: Path) -> None:
    """Scene scores run 0-1, and a black-to-white splice scores the maximum, so
    only a threshold above 1.0 can suppress it. 0.99 does not -- which is worth
    recording, because it means the configured 0.4 has generous headroom before
    it would start missing genuine cuts."""
    assert detect_cuts(str(clip_with_a_cut), threshold=1.01) == ()
    assert detect_cuts(str(clip_with_a_cut), threshold=0.99) != ()
