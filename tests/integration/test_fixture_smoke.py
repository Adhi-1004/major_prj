"""Fixtures must be trustworthy before anything is built on top of them."""

from __future__ import annotations

import subprocess
from pathlib import Path

from tests.conftest import FFMPEG, decode_frame_index, requires_ffmpeg


@requires_ffmpeg
def test_counter_clip_builds(counter_clip_25fps: Path) -> None:
    assert counter_clip_25fps.exists() and counter_clip_25fps.stat().st_size > 0


@requires_ffmpeg
def test_no_audio_clip_builds(clip_no_audio: Path) -> None:
    assert clip_no_audio.exists()


@requires_ffmpeg
def test_vfr_clip_builds(clip_vfr: Path) -> None:
    assert clip_vfr.exists()


@requires_ffmpeg
def test_stereo_clip_builds(clip_stereo_44k: Path) -> None:
    assert clip_stereo_44k.exists()


@requires_ffmpeg
def test_frame_index_survives_h264_encoding(counter_clip_25fps: Path, tmp_path: Path) -> None:
    """Every decoded frame must report its own index.

    The Phase 6 frame-exactness tests assert "I asked for frame N and got frame
    N" by decoding this colour back to an integer. If lossy encoding perturbed
    the colour, those tests would silently be checking nothing -- so verify the
    channel is lossless here, once, before relying on it.
    """
    from PIL import Image

    out_dir = tmp_path / "frames"
    out_dir.mkdir()
    subprocess.run(
        [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(counter_clip_25fps),
            "-fps_mode", "passthrough",
            str(out_dir / "f%05d.png"),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )

    pngs = sorted(out_dir.glob("f*.png"))
    assert len(pngs) == 100, f"expected 100 frames, decoded {len(pngs)}"

    # ffmpeg's image2 muxer numbers from 1; frame index is that minus 1.
    for offset, png in enumerate(pngs):
        rgb = Image.open(png).convert("RGB").getpixel((160, 120))
        assert decode_frame_index(rgb) == offset, (
            f"{png.name}: encoded index {decode_frame_index(rgb)}, expected {offset}"
        )
