"""Shared test fixtures.

Media fixtures are *generated* with ffmpeg rather than committed: it keeps the
repository small, makes each fixture's properties self-documenting (the command
that built it states its fps, duration, and content), and lets a test ask for an
exact frame rate instead of hunting for a sample that happens to have one.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from dialogue_extractor.domain.errors import FFmpegNotFoundError
from dialogue_extractor.infra.ffmpeg_runner import resolve_binary


def _try_resolve(name: str) -> str | None:
    """Use the project's own capability-based resolver, not shutil.which.

    shutil.which returns whatever is first on PATH, which on a machine with
    several ffmpeg builds installed may well be one that cannot do -fps_mode.
    The fixtures must build media with the same binary the pipeline will use.
    """
    try:
        return str(resolve_binary(name))
    except FFmpegNotFoundError:
        return None


FFMPEG = _try_resolve("ffmpeg")
FFPROBE = _try_resolve("ffprobe")

requires_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None,
    reason="no usable ffmpeg/ffprobe found (needs >= 5.1 with libx264)",
)


@pytest.fixture(scope="session")
def ffmpeg_bin() -> str:
    if FFMPEG is None:
        pytest.skip("no usable ffmpeg found")
    return FFMPEG


@pytest.fixture(scope="session")
def ffprobe_bin() -> str:
    if FFPROBE is None:
        pytest.skip("no usable ffprobe found")
    return FFPROBE


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("media")


def run_ffmpeg(ffmpeg: str, args: list[str]) -> None:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-2000:]}")


_LEVEL_STEP = 25
"""Spacing between adjacent encodable values, per channel.

Not 1. Even at crf 0 the RGB -> YUV colour-matrix conversion rounds, so indices
0 and 1 encoded as (0,0,0) and (0,0,1) both come back as (0,0,0) and the
channel silently carries no information. 25 leaves an order of magnitude more
headroom than the observed error, and decoding snaps to the nearest level.
"""


def encode_frame_index(index: int) -> tuple[int, int, int]:
    """Frame index as an RGB triple, base 10 per channel at 25-level spacing.

    Supports indices 0-999, which is ample for short test fixtures.
    """
    if not 0 <= index < 1000:
        raise ValueError(f"index {index} outside encodable range 0-999")
    return (
        (index % 10) * _LEVEL_STEP,
        ((index // 10) % 10) * _LEVEL_STEP,
        ((index // 100) % 10) * _LEVEL_STEP,
    )


def decode_frame_index(rgb: tuple[int, int, int]) -> int:
    """Inverse of encode_frame_index, snapping each channel to its nearest level."""
    r, g, b = (min(9, max(0, round(c / _LEVEL_STEP))) for c in rgb)
    return r + g * 10 + b * 100


@pytest.fixture(scope="session")
def counter_clip_25fps(ffmpeg_bin: str, media_dir: Path) -> Path:
    """4 s at 25 fps with each frame's index encoded in its own pixels.

    Lets a frame-extraction test assert it got frame N by *reading N back out of
    the image*, rather than trusting the decoder's own accounting.

    The index is carried as a solid colour rather than burned-in text: no font
    dependency (drawtext needs fontconfig, which is not configured on a stock
    Windows box), and the assertion is an exact integer comparison instead of
    OCR. Encoded losslessly by keeping each frame a flat colour, which survives
    yuv420p chroma subsampling and H.264 quantization intact.
    """
    out = media_dir / "counter_25fps.mp4"
    if out.exists():
        return out

    from PIL import Image

    frames_dir = media_dir / "counter_frames"
    frames_dir.mkdir(exist_ok=True)
    n_frames = 100  # 4 s at 25 fps
    for i in range(n_frames):
        Image.new("RGB", (320, 240), encode_frame_index(i)).save(frames_dir / f"f{i:05d}.png")

    run_ffmpeg(
        ffmpeg_bin,
        [
            "-framerate", "25", "-i", str(frames_dir / "f%05d.png"),
            "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
            "-c:v", "libx264", "-crf", "0", "-preset", "veryfast",
            "-pix_fmt", "yuv444p", "-fps_mode", "cfr", "-r", "25",
            "-c:a", "aac", "-shortest", "-movflags", "+faststart",
            str(out),
        ],
    )
    return out


@pytest.fixture(scope="session")
def clip_no_audio(ffmpeg_bin: str, media_dir: Path) -> Path:
    """Video with no audio track -- must fail validation before ASR runs."""
    out = media_dir / "no_audio.mp4"
    if not out.exists():
        run_ffmpeg(
            ffmpeg_bin,
            [
                "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
                str(out),
            ],
        )
    return out


@pytest.fixture(scope="session")
def clip_vfr(ffmpeg_bin: str, media_dir: Path) -> Path:
    """Variable frame rate source -- r_frame_rate and avg_frame_rate disagree.

    Normalization must force this to CFR, or invariant 2 does not hold.
    """
    out = media_dir / "vfr.mp4"
    if not out.exists():
        run_ffmpeg(
            ffmpeg_bin,
            [
                "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=3",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                "-vf", "select='not(mod(n,3))'",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-fps_mode", "vfr",
                "-c:a", "aac", "-shortest",
                str(out),
            ],
        )
    return out


@pytest.fixture(scope="session")
def clip_stereo_44k(ffmpeg_bin: str, media_dir: Path) -> Path:
    """Stereo 44.1 kHz audio -- must be normalized to mono 16 kHz."""
    out = media_dir / "stereo_44k.mp4"
    if not out.exists():
        run_ffmpeg(
            ffmpeg_bin,
            [
                "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=2",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-ac", "2", "-ar", "44100", "-shortest",
                str(out),
            ],
        )
    return out


def write_silent_wav(
    path: Path, *, seconds: float = 0.1, sample_rate: int = 16000, channels: int = 1
) -> Path:
    """Write a real PCM s16le WAV.

    Several stages now decode audio in-process rather than handing a path to a
    library, so fixtures have to be genuinely decodable rather than a plausible
    byte string.
    """
    import struct
    import wave as wave_module

    frame_count = int(seconds * sample_rate)
    with wave_module.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        total = frame_count * channels
        handle.writeframes(struct.pack(f"<{total}h", *([0] * total)))
    return path


def credential(name: str) -> str | None:
    """Read a credential the way the application does, not just from the shell.

    Tests previously used os.getenv, which only sees exported variables -- so a
    perfectly good token in .env left every credentialed test silently skipped.
    Going through Settings means the test and the code under test agree on where
    configuration comes from.
    """
    from dialogue_extractor.config.settings import Settings

    field = getattr(Settings(), name.lower(), None)
    if field is None:
        return None
    value = field.get_secret_value() if hasattr(field, "get_secret_value") else str(field)
    return value or None
