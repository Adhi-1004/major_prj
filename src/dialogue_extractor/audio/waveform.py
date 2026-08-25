"""Loading audio into memory, without torchcodec.

pyannote.audio 4.x decodes audio through torchcodec, which links against
FFmpeg's *shared libraries* (avcodec-*.dll and friends). A static FFmpeg build
provides `ffmpeg.exe` but none of those, so torchcodec fails with

    Could not load libtorchcodec. Likely causes: 1. FFmpeg is not properly
    installed in your environment...

even on a machine with a perfectly good, current FFmpeg. Requiring a second,
shared-library FFmpeg build would push that same trap onto anyone else running
this.

Avoided entirely instead: pyannote pipelines accept an in-memory waveform, and
ingestion already guarantees the audio is PCM signed-16 mono at 16 kHz, which
the standard library can decode with no extra dependency. Anything else is
transcoded with the FFmpeg binary we already validated.
"""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np

from dialogue_extractor.config.constants import (
    TARGET_AUDIO_CHANNELS,
    TARGET_AUDIO_SAMPLE_RATE,
)
from dialogue_extractor.domain.errors import DiarizationError
from dialogue_extractor.infra.logging import get_logger

log = get_logger(__name__)

_SAMPLE_WIDTH_DTYPES: dict[int, Any] = {
    1: np.uint8,
    2: np.int16,
    4: np.int32,
}


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Decode a PCM WAV into float32 samples in [-1, 1], shaped (channels, samples).

    Raises:
        wave.Error: if `path` is not a PCM WAV the standard library can read.
    """
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())

    dtype = _SAMPLE_WIDTH_DTYPES.get(sample_width)
    if dtype is None:
        raise wave.Error(f"unsupported sample width: {sample_width} bytes")

    raw = np.frombuffer(frames, dtype=dtype)

    if sample_width == 1:
        # 8-bit WAV is unsigned, centred on 128.
        samples = (raw.astype(np.float32) - 128.0) / 128.0
    else:
        max_value = float(np.iinfo(dtype).max) + 1.0
        samples = raw.astype(np.float32) / max_value

    # Interleaved frames -> (channels, samples), which is pyannote's layout.
    samples = samples.reshape(-1, channels).T if channels > 1 else samples.reshape(1, -1)

    return samples, sample_rate


def _transcode_to_wav(path: Path, dest: Path) -> None:
    """Convert arbitrary audio to the PCM WAV shape read_wav expects."""
    from dialogue_extractor.infra.ffmpeg_runner import run_ffmpeg

    run_ffmpeg(
        [
            "-i", str(path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(TARGET_AUDIO_SAMPLE_RATE),
            "-ac", str(TARGET_AUDIO_CHANNELS),
            str(dest),
        ]
    )


def load_waveform(path: Path) -> tuple[Any, int]:
    """Load `path` as a torch tensor shaped (channels, samples), plus its rate.

    Returns what pyannote expects as `{"waveform": ..., "sample_rate": ...}`.
    """
    try:
        import torch
    except ImportError as exc:
        raise DiarizationError(
            "torch_missing",
            "torch is not installed.",
            remediation="pip install -r requirements.txt",
        ) from exc

    if not path.exists():
        raise DiarizationError("audio_not_found", f"Audio file does not exist: {path}")

    try:
        samples, sample_rate = read_wav(path)
    except (wave.Error, EOFError, ValueError) as exc:
        # Not a plain PCM WAV. Transcode with the validated FFmpeg binary rather
        # than reaching for torchcodec.
        log.debug("waveform.transcoding", file=path.name, reason=type(exc).__name__)
        with tempfile.TemporaryDirectory() as tmp:
            converted = Path(tmp) / "converted.wav"
            _transcode_to_wav(path, converted)
            samples, sample_rate = read_wav(converted)

    if samples.size == 0:
        raise DiarizationError(
            "empty_audio",
            f"{path.name} decoded to zero samples.",
            remediation="Check that the audio file is not truncated.",
        )

    return torch.from_numpy(np.ascontiguousarray(samples)), sample_rate


def as_pyannote_input(path: Path) -> dict[str, Any]:
    """Build the in-memory input dict a pyannote pipeline accepts."""
    waveform, sample_rate = load_waveform(path)
    return {"waveform": waveform, "sample_rate": sample_rate}
