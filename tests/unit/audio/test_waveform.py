"""In-memory audio decoding.

Exists because pyannote 4.x decodes through torchcodec, which needs FFmpeg
*shared libraries*. A static FFmpeg build provides ffmpeg.exe and none of those,
so diarization fails with "Could not load libtorchcodec" on a machine with a
perfectly good FFmpeg. Decoding here removes that dependency entirely.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from dialogue_extractor.audio.waveform import as_pyannote_input, load_waveform, read_wav
from dialogue_extractor.domain.errors import DiarizationError


def write_wav(
    path: Path,
    samples: list[int],
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        fmt = {1: "b", 2: "h", 4: "i"}[sample_width]
        handle.writeframes(struct.pack(f"<{len(samples)}{fmt}", *samples))
    return path


class TestReadWav:
    def test_mono_shape_and_rate(self, tmp_path: Path) -> None:
        path = write_wav(tmp_path / "a.wav", [0, 1000, -1000, 32767])
        samples, rate = read_wav(path)
        assert samples.shape == (1, 4)
        assert rate == 16000

    def test_normalizes_to_unit_range(self, tmp_path: Path) -> None:
        path = write_wav(tmp_path / "a.wav", [32767, -32768, 0])
        samples, _ = read_wav(path)
        assert samples.max() <= 1.0
        assert samples.min() >= -1.0
        assert samples[0][2] == pytest.approx(0.0)

    def test_is_float32(self, tmp_path: Path) -> None:
        path = write_wav(tmp_path / "a.wav", [0, 100])
        samples, _ = read_wav(path)
        assert samples.dtype == np.float32

    def test_stereo_is_deinterleaved(self, tmp_path: Path) -> None:
        """Frames arrive interleaved; pyannote wants (channels, samples)."""
        path = write_wav(tmp_path / "a.wav", [100, 200, 300, 400], channels=2)
        samples, _ = read_wav(path)
        assert samples.shape == (2, 2)
        assert samples[0][0] < samples[1][0]  # left 100 vs right 200

    def test_preserves_sample_rate(self, tmp_path: Path) -> None:
        path = write_wav(tmp_path / "a.wav", [0, 1], sample_rate=44100)
        _, rate = read_wav(path)
        assert rate == 44100

    def test_rejects_non_wav(self, tmp_path: Path) -> None:
        path = tmp_path / "not.wav"
        path.write_bytes(b"this is not a wav file at all")
        with pytest.raises(wave.Error):
            read_wav(path)


class TestLoadWaveform:
    def test_returns_a_torch_tensor(self, tmp_path: Path) -> None:
        import torch

        path = write_wav(tmp_path / "a.wav", [0, 1000, -1000])
        tensor, rate = load_waveform(path)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3)
        assert rate == 16000

    def test_missing_file_raises_a_stage_error(self, tmp_path: Path) -> None:
        with pytest.raises(DiarizationError) as excinfo:
            load_waveform(tmp_path / "absent.wav")
        assert excinfo.value.reason == "audio_not_found"

    def test_empty_audio_raises(self, tmp_path: Path) -> None:
        path = write_wav(tmp_path / "empty.wav", [])
        with pytest.raises(DiarizationError) as excinfo:
            load_waveform(path)
        assert excinfo.value.reason == "empty_audio"


class TestAsPyannoteInput:
    def test_shape_matches_what_pyannote_expects(self, tmp_path: Path) -> None:
        path = write_wav(tmp_path / "a.wav", [0, 100, 200])
        payload = as_pyannote_input(path)
        assert set(payload) == {"waveform", "sample_rate"}
        assert payload["sample_rate"] == 16000
        assert payload["waveform"].shape[0] == 1
