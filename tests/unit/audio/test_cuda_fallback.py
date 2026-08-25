"""CUDA runtime handling.

Two distinct problems, both observed on this machine:

  1. ctranslate2 reports a CUDA device whenever the *driver* is installed, but
     the compute libraries ship separately. So the device probe passes, the
     model loads, and the run dies on the first inference. The fallback
     therefore has to sit at transcribe time, not just at load time.

  2. The pip-installed CUDA DLLs are not on the Windows DLL search path, so
     installing them changes nothing until the path is fixed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

from dialogue_extractor.audio.whisper_transcriber import (
    WhisperTranscriber,
    is_cuda_runtime_failure,
)
from dialogue_extractor.domain.errors import TranscriptionError
from dialogue_extractor.infra import cuda_runtime


class TestIsCudaRuntimeFailure:
    @pytest.mark.parametrize(
        "message",
        [
            "Library cublas64_12.dll is not found or cannot be loaded",
            "Library cudnn64_9.dll is not found or cannot be loaded",
            "CUDA failed with error out of memory",
            "libcudart.so.12: cannot open shared object file",
            "no kernel image is available for execution on the device",
            "GPU is not available",
        ],
    )
    def test_recognizes_cuda_failures(self, message: str) -> None:
        assert is_cuda_runtime_failure(RuntimeError(message)) is True

    @pytest.mark.parametrize(
        "message",
        [
            "audio file is corrupt",
            "tokenizer.json not found",
            "connection reset by peer",
            "",
        ],
    )
    def test_ignores_unrelated_failures(self, message: str) -> None:
        """A non-CUDA error must surface, not be silently retried on CPU."""
        assert is_cuda_runtime_failure(RuntimeError(message)) is False


class FakeInfo:
    language = "en"
    duration = 5.0


class FakeWord:
    def __init__(self, word: str, start: float, end: float) -> None:
        self.word = word
        self.start = start
        self.end = end
        self.probability = 0.9


class FakeSegment:
    def __init__(self, words: list[FakeWord]) -> None:
        self.words = words


class CudaFailingModel:
    """Loads fine, then dies on inference -- the real observed behaviour."""

    def __init__(self, message: str = "Library cublas64_12.dll is not found") -> None:
        self.message = message
        self.calls = 0

    def transcribe(self, audio: str, **options: Any) -> Any:
        self.calls += 1
        raise RuntimeError(self.message)


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "audio.wav"
    path.write_bytes(b"RIFF....WAVEfmt ")
    return path


class TestTranscribeTimeFallback:
    def test_falls_back_to_cpu_and_succeeds(
        self, audio_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rebuilt: dict[str, Any] = {}

        class CpuModel:
            def transcribe(self, audio: str, **options: Any) -> Any:
                return [FakeSegment([FakeWord("recovered", 0.0, 0.5)])], FakeInfo()

        def fake_whisper_model(size: str, device: str, compute_type: str) -> Any:
            rebuilt["device"] = device
            rebuilt["compute_type"] = compute_type
            return CpuModel()

        import faster_whisper

        monkeypatch.setattr(faster_whisper, "WhisperModel", fake_whisper_model)

        transcriber = WhisperTranscriber(model=CudaFailingModel(), device="cuda")
        transcript = transcriber.transcribe(audio_file)

        assert transcript.text == "recovered"
        assert rebuilt == {"device": "cpu", "compute_type": "int8"}
        assert transcriber.device == "cpu", "the instance must report where it ended up"

    def test_non_cuda_failure_is_not_retried(
        self, audio_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Retrying an unrelated error on CPU would just fail twice as slowly."""
        model = CudaFailingModel("audio file is corrupt")
        transcriber = WhisperTranscriber(model=model, device="cuda")

        with pytest.raises(TranscriptionError) as excinfo:
            transcriber.transcribe(audio_file)
        assert excinfo.value.reason == "transcription_failed"
        assert model.calls == 1

    def test_failure_on_both_devices_reports_both(
        self, audio_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_whisper_model(size: str, device: str, compute_type: str) -> Any:
            raise RuntimeError("cpu backend unavailable")

        import faster_whisper

        monkeypatch.setattr(faster_whisper, "WhisperModel", fake_whisper_model)

        transcriber = WhisperTranscriber(model=CudaFailingModel(), device="cuda")
        with pytest.raises(TranscriptionError) as excinfo:
            transcriber.transcribe(audio_file)

        detail = excinfo.value.detail
        assert "cublas" in detail.lower()
        assert "cpu backend unavailable" in detail
        assert excinfo.value.to_dict()["remediation"]

    def test_cpu_device_never_attempts_fallback(
        self, audio_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = CudaFailingModel()
        transcriber = WhisperTranscriber(model=model, device="cpu")

        with pytest.raises(TranscriptionError):
            transcriber.transcribe(audio_file)
        assert model.calls == 1


class TestCudaLibraryRegistration:
    @pytest.fixture(autouse=True)
    def reset_state(self) -> None:
        cuda_runtime.reset()

    def test_is_idempotent(self) -> None:
        first = cuda_runtime.register_cuda_libraries()
        second = cuda_runtime.register_cuda_libraries()
        assert second == [], "already-registered calls should be a no-op"
        assert isinstance(first, list)

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only DLL search path")
    def test_puts_directories_on_path(self) -> None:
        """PATH is the mechanism that actually works: CTranslate2 loads cuBLAS
        via its own LoadLibrary at first inference, which consults PATH.
        os.add_dll_directory alone was verified insufficient."""
        registered = cuda_runtime.register_cuda_libraries()
        if not registered:
            pytest.skip("no pip-installed CUDA packages present")
        for directory in registered:
            assert directory in os.environ["PATH"]

    @pytest.mark.skipif(sys.platform == "win32", reason="non-Windows behaviour")
    def test_is_a_noop_off_windows(self) -> None:
        assert cuda_runtime.register_cuda_libraries() == []

    def test_survives_absent_cuda_packages(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A machine with no GPU must not fail here."""
        monkeypatch.setattr(sys, "path", [str(tmp_path)])
        cuda_runtime.reset()
        assert cuda_runtime.register_cuda_libraries() == []
