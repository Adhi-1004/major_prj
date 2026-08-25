"""Speech-to-text with word-level timestamps, via faster-whisper.

Two things here are load-bearing rather than incidental:

**Determinism.** faster-whisper defaults to a *temperature fallback ladder*: on
low-confidence output it resamples at successively higher temperatures. That
makes the same audio produce different transcripts across runs, which
contradicts the Reproducibility requirement and would make a matching threshold
meaningless. Temperature is pinned to 0.0 and `condition_on_previous_text` is
off, so each window is decoded independently and identically every time.

**Word timestamps are a search hint, not the answer.** faster-whisper derives
them from cross-attention alignment, and they drift by roughly +/-0.2-0.4s. That
is fine here: they only decide *where to look*, and the candidate window is
padded to absorb the drift. The exact frame comes from the active-speaker model
downstream, at frame granularity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dialogue_extractor.config import constants
from dialogue_extractor.domain.errors import TranscriptionError
from dialogue_extractor.domain.models import Transcript, Word
from dialogue_extractor.infra.cuda_runtime import register_cuda_libraries
from dialogue_extractor.infra.logging import get_logger, stage
from dialogue_extractor.interfaces.ports import Transcriber

log = get_logger(__name__)


def resolve_device(requested: str = "auto") -> tuple[str, str]:
    """Pick (device, compute_type).

    int8_float16 on GPU keeps large-v3 inside 8 GB of VRAM with no meaningful
    accuracy cost; int8 on CPU is the only tolerable option for a feature-length
    file.
    """
    if requested == "cpu":
        return "cpu", "int8"
    if requested == "cuda":
        return "cuda", "int8_float16"

    register_cuda_libraries()
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "int8_float16"
    except Exception as exc:
        log.debug("cuda.probe_failed", error=type(exc).__name__, detail=str(exc)[:200])
    return "cpu", "int8"


_CUDA_ERROR_MARKERS = (
    "cublas",
    "cudnn",
    "cuda",
    "libcu",
    "no kernel image",
    "gpu",
)


def is_cuda_runtime_failure(exc: BaseException) -> bool:
    """True when `exc` looks like a missing or broken CUDA runtime.

    ctranslate2 reports a CUDA device as present whenever the *driver* is
    installed, but the compute libraries (cuBLAS, cuDNN) ship separately. So a
    machine can pass the device probe, load the model successfully, and only
    fail on the first actual inference -- which is why this is checked at
    transcribe time, not just at load time.
    """
    message = str(exc).lower()
    return any(marker in message for marker in _CUDA_ERROR_MARKERS)


class WhisperTranscriber(Transcriber):
    """Transcriber backed by faster-whisper."""

    def __init__(
        self,
        model_size: str = constants.DEFAULT_WHISPER_MODEL,
        *,
        device: str = "auto",
        language: str | None = "en",
        model: Any | None = None,
    ) -> None:
        """
        Args:
            model: A preloaded model, used by tests to avoid downloading weights.
            language: None auto-detects; naming it skips detection and avoids a
                short English clip being mistaken for another language.
        """
        self.model_size = model_size
        self.language = language
        self._model = model
        self._device, self._compute_type = resolve_device(device)

    @property
    def device(self) -> str:
        return self._device

    @property
    def compute_type(self) -> str:
        return self._compute_type

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscriptionError(
                "faster_whisper_missing",
                "faster-whisper is not installed.",
                remediation="pip install -r requirements.txt",
            ) from exc

        log.info(
            "whisper.loading",
            model=self.model_size,
            device=self._device,
            compute_type=self._compute_type,
        )
        try:
            self._model = WhisperModel(
                self.model_size, device=self._device, compute_type=self._compute_type
            )
        except Exception as exc:
            # A CUDA-capable machine missing the cuDNN runtime is the common
            # case; falling back beats failing outright, but say so loudly.
            if self._device == "cuda":
                log.warning(
                    "whisper.cuda_load_failed_falling_back_to_cpu",
                    error=type(exc).__name__,
                    detail=str(exc)[:300],
                )
                self._device, self._compute_type = "cpu", "int8"
                try:
                    self._model = WhisperModel(
                        self.model_size, device="cpu", compute_type="int8"
                    )
                except Exception as cpu_exc:
                    raise TranscriptionError(
                        "model_load_failed",
                        f"Could not load Whisper model {self.model_size!r}: {cpu_exc}",
                        remediation="Check disk space and network access for the model download.",
                    ) from cpu_exc
            else:
                raise TranscriptionError(
                    "model_load_failed",
                    f"Could not load Whisper model {self.model_size!r}: {exc}",
                    remediation="Check disk space and network access for the model download.",
                ) from exc
        return self._model

    def _transcribe_options(self) -> dict[str, Any]:
        """Decoding options. Kept separate so a test can assert them directly."""
        return {
            "word_timestamps": True,
            "language": self.language,
            "beam_size": constants.WHISPER_BEAM_SIZE,
            "temperature": constants.WHISPER_TEMPERATURE,
            "condition_on_previous_text": constants.WHISPER_CONDITION_ON_PREVIOUS_TEXT,
            "vad_filter": True,
        }

    def transcribe(self, audio_path: Path) -> Transcript:
        if not audio_path.exists():
            raise TranscriptionError(
                "audio_not_found", f"Audio file does not exist: {audio_path}"
            )

        model = self._load_model()

        with stage("transcription", audio=audio_path.name, device=self._device) as report:
            try:
                segments, info = model.transcribe(str(audio_path), **self._transcribe_options())
                words = extract_words(segments)
            except TranscriptionError:
                raise
            except Exception as exc:
                if self._device == "cuda" and is_cuda_runtime_failure(exc):
                    segments, info, words = self._retry_on_cpu(audio_path, exc)
                else:
                    raise TranscriptionError(
                        "transcription_failed",
                        f"Transcription failed: {type(exc).__name__}: {str(exc)[:300]}",
                    ) from exc

            language = getattr(info, "language", None) or self.language or "unknown"
            duration = float(getattr(info, "duration", 0.0) or 0.0)
            report["words"] = len(words)
            report["language"] = language
            # After a fallback this differs from the device logged at entry.
            report["device_used"] = self._device

        if not words:
            raise TranscriptionError(
                "no_speech_detected",
                "No speech was found in the audio.",
                remediation="Check that the video actually contains spoken dialogue.",
            )

        return Transcript(words=words, language=language, duration=duration)

    def _retry_on_cpu(self, audio_path: Path, original: BaseException) -> tuple[Any, Any, Any]:
        """Rebuild the model on CPU and transcribe again.

        Slower, but a correct answer slowly beats a crash. The warning names the
        missing library so the user can install it and get the GPU back.
        """
        log.warning(
            "whisper.cuda_runtime_failed_falling_back_to_cpu",
            error=type(original).__name__,
            detail=str(original)[:300],
            remediation="pip install nvidia-cublas-cu12 nvidia-cudnn-cu12",
        )
        try:
            from faster_whisper import WhisperModel

            self._device, self._compute_type = "cpu", "int8"
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            segments, info = self._model.transcribe(
                str(audio_path), **self._transcribe_options()
            )
            return segments, info, extract_words(segments)
        except Exception as exc:
            raise TranscriptionError(
                "transcription_failed",
                f"Transcription failed on GPU ({original}) and on CPU ({exc}).",
                remediation="Install the CUDA runtime libraries, or run with --device cpu.",
            ) from exc


def extract_words(segments: Any) -> tuple[Word, ...]:
    """Flatten faster-whisper's segment/word structure into domain Words.

    Separate from the class so it can be tested against a fake generator without
    loading a model. Words with missing or nonsensical timings are dropped: they
    cannot be matched against or turned into a candidate window, and keeping
    them would corrupt the ordering that matching relies on.
    """
    words: list[Word] = []

    for segment in segments:
        for raw in getattr(segment, "words", None) or ():
            text = (getattr(raw, "word", None) or "").strip()
            start = getattr(raw, "start", None)
            end = getattr(raw, "end", None)

            if not text or start is None or end is None:
                continue
            try:
                start_f, end_f = float(start), float(end)
            except (TypeError, ValueError):
                continue
            if end_f < start_f or start_f < 0:
                continue

            probability = getattr(raw, "probability", None)
            try:
                prob = float(probability) if probability is not None else 1.0
            except (TypeError, ValueError):
                prob = 1.0

            words.append(
                Word(
                    text=text,
                    start=start_f,
                    end=end_f,
                    probability=min(1.0, max(0.0, prob)),
                )
            )

    words.sort(key=lambda w: (w.start, w.end))
    return tuple(words)
