"""Speaker diarization via pyannote.audio.

Answers "who spoke when", independently of what was said. Its output is joined
to the transcript by `aligner.py`.

Two practical notes:

**The pipeline is gated.** `pyannote/speaker-diarization-community-1` requires
accepting terms on Hugging Face *in addition to* holding a valid token, and the
failure for "token is fine but terms not accepted" is a bare 401 that says
nothing useful. That case is detected and turned into an instruction, because it
is the single most common way this stage fails on a fresh machine.

**The v4 API differs from v3.** The pipeline id changed, and the auth keyword is
now `token=` rather than the pre-4.x `use_auth_token=`. Passing the old keyword
raises a TypeError that looks unrelated to authentication.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dialogue_extractor.audio.waveform import as_pyannote_input
from dialogue_extractor.config import constants
from dialogue_extractor.domain.errors import DiarizationError, GatedModelError
from dialogue_extractor.domain.models import SpeakerTurn
from dialogue_extractor.infra.logging import get_logger, stage
from dialogue_extractor.interfaces.ports import Diarizer

log = get_logger(__name__)

_GATED_MARKERS = (
    "gated",
    "awaiting a review",
    "accept the conditions",
    "user conditions",
    "401",
    "unauthorized",
    "authentication",
    "access to model",
    "is not authorized",
)


def looks_like_gated_access(exc: BaseException) -> bool:
    """True when the failure is really "you have not accepted the terms"."""
    return any(marker in str(exc).lower() for marker in _GATED_MARKERS)


def _gated_error(exc: BaseException) -> GatedModelError:
    return GatedModelError(
        "gated_model_access",
        f"Could not load {constants.PYANNOTE_PIPELINE!r}: {str(exc)[:300]}",
        remediation=(
            "This pipeline is gated. Two separate steps are required: "
            f"(1) visit https://huggingface.co/{constants.PYANNOTE_PIPELINE} while "
            "signed in and accept the user conditions; "
            "(2) create a read token at https://huggingface.co/settings/tokens and "
            "set it as HF_TOKEN in .env. A valid token alone is not sufficient."
        ),
    )


class PyannoteDiarizer(Diarizer):
    """Diarizer backed by pyannote.audio."""

    def __init__(
        self,
        token: str | None = None,
        *,
        pipeline_name: str = constants.PYANNOTE_PIPELINE,
        device: str = "auto",
        pipeline: Any | None = None,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> None:
        """
        Args:
            pipeline: A preloaded pipeline, so tests never touch the network.
            num_speakers: Fixed count when known. Left None, pyannote estimates
                it -- which is the right default for an arbitrary video.
        """
        self.pipeline_name = pipeline_name
        self._token = token
        self._pipeline = pipeline
        self._device = device
        self._num_speakers = num_speakers
        self._min_speakers = min_speakers
        self._max_speakers = max_speakers

    def _resolve_device(self) -> str:
        if self._device in ("cpu", "cuda"):
            return self._device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception as exc:
            log.debug("torch.probe_failed", error=type(exc).__name__)
            return "cpu"

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        # Configuration is checked before dependencies: it is cheaper, it is the
        # thing most often missing on an otherwise-working machine, and it keeps
        # this failure path testable without pyannote installed.
        if not self._token:
            raise GatedModelError(
                "missing_hf_token",
                "No Hugging Face token was provided for the gated diarization pipeline.",
                remediation=(
                    "Set HF_TOKEN in .env, and accept the user conditions at "
                    f"https://huggingface.co/{constants.PYANNOTE_PIPELINE}"
                ),
            )

        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise DiarizationError(
                "pyannote_missing",
                "pyannote.audio is not installed.",
                remediation="pip install -r requirements.txt",
            ) from exc

        log.info("pyannote.loading", pipeline=self.pipeline_name)
        try:
            # pyannote.audio 4.x uses `token=`; the pre-4.x `use_auth_token=`
            # raises a TypeError that reads as unrelated to authentication.
            pipeline = Pipeline.from_pretrained(self.pipeline_name, token=self._token)
        except Exception as exc:
            if looks_like_gated_access(exc):
                raise _gated_error(exc) from exc
            raise DiarizationError(
                "pipeline_load_failed",
                f"Could not load {self.pipeline_name!r}: {type(exc).__name__}: {str(exc)[:300]}",
            ) from exc

        if pipeline is None:
            # from_pretrained returns None rather than raising when access is
            # refused -- a null pipeline here means the terms were not accepted.
            raise _gated_error(RuntimeError("from_pretrained returned None (access refused)"))

        device = self._resolve_device()
        if device == "cuda":
            try:
                import torch

                pipeline.to(torch.device("cuda"))
            except Exception as exc:
                log.warning(
                    "pyannote.cuda_unavailable_using_cpu",
                    error=type(exc).__name__,
                    detail=str(exc)[:200],
                )
                device = "cpu"

        log.info("pyannote.loaded", pipeline=self.pipeline_name, device=device)
        self._pipeline = pipeline
        return pipeline

    def _pipeline_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self._num_speakers is not None:
            kwargs["num_speakers"] = self._num_speakers
        if self._min_speakers is not None:
            kwargs["min_speakers"] = self._min_speakers
        if self._max_speakers is not None:
            kwargs["max_speakers"] = self._max_speakers
        return kwargs

    def diarize(self, audio_path: Path) -> tuple[SpeakerTurn, ...]:
        if not audio_path.exists():
            raise DiarizationError(
                "audio_not_found", f"Audio file does not exist: {audio_path}"
            )

        pipeline = self._load_pipeline()

        # Decoded in-process rather than handing pyannote a path: its 4.x
        # decoder (torchcodec) needs FFmpeg *shared libraries*, which a static
        # FFmpeg build does not provide, and failing on that would be absurd on
        # a machine that already has a working FFmpeg. See audio/waveform.py.
        audio_input = as_pyannote_input(audio_path)

        with stage("diarization", audio=audio_path.name) as report:
            try:
                annotation = pipeline(audio_input, **self._pipeline_kwargs())
            except DiarizationError:
                raise
            except Exception as exc:
                if looks_like_gated_access(exc):
                    raise _gated_error(exc) from exc
                raise DiarizationError(
                    "diarization_failed",
                    f"Diarization failed: {type(exc).__name__}: {str(exc)[:300]}",
                ) from exc

            # Inside the boundary on purpose: conversion can fail on an
            # unexpected result shape, and that must surface as a structured
            # error like everything else, not as a traceback.
            try:
                turns = extract_turns(annotation)
            except DiarizationError:
                raise
            except Exception as exc:
                raise DiarizationError(
                    "unexpected_result_shape",
                    f"Could not read diarization output: {type(exc).__name__}: {str(exc)[:300]}",
                    remediation=(
                        "This usually means the installed pyannote.audio version returns a "
                        "different result type than expected. Check the pinned version in "
                        "requirements.txt."
                    ),
                ) from exc

            report["turns"] = len(turns)
            report["speakers"] = len({t.speaker_id for t in turns})

        if not turns:
            raise DiarizationError(
                "no_speakers_detected",
                "Diarization found no speaker turns in the audio.",
                remediation="Check that the audio contains speech.",
            )
        return turns


def unwrap_annotation(result: Any) -> Any:
    """Get the Annotation out of whatever the pipeline returned.

    pyannote.audio 4.x returns a `DiarizeOutput` carrying several annotations
    (`speaker_diarization`, `exclusive_speaker_diarization`, plus embeddings)
    rather than the bare `Annotation` that 3.x returned. The overlapping
    `speaker_diarization` is the one wanted here: alignment resolves overlaps by
    greatest overlap, so crosstalk is information rather than a problem.
    """
    if hasattr(result, "itertracks"):
        return result
    for attribute in ("speaker_diarization", "exclusive_speaker_diarization"):
        candidate = getattr(result, attribute, None)
        if candidate is not None and hasattr(candidate, "itertracks"):
            return candidate
    raise DiarizationError(
        "unexpected_result_shape",
        f"Diarization returned {type(result).__name__}, which carries no readable annotation.",
    )


def extract_turns(
    result: Any,
    *,
    min_duration: float = constants.MIN_SPEAKER_TURN_SECONDS,
) -> tuple[SpeakerTurn, ...]:
    """Convert a pyannote diarization result into domain SpeakerTurns.

    Kept a free function so it can be tested against a fake annotation without
    loading a model. Inverted turns are dropped, and so are turns shorter than
    `min_duration` -- see MIN_SPEAKER_TURN_SECONDS for why those are artifacts
    rather than data, and what leaving them in does to speaker attribution.
    """
    annotation = unwrap_annotation(result)
    turns: list[SpeakerTurn] = []
    discarded = 0

    for item in annotation.itertracks(yield_label=True):
        segment, _track, label = item
        try:
            start = float(segment.start)
            end = float(segment.end)
        except (TypeError, ValueError, AttributeError):
            continue
        if end <= start or start < 0:
            continue
        if (end - start) < min_duration:
            discarded += 1
            continue
        turns.append(SpeakerTurn(speaker_id=str(label), start=start, end=end))

    if discarded:
        log.debug(
            "diarization.short_turns_discarded",
            discarded=discarded,
            kept=len(turns),
            min_duration=min_duration,
        )

    turns.sort(key=lambda t: (t.start, t.end))
    return tuple(turns)


def turns_to_dict(turns: tuple[SpeakerTurn, ...]) -> list[dict[str, Any]]:
    return [{"speaker_id": t.speaker_id, "start": t.start, "end": t.end} for t in turns]


def turns_from_dict(payload: list[dict[str, Any]]) -> tuple[SpeakerTurn, ...]:
    return tuple(
        SpeakerTurn(
            speaker_id=str(item["speaker_id"]),
            start=float(item["start"]),
            end=float(item["end"]),
        )
        for item in payload
    )
