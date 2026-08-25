"""Structured errors.

The brief requires a clear, structured error rather than a silent crash. These
tests pin that every failure type names its stage and serializes to a document
the CLI and the API can both return.
"""

from __future__ import annotations

import json

import pytest

from dialogue_extractor.domain import errors
from dialogue_extractor.domain.enums import StageName, Status

# Every concrete error, paired with the stage it must attribute itself to.
ERROR_STAGE_PAIRS = [
    (errors.InvalidURLError, StageName.INGESTION),
    (errors.DownloadError, StageName.INGESTION),
    (errors.GeoRestrictedError, StageName.INGESTION),
    (errors.DRMProtectedError, StageName.INGESTION),
    (errors.NoAudioStreamError, StageName.VALIDATION),
    (errors.NoVideoStreamError, StageName.VALIDATION),
    (errors.CorruptMediaError, StageName.VALIDATION),
    (errors.FFmpegError, StageName.NORMALIZATION),
    (errors.FFmpegNotFoundError, StageName.NORMALIZATION),
    (errors.TranscriptionError, StageName.TRANSCRIPTION),
    (errors.DiarizationError, StageName.DIARIZATION),
    (errors.GatedModelError, StageName.DIARIZATION),
    (errors.DialogueNotFoundError, StageName.MATCHING),
    (errors.ClipBuildError, StageName.CLIP_BUILD),
    (errors.ASDAuthError, StageName.ACTIVE_SPEAKER_DETECTION),
    (errors.ASDQuotaError, StageName.ACTIVE_SPEAKER_DETECTION),
    (errors.ASDTimeoutError, StageName.ACTIVE_SPEAKER_DETECTION),
    (errors.ASDUnavailableError, StageName.ACTIVE_SPEAKER_DETECTION),
    (errors.MalformedDiarizationError, StageName.ACTIVE_SPEAKER_DETECTION),
    (errors.DiarizationDroppedError, StageName.ACTIVE_SPEAKER_DETECTION),
    (errors.FrameExtractionError, StageName.FRAME_EXTRACTION),
    (errors.MissingCredentialError, StageName.CONFIG),
]


@pytest.mark.parametrize(("error_cls", "expected_stage"), ERROR_STAGE_PAIRS)
class TestEveryErrorIsAttributable:
    def test_carries_its_stage(
        self, error_cls: type[errors.StageError], expected_stage: StageName
    ) -> None:
        assert error_cls("some_reason", "Something went wrong.").stage is expected_stage

    def test_serializes_to_the_documented_shape(
        self, error_cls: type[errors.StageError], expected_stage: StageName
    ) -> None:
        payload = error_cls("some_reason", "Something went wrong.").to_dict()
        assert payload == {
            "status": Status.ERROR.value,
            "stage": expected_stage.value,
            "reason": "some_reason",
            "detail": "Something went wrong.",
        }

    def test_is_json_serializable(
        self, error_cls: type[errors.StageError], expected_stage: StageName
    ) -> None:
        payload = error_cls("r", "d").to_dict()
        assert json.loads(json.dumps(payload))["stage"] == expected_stage.value

    def test_is_catchable_as_stage_error(
        self, error_cls: type[errors.StageError], expected_stage: StageName
    ) -> None:
        """The orchestrator's single except clause must catch all of them."""
        with pytest.raises(errors.StageError):
            raise error_cls("r", "d")


class TestRemediation:
    def test_included_when_provided(self) -> None:
        payload = errors.MissingCredentialError(
            "missing_hf_token", "HF_TOKEN is not set.", remediation="Create a token."
        ).to_dict()
        assert payload["remediation"] == "Create a token."

    def test_omitted_when_absent(self) -> None:
        assert "remediation" not in errors.DownloadError("r", "d").to_dict()


class TestMessage:
    def test_str_names_stage_and_reason(self) -> None:
        message = str(errors.NoAudioStreamError("no_audio_stream", "No audio track."))
        assert "validation" in message
        assert "no_audio_stream" in message
        assert "No audio track." in message


class TestStageOverride:
    def test_explicit_stage_wins(self) -> None:
        """timing.py raises a bare StageError attributed to validation."""
        err = errors.StageError("invalid_fps", "bad fps", stage=StageName.VALIDATION)
        assert err.stage is StageName.VALIDATION
        assert err.to_dict()["stage"] == "validation"


class TestHierarchy:
    def test_diarization_dropped_is_an_asd_error(self) -> None:
        assert issubclass(errors.DiarizationDroppedError, errors.ASDError)

    def test_gated_model_is_a_diarization_error(self) -> None:
        assert issubclass(errors.GatedModelError, errors.DiarizationError)

    def test_specific_ingestion_errors_are_catchable_together(self) -> None:
        for cls in (errors.DownloadError, errors.GeoRestrictedError, errors.DRMProtectedError):
            with pytest.raises(errors.IngestionError):
                raise cls("r", "d")
