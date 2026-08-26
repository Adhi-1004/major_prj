"""The orchestrator's promise: a failure anywhere becomes a result document.

The design document says any stage failure surfaces as structured JSON naming
the stage and the reason, never an unhandled crash. This drives every stage into
failing, in both the anticipated way (a `StageError`) and the unanticipated one
(anything else), and asserts the caller always gets a document back.

The unanticipated case is the one that matters. A `StageError` is easy to
handle -- somebody already thought about it. A `KeyError` from a decoder nobody
predicted is where a promise like this is actually tested, and it is exactly
when the caller most needs to be told which stage was running.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dialogue_extractor.domain.enums import Mode, StageName, Status
from dialogue_extractor.domain.errors import (
    DownloadError,
    GatedModelError,
    NoAudioStreamError,
    StageError,
    TranscriptionError,
)
from dialogue_extractor.domain.models import (
    MediaManifest,
    RawMedia,
    SpeakerTurn,
    Transcript,
    Word,
)
from dialogue_extractor.pipeline.orchestrator import (
    Orchestrator,
    PipelineRequest,
    Stages,
)

# --- doubles -----------------------------------------------------------------


class FakeSource:
    def __init__(self, raise_: BaseException | None = None) -> None:
        self.raise_ = raise_

    def supports(self, uri: str) -> bool:
        return True

    def acquire(self, uri: str, dest_dir: Path) -> RawMedia:
        if self.raise_:
            raise self.raise_
        return RawMedia(path=Path("video.mp4"), source_uri=uri, title="fake")


class FakeNormalizer:
    def __init__(self, raise_: BaseException | None = None) -> None:
        self.raise_ = raise_

    def normalize(self, raw: RawMedia, work_dir: Path) -> MediaManifest:
        if self.raise_:
            raise self.raise_
        return MediaManifest(
            video_path=Path("normalized.mp4"),
            audio_path=Path("audio.wav"),
            fps=25.0,
            frame_count=250,
            duration=10.0,
            width=640,
            height=360,
            cfr_forced=True,
            source_was_vfr=False,
            video_sha256="v",
            audio_sha256="a",
        )


class FakeTranscriber:
    def __init__(self, raise_: BaseException | None = None) -> None:
        self.raise_ = raise_

    def transcribe(self, audio_path: Path) -> Transcript:
        if self.raise_:
            raise self.raise_
        return Transcript(
            words=(Word("hello", 1.0, 1.4, 0.9, None),),
            language="en",
            duration=10.0,
        )


class FakeDiarizer:
    def __init__(self, raise_: BaseException | None = None) -> None:
        self.raise_ = raise_

    def diarize(self, audio_path: Path) -> tuple[SpeakerTurn, ...]:
        if self.raise_:
            raise self.raise_
        return (SpeakerTurn(start=0.0, end=10.0, speaker_id="SPEAKER_00"),)


class FakeASD:
    name = "fake"

    def __init__(self, raise_: BaseException | None = None) -> None:
        self.raise_ = raise_

    def health_check(self) -> None:
        return None

    def detect(self, *args: Any, **kwargs: Any) -> Any:
        if self.raise_:
            raise self.raise_
        raise AssertionError("detect should not be reached in these tests")


class FakeExtractor:
    def extract(self, video_path: Path, frame_number: int, dest: Path) -> Path:
        return dest


def build(
    *,
    source: BaseException | None = None,
    normalizer: BaseException | None = None,
    transcriber: BaseException | None = None,
    diarizer: BaseException | None = None,
    asd: BaseException | None = None,
    tmp_path: Path,
) -> Orchestrator:
    stages = Stages(
        source_for=lambda uri: FakeSource(source),  # type: ignore[arg-type]
        normalizer=FakeNormalizer(normalizer),  # type: ignore[arg-type]
        transcriber=lambda: FakeTranscriber(transcriber),  # type: ignore[arg-type,return-value]
        diarizer=lambda: FakeDiarizer(diarizer),  # type: ignore[arg-type,return-value]
        asd=lambda: FakeASD(asd),  # type: ignore[arg-type,return-value]
        extractor=FakeExtractor(),  # type: ignore[arg-type]
    )
    return Orchestrator(stages, work_dir=tmp_path / "work", output_dir=tmp_path / "out")


REQUEST = PipelineRequest(url="https://example.com/v", dialogue="hello")


# --- anticipated failures ----------------------------------------------------


class TestStageErrorsBecomeDocuments:
    @pytest.mark.parametrize(
        ("kwarg", "error", "expected_stage"),
        [
            ("source", DownloadError("http_404", "gone"), StageName.INGESTION),
            (
                "normalizer",
                NoAudioStreamError("no_audio_stream", "silent film"),
                StageName.VALIDATION,
            ),
            (
                "transcriber",
                TranscriptionError("model_load_failed", "cuda gone"),
                StageName.TRANSCRIPTION,
            ),
            (
                "diarizer",
                GatedModelError("gated", "accept the terms"),
                StageName.DIARIZATION,
            ),
        ],
    )
    def test_each_stage_names_itself(
        self,
        kwarg: str,
        error: StageError,
        expected_stage: StageName,
        tmp_path: Path,
    ) -> None:
        orchestrator = build(**{kwarg: error}, tmp_path=tmp_path)  # type: ignore[arg-type]
        outcome = orchestrator.run(REQUEST)

        assert outcome.payload["status"] == Status.ERROR.value
        assert outcome.payload["stage"] == expected_stage.value
        assert outcome.payload["reason"] == error.reason
        assert not outcome.ok

    def test_remediation_survives_into_the_document(self, tmp_path: Path) -> None:
        """The one part of an error a user can act on must not be dropped."""
        error = GatedModelError(
            "gated", "terms not accepted", remediation="Accept them at huggingface.co"
        )
        outcome = build(diarizer=error, tmp_path=tmp_path).run(REQUEST)
        assert outcome.payload["remediation"] == "Accept them at huggingface.co"

    def test_nothing_is_raised_to_the_caller(self, tmp_path: Path) -> None:
        orchestrator = build(
            source=DownloadError("http_404", "gone"), tmp_path=tmp_path
        )
        orchestrator.run(REQUEST)  # must not raise


# --- unanticipated failures --------------------------------------------------


class TestUnexpectedFailuresAreContained:
    """The case the boundary actually exists for."""

    @pytest.mark.parametrize(
        "error",
        [
            KeyError("frame_id"),
            ValueError("bad literal"),
            RuntimeError("CUDA out of memory"),
            AttributeError("'NoneType' object has no attribute 'speaker'"),
            ZeroDivisionError("float division by zero"),
        ],
    )
    def test_any_exception_becomes_a_document(
        self, error: BaseException, tmp_path: Path
    ) -> None:
        outcome = build(transcriber=error, tmp_path=tmp_path).run(REQUEST)

        assert outcome.payload["status"] == Status.ERROR.value
        assert outcome.payload["reason"] == "unexpected_error"
        assert not outcome.ok

    def test_it_still_names_the_stage_that_was_running(self, tmp_path: Path) -> None:
        """Without this the document would be true but useless."""
        outcome = build(diarizer=KeyError("boom"), tmp_path=tmp_path).run(REQUEST)
        assert outcome.payload["stage"] == StageName.DIARIZATION.value
        assert outcome.stage_reached is StageName.DIARIZATION

    def test_the_exception_type_is_named_but_the_traceback_is_not(
        self, tmp_path: Path
    ) -> None:
        """A traceback can carry absolute paths and a dependency's arguments.

        The type is enough to act on; the rest belongs in the log.
        """
        outcome = build(
            transcriber=RuntimeError("secret-token-abc123 leaked here"),
            tmp_path=tmp_path,
        ).run(REQUEST)

        document = str(outcome.payload)
        assert "RuntimeError" in document
        assert "secret-token-abc123" not in document
        assert "Traceback" not in document

    def test_a_failure_before_any_stage_is_attributed_to_config(
        self, tmp_path: Path
    ) -> None:
        stages = Stages(
            source_for=lambda uri: (_ for _ in ()).throw(KeyError("registry")),
            normalizer=FakeNormalizer(),  # type: ignore[arg-type]
            transcriber=lambda: FakeTranscriber(),  # type: ignore[return-value]
            diarizer=lambda: FakeDiarizer(),  # type: ignore[return-value]
            asd=lambda: FakeASD(),  # type: ignore[return-value]
            extractor=FakeExtractor(),  # type: ignore[arg-type]
        )
        orchestrator = Orchestrator(
            stages, work_dir=tmp_path / "w", output_dir=tmp_path / "o"
        )
        outcome = orchestrator.run(REQUEST)
        assert outcome.payload["stage"] == StageName.INGESTION.value


# --- progress ----------------------------------------------------------------


class TestProgressReporting:
    def test_stages_are_announced_in_order(self, tmp_path: Path) -> None:
        seen: list[str] = []
        build(asd=RuntimeError("stop here"), tmp_path=tmp_path).run(
            REQUEST, on_progress=lambda stage, _msg: seen.append(stage.value)
        )
        assert seen[:5] == [
            StageName.INGESTION.value,
            StageName.NORMALIZATION.value,
            StageName.TRANSCRIPTION.value,
            StageName.DIARIZATION.value,
            StageName.ALIGNMENT.value,
        ]

    def test_progress_stops_where_the_failure_is(self, tmp_path: Path) -> None:
        seen: list[str] = []
        build(transcriber=KeyError("x"), tmp_path=tmp_path).run(
            REQUEST, on_progress=lambda stage, _msg: seen.append(stage.value)
        )
        assert seen[-1] == StageName.TRANSCRIPTION.value

    def test_a_broken_progress_callback_does_not_lose_the_result(
        self, tmp_path: Path
    ) -> None:
        """A caller's reporting bug must not be reported as a pipeline failure.

        It still fails the run -- the callback is the caller's code and cannot
        be silently swallowed -- but it fails as a contained document naming the
        stage, not as a traceback.
        """

        def broken(stage: StageName, message: str) -> None:
            raise RuntimeError("the caller's logger is broken")

        outcome = build(tmp_path=tmp_path).run(REQUEST, on_progress=broken)
        assert outcome.payload["status"] == Status.ERROR.value
        assert outcome.payload["reason"] == "unexpected_error"


# --- mode selection ----------------------------------------------------------


class TestModeIsChosenByTheRequest:
    def test_a_phrase_selects_target_mode(self) -> None:
        assert PipelineRequest(url="u", dialogue="a line").mode is Mode.TARGET

    def test_no_phrase_selects_discovery(self) -> None:
        assert PipelineRequest(url="u").mode is Mode.DISCOVERY

    def test_an_empty_phrase_is_not_a_target(self) -> None:
        """Otherwise it would search for nothing and report not_found."""
        assert PipelineRequest(url="u", dialogue="").mode is Mode.DISCOVERY


# --- the ok/status distinction ----------------------------------------------


class TestOkMeansAFrameWasProduced:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (Status.SUCCESS.value, True),
            (Status.AMBIGUOUS.value, True),
            (Status.NOT_ON_SCREEN.value, False),
            (Status.NOT_FOUND.value, False),
            (Status.ERROR.value, False),
        ],
    )
    def test_ambiguous_counts_but_a_correct_no_does_not(
        self, status: str, expected: bool, tmp_path: Path
    ) -> None:
        from dialogue_extractor.pipeline.orchestrator import PipelineOutcome

        assert PipelineOutcome(payload={"status": status}).ok is expected

    def test_a_payload_with_no_status_is_not_ok(self) -> None:
        from dialogue_extractor.pipeline.orchestrator import PipelineOutcome

        assert PipelineOutcome(payload={}).ok is False
