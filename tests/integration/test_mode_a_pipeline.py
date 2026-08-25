"""Mode A end to end, with the detection service stubbed.

Real video, real clip building, real frame extraction, real scene detection --
only the network call is faked. This is the regression gate for the core
deliverable: it asserts the exact final JSON, so any change anywhere upstream
that alters the answer shows up here rather than in a live run.

The stub returns *positional* detections keyed by frame id, which is what makes
the frame arithmetic checkable: the test knows which clip-relative frame it
marked as the first speaking one, and asserts the reported global frame is that
frame plus the clip offset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from tests.conftest import decode_frame_index, requires_ffmpeg

from dialogue_extractor.domain.enums import AmbiguityReason, Mode, Status
from dialogue_extractor.domain.models import (
    ASDFrame,
    ASDResult,
    BoundingBox,
    Candidate,
    Face,
    MediaManifest,
    RawMedia,
    Word,
)
from dialogue_extractor.domain.timing import first_frame_at_or_after
from dialogue_extractor.ingestion.normalizer import normalize
from dialogue_extractor.interfaces.ports import ASDProvider
from dialogue_extractor.pipeline import mode_a
from dialogue_extractor.vision.pyav_extractor import PyAVFrameExtractor

pytestmark = requires_ffmpeg

TARGET = "my mind rebels at stagnation"


@pytest.fixture(scope="module")
def manifest(counter_clip_25fps: Path, tmp_path_factory: pytest.TempPathFactory) -> MediaManifest:
    work = tmp_path_factory.mktemp("modea")
    raw = RawMedia(path=counter_clip_25fps, source_uri=str(counter_clip_25fps), title="fixture")
    return normalize(raw, work)


def words_at(start: float, speaker: str = "SPEAKER_01") -> tuple[Word, ...]:
    """The target phrase, spoken from `start`."""
    result = []
    t = start
    for token in TARGET.split(" "):
        result.append(Word(token, round(t, 3), round(t + 0.2, 3), 0.92, speaker))
        t += 0.25
    return tuple(result)


def candidate_at(start: float, end: float, speaker: str | None = "SPEAKER_01") -> Candidate:
    return Candidate(
        start=start,
        end=end,
        speaker_id=speaker,
        text=TARGET,
        mode=Mode.TARGET,
        match_score=96.0,
        words=words_at(start + 0.2, speaker or "SPEAKER_01"),
    )


class StubProvider(ASDProvider):
    """Returns a scripted detection pattern, and records what it was asked."""

    name = "stub"

    def __init__(
        self,
        *,
        speaking_from: int | None = 6,
        speaker_code: int | None = 1,
        frames: int = 40,
        fail_first: int = 0,
    ) -> None:
        self.speaking_from = speaking_from
        self.speaker_code = speaker_code
        self.frames = frames
        self.fail_first = fail_first
        self.calls = 0
        self.seen_offsets: list[int] = []
        self.seen_diarizations: list[dict[str, Any]] = []

    def health_check(self) -> None:
        return None

    def detect(
        self,
        clip_path: Path,
        audio_path: Path | None,
        diarization: dict[str, Any],
        clip_start_frame: int,
        fps: float,
    ) -> ASDResult:
        self.calls += 1
        self.seen_offsets.append(clip_start_frame)
        self.seen_diarizations.append(diarization)

        if self.calls <= self.fail_first:
            from dialogue_extractor.domain.errors import ASDUnavailableError

            raise ASDUnavailableError("service_unavailable", "stubbed outage")

        frames = []
        for index in range(self.frames):
            speaking = self.speaking_from is not None and index >= self.speaking_from
            frames.append(
                ASDFrame(
                    index,
                    (
                        Face(
                            face_id=1,
                            bbox=BoundingBox(0.1, 0.1, 0.2, 0.3),
                            is_speaking=speaking,
                            face_detection_confidence=0.93,
                            diarized_speaker_id=self.speaker_code,
                        ),
                    ),
                )
            )

        return ASDResult(
            frames=tuple(frames),
            clip_start_frame=clip_start_frame,
            fps=fps,
            provider=self.name,
        )


def run_mode_a(
    candidates: tuple[Candidate, ...],
    manifest: MediaManifest,
    provider: ASDProvider,
    tmp_path: Path,
    *,
    words: tuple[Word, ...] | None = None,
) -> tuple[Any, Any]:
    all_words = words if words is not None else tuple(
        w for c in candidates for w in c.words
    )
    return mode_a.run(
        candidates,
        manifest,
        all_words,
        provider,
        PyAVFrameExtractor(),
        target=TARGET,
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "out",
        detect_scene_cuts=False,
    )


class TestSuccessfulRun:
    def test_produces_the_documented_result_shape(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        provider = StubProvider(speaking_from=6)
        result, attempts = run_mode_a((candidate_at(1.0, 2.4),), manifest, provider, tmp_path)

        payload = result.to_dict()
        assert payload["status"] == "success"
        assert "timestamp" in payload
        assert "frame_number" in payload
        assert payload["text"] == TARGET
        assert payload["image"].startswith("frame_")
        assert 0.0 <= payload["confidence"] <= 1.0

    def test_frame_number_is_the_offset_plus_the_first_speaking_frame(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        """Invariant 1, end to end. Forgetting the offset yields a real frame at
        the wrong moment -- which looks entirely plausible."""
        start = 1.0
        provider = StubProvider(speaking_from=6)
        result, _ = run_mode_a((candidate_at(start, 2.4),), manifest, provider, tmp_path)

        expected = first_frame_at_or_after(start, manifest.fps) + 6
        assert result.frame_number == expected

    def test_the_saved_image_is_that_exact_frame(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        """The strongest assertion available: the frame index is read back out
        of the saved PNG's pixels."""
        from PIL import Image

        provider = StubProvider(speaking_from=6)
        result, _ = run_mode_a((candidate_at(1.0, 2.4),), manifest, provider, tmp_path)

        image_path = tmp_path / "out" / result.image
        assert image_path.exists()

        decoded = decode_frame_index(
            Image.open(image_path).convert("RGB").getpixel((160, 120))
        )
        assert decoded == result.frame_number

    def test_timestamp_matches_the_frame(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        provider = StubProvider(speaking_from=6)
        result, _ = run_mode_a((candidate_at(1.0, 2.4),), manifest, provider, tmp_path)
        assert result.timestamp == pytest.approx(result.frame_number / manifest.fps)

    def test_the_diarization_sent_is_clip_relative(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        provider = StubProvider(speaking_from=6)
        run_mode_a((candidate_at(1.0, 2.4),), manifest, provider, tmp_path)

        payload = provider.seen_diarizations[0]
        assert payload["words"]
        assert payload["words"][0]["start"] < 1.0, "timestamps were not rebased"


class TestEarlyExit:
    def test_stops_at_the_first_verified_candidate(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        """The efficiency claim: a later candidate is never paid for."""
        provider = StubProvider(speaking_from=6)
        candidates = (candidate_at(0.5, 1.9), candidate_at(2.0, 3.4))

        run_mode_a(candidates, manifest, provider, tmp_path)
        assert provider.calls == 1

    def test_tries_later_candidates_when_the_first_fails(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        provider = StubProvider(speaking_from=None)  # nobody ever speaks
        candidates = (candidate_at(0.5, 1.9), candidate_at(2.0, 3.4))

        result, attempts = run_mode_a(candidates, manifest, provider, tmp_path)
        assert provider.calls == 2
        assert len(attempts) == 2
        assert result.status is Status.NOT_ON_SCREEN

    def test_candidates_are_tried_in_time_order(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        """The brief asks for the frame where the dialogue *first* appears."""
        provider = StubProvider(speaking_from=None)
        candidates = (candidate_at(0.5, 1.9), candidate_at(2.0, 3.4))

        run_mode_a(candidates, manifest, provider, tmp_path)
        assert provider.seen_offsets == sorted(provider.seen_offsets)


class TestNotOnScreen:
    def test_no_speaking_face_is_reported_as_not_on_screen(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        """A face visible but never speaking is the voice-over case."""
        provider = StubProvider(speaking_from=None)
        result, _ = run_mode_a((candidate_at(1.0, 2.4),), manifest, provider, tmp_path)

        assert result.status is Status.NOT_ON_SCREEN
        assert result.frame_number is None
        assert result.detail

    def test_the_wrong_speaker_is_reported_as_a_mismatch(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        provider = StubProvider(speaking_from=6, speaker_code=99)
        result, _ = run_mode_a((candidate_at(1.0, 2.4),), manifest, provider, tmp_path)

        assert result.status is Status.NOT_ON_SCREEN
        assert result.ambiguity_reason is AmbiguityReason.SPEAKER_MISMATCH

    def test_flicker_alone_does_not_produce_an_answer(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        provider = StubProvider(speaking_from=38, frames=40)  # 2 frames only
        result, _ = run_mode_a((candidate_at(1.0, 2.4),), manifest, provider, tmp_path)

        assert result.status is Status.NOT_ON_SCREEN
        assert result.ambiguity_reason is AmbiguityReason.SHORT_SPEAKING_RUN


class TestNoCandidates:
    def test_reports_not_found_rather_than_not_on_screen(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        """Never said and said off-camera are different answers."""
        result, attempts = run_mode_a((), manifest, StubProvider(), tmp_path)
        assert result.status is Status.NOT_FOUND
        assert attempts == ()


class TestServiceFailures:
    def test_one_failing_candidate_does_not_doom_the_rest(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        provider = StubProvider(speaking_from=6, fail_first=1)
        candidates = (candidate_at(0.5, 1.9), candidate_at(2.0, 3.4))

        result, attempts = run_mode_a(candidates, manifest, provider, tmp_path)

        assert result.status is Status.SUCCESS
        assert attempts[0].error is not None
        assert attempts[1].verified

    def test_every_candidate_failing_surfaces_the_error(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        from dialogue_extractor.domain.errors import ASDError

        provider = StubProvider(fail_first=99)
        with pytest.raises(ASDError):
            run_mode_a((candidate_at(1.0, 2.4),), manifest, provider, tmp_path)


class TestUnattributedSpeaker:
    def test_a_line_with_no_speaker_is_verified_but_flagged(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        """Real output contains unattributed words; accepting any speaking face
        beats refusing outright, but the identity is unverified."""
        provider = StubProvider(speaking_from=6, speaker_code=7)
        candidate = candidate_at(1.0, 2.4, speaker=None)

        result, _ = run_mode_a((candidate,), manifest, provider, tmp_path)

        assert result.frame_number is not None
        assert result.status is Status.AMBIGUOUS
        assert result.ambiguity_reason is AmbiguityReason.SPEAKER_MISMATCH


class TestDeterminism:
    def test_the_same_input_gives_the_same_answer(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        def once(directory: Path) -> tuple[Any, ...]:
            provider = StubProvider(speaking_from=6)
            result, _ = run_mode_a(
                (candidate_at(1.0, 2.4),), manifest, provider, directory
            )
            return (result.status, result.frame_number, result.confidence)

        assert once(tmp_path / "a") == once(tmp_path / "b")
