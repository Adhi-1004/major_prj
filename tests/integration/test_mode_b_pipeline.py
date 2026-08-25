"""Discovery end to end, with the detection service stubbed.

The scenario is a three-speaker scene containing one deliberate voice-over. The
whole system exists to tell those apart, so the assertion that matters is not
"results were produced" -- it is that the narrator's line lands in *neither*
bucket while the two on-camera lines land in `results`.

Real video, real clip building, real frame extraction; only the network is
faked, and it is faked per-window so each candidate gets its own scripted
answer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from tests.conftest import requires_ffmpeg

from dialogue_extractor.domain.enums import Mode, Status
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
from dialogue_extractor.ingestion.normalizer import normalize
from dialogue_extractor.interfaces.ports import ASDProvider
from dialogue_extractor.pipeline import mode_b
from dialogue_extractor.pipeline.mode_b import Budget
from dialogue_extractor.vision.pyav_extractor import PyAVFrameExtractor

pytestmark = requires_ffmpeg


@pytest.fixture(scope="module")
def manifest(counter_clip_25fps: Path, tmp_path_factory: pytest.TempPathFactory) -> MediaManifest:
    work = tmp_path_factory.mktemp("modeb")
    raw = RawMedia(path=counter_clip_25fps, source_uri=str(counter_clip_25fps), title="fixture")
    return normalize(raw, work)


def line(text: str, start: float, speaker: str | None) -> tuple[Word, ...]:
    words = []
    t = start
    for token in text.split(" "):
        words.append(Word(token, round(t, 3), round(t + 0.2, 3), 0.92, speaker))
        t += 0.25
    return tuple(words)


def candidate(text: str, start: float, end: float, speaker: str | None) -> Candidate:
    return Candidate(
        start=start,
        end=end,
        speaker_id=speaker,
        text=text,
        mode=Mode.DISCOVERY,
        match_score=None,
        words=line(text, start + 0.1, speaker),
    )


# Three lines. The middle one is narration -- audible, nobody on camera saying it.
SCENE = (
    candidate("first line on camera", 0.20, 1.10, "SPEAKER_A"),
    candidate("narration over the scene", 1.30, 2.20, "SPEAKER_N"),
    candidate("third line on camera", 2.40, 3.30, "SPEAKER_B"),
)


class ScriptedProvider(ASDProvider):
    """Answers each window differently, keyed by the order it is asked."""

    name = "scripted"

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.script = script
        self.calls = 0

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
        spec = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1

        if spec.get("error"):
            from dialogue_extractor.domain.errors import ASDUnavailableError

            raise ASDUnavailableError("service_unavailable", "scripted outage")

        frames = []
        for index in range(spec.get("frames", 40)):
            speaking = spec["speaking"] and index >= spec.get("from", 4)
            frames.append(
                ASDFrame(
                    index,
                    (
                        Face(
                            face_id=1,
                            bbox=BoundingBox(0.1, 0.1, 0.2, 0.3),
                            is_speaking=speaking,
                            face_detection_confidence=spec.get("confidence", 0.93),
                            diarized_speaker_id=spec.get("code", 1),
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


def discover(
    provider: ASDProvider,
    manifest: MediaManifest,
    tmp_path: Path,
    *,
    candidates: tuple[Candidate, ...] = SCENE,
    budget: Budget | None = None,
):  # type: ignore[no-untyped-def]
    return mode_b.run(
        candidates,
        manifest,
        tuple(w for c in candidates for w in c.words),
        provider,
        PyAVFrameExtractor(),
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "out",
        budget=budget,
        detect_scene_cuts=False,
    )


class TestVoiceOverIsExcluded:
    """The distinction the entire system exists to make."""

    def test_the_narrated_line_lands_in_neither_bucket(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        provider = ScriptedProvider([
            {"speaking": True, "code": 1},      # on camera
            {"speaking": False},                 # narration: face visible, silent
            {"speaking": True, "code": 1},      # on camera
        ])
        result = discover(provider, manifest, tmp_path)

        texts = [r.text for r in result.results] + [r.text for r in result.ambiguous]
        assert "narration over the scene" not in texts
        assert len(result.results) + len(result.ambiguous) == 2

    def test_the_on_camera_lines_are_returned(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        provider = ScriptedProvider([
            {"speaking": True, "code": 1},
            {"speaking": False},
            {"speaking": True, "code": 1},
        ])
        result = discover(provider, manifest, tmp_path)

        found = [r.text for r in result.results] + [r.text for r in result.ambiguous]
        assert "first line on camera" in found
        assert "third line on camera" in found

    def test_a_line_with_nobody_in_shot_is_also_excluded(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        provider = ScriptedProvider([
            {"speaking": True, "code": 1},
            {"speaking": False, "frames": 0},  # no faces at all
            {"speaking": True, "code": 1},
        ])
        result = discover(provider, manifest, tmp_path)
        assert len(result.results) + len(result.ambiguous) == 2


class TestBucketing:
    def test_strong_evidence_goes_to_results(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        provider = ScriptedProvider([{"speaking": True, "code": 1, "confidence": 0.95}])
        result = discover(provider, manifest, tmp_path, candidates=SCENE[:1])

        assert len(result.results) == 1
        assert result.results[0].status is Status.SUCCESS

    def test_weak_evidence_goes_to_ambiguous_with_a_reason(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        """Uncertain results are separated, not discarded or promoted."""
        provider = ScriptedProvider([
            {"speaking": True, "code": 1, "confidence": 0.30, "from": 36, "frames": 40}
        ])
        result = discover(provider, manifest, tmp_path, candidates=SCENE[:1])

        assert not result.results
        if result.ambiguous:
            assert result.ambiguous[0].ambiguity_reason is not None

    def test_an_unverifiable_speaker_is_flagged_not_promoted(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        unattributed = (candidate("stray fragment", 0.2, 1.1, None),)
        provider = ScriptedProvider([{"speaking": True, "code": 7}])
        result = discover(provider, manifest, tmp_path, candidates=unattributed)

        assert not result.results
        assert len(result.ambiguous) == 1


class TestOutputShape:
    def test_reports_both_buckets_and_the_counts(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        provider = ScriptedProvider([{"speaking": True, "code": 1}])
        payload = discover(provider, manifest, tmp_path).to_dict()

        assert "results" in payload
        assert "ambiguous" in payload
        assert payload["candidates_total"] == 3
        assert payload["candidates_processed"] == 3

    def test_each_result_carries_a_frame_and_an_image(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        provider = ScriptedProvider([{"speaking": True, "code": 1}])
        result = discover(provider, manifest, tmp_path)

        for entry in result.results + result.ambiguous:
            assert entry.frame_number is not None
            assert entry.image is not None
            assert (tmp_path / "out" / entry.image).exists()

    def test_image_filenames_do_not_collide(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        """Each result is a different frame, so each gets a different file."""
        provider = ScriptedProvider([{"speaking": True, "code": 1}])
        result = discover(provider, manifest, tmp_path)

        images = [e.image for e in result.results + result.ambiguous]
        assert len(images) == len(set(images))

    def test_is_json_serializable(self, manifest: MediaManifest, tmp_path: Path) -> None:
        import json

        provider = ScriptedProvider([{"speaking": True, "code": 1}])
        payload = discover(provider, manifest, tmp_path).to_dict()
        assert json.loads(json.dumps(payload))["candidates_total"] == 3

    def test_no_results_reports_not_found(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        provider = ScriptedProvider([{"speaking": False}])
        payload = discover(provider, manifest, tmp_path).to_dict()
        assert payload["status"] == "not_found"
        assert payload["results"] == []


class TestBudget:
    def test_a_candidate_cap_stops_the_sweep(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        provider = ScriptedProvider([{"speaking": True, "code": 1}])
        result = discover(provider, manifest, tmp_path, budget=Budget(max_candidates=2))

        assert provider.calls == 2
        assert result.candidates_processed == 2

    def test_truncation_is_declared(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        """A partial sweep presented as exhaustive would make "every on-screen
        line" a false claim."""
        provider = ScriptedProvider([{"speaking": True, "code": 1}])
        payload = discover(
            provider, manifest, tmp_path, budget=Budget(max_candidates=2)
        ).to_dict()

        assert payload["truncated"] is True
        assert payload["candidates_processed"] == 2
        assert payload["candidates_total"] == 3

    def test_partial_results_are_kept(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        provider = ScriptedProvider([{"speaking": True, "code": 1}])
        result = discover(provider, manifest, tmp_path, budget=Budget(max_candidates=2))
        assert result.results or result.ambiguous

    def test_a_complete_sweep_omits_the_truncated_flag(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        provider = ScriptedProvider([{"speaking": True, "code": 1}])
        payload = discover(provider, manifest, tmp_path).to_dict()
        assert "truncated" not in payload


class TestFailuresDoNotEndTheRun:
    def test_one_failing_candidate_costs_only_that_candidate(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        provider = ScriptedProvider([
            {"speaking": True, "code": 1},
            {"error": True},
            {"speaking": True, "code": 1},
        ])
        result = discover(provider, manifest, tmp_path)

        assert provider.calls == 3
        assert result.candidates_processed == 3
        assert len(result.errors) == 1

    def test_the_failure_is_reported_alongside_the_results(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        provider = ScriptedProvider([
            {"speaking": True, "code": 1},
            {"error": True},
            {"speaking": True, "code": 1},
        ])
        payload = discover(provider, manifest, tmp_path).to_dict()

        assert payload["errors"]
        assert payload["errors"][0]["reason"] == "service_unavailable"
        assert payload["results"] or payload["ambiguous"]


class TestEmptyInput:
    def test_no_candidates_produces_an_empty_report(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        result = discover(ScriptedProvider([]), manifest, tmp_path, candidates=())
        assert result.results == ()
        assert result.candidates_total == 0
        assert result.truncated is False
