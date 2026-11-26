"""What the "check it yourself" demo needs from the API: every line, and the video.

A discovery sweep must hand back each line with its time span and verdict --
failures included -- and keep the normalized video playable after the working
file is overwritten by the next run.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dialogue_extractor.api.app import create_app
from dialogue_extractor.config.settings import Settings
from dialogue_extractor.domain.enums import Mode, Status
from dialogue_extractor.domain.models import ExtractionResult, MediaManifest
from dialogue_extractor.pipeline.mode_b import CandidateProgress
from dialogue_extractor.pipeline.orchestrator import (
    Orchestrator,
    PipelineOutcome,
    PipelineRequest,
)


class SweepStub(Orchestrator):
    """Emits three candidates (on, off, failed) and a manifest with a real file."""

    def __init__(self, output_dir: Path, video: Path) -> None:
        self._output_dir = output_dir
        self._video = video

    def run(
        self, request: PipelineRequest, *, on_progress: Any = None, on_candidate: Any = None
    ) -> PipelineOutcome:
        rows = [
            (
                ExtractionResult(
                    Status.SUCCESS, text="hello", timestamp=1.0, frame_number=25, confidence=0.9
                ),
                None,
            ),
            (ExtractionResult(Status.NOT_ON_SCREEN, text="narration", timestamp=5.0), None),
            (None, "service_unavailable"),
        ]
        for i, (result, error) in enumerate(rows):
            on_candidate(
                CandidateProgress(
                    index=i,
                    total=3,
                    processed=i + 1,
                    start=i * 4.0,
                    end=i * 4.0 + 3.5,
                    speaker_id="SPEAKER_00",
                    text=f"line {i}",
                    result=result,
                    error_reason=error,
                    line_start=i * 4.0 + 1.0,
                    line_end=i * 4.0 + 2.0,
                )
            )
        manifest = MediaManifest(
            video_path=self._video,
            audio_path=self._video,
            fps=25.0,
            frame_count=250,
            duration=10.0,
            width=64,
            height=64,
            cfr_forced=True,
            source_was_vfr=False,
            video_sha256="x",
            audio_sha256="y",
        )
        payload = {
            "status": "success",
            "mode": Mode.DISCOVERY.value,
            "results": [],
            "ambiguous": [],
            "not_on_screen": [],
            "candidates_processed": 3,
            "candidates_total": 3,
        }
        return PipelineOutcome(payload=payload, manifest=manifest, diagnostics={"fps": 25.0})


@pytest.fixture
def setup(tmp_path: Path) -> tuple[TestClient, Path]:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        cache_dir=tmp_path / "cache",
    )
    video = tmp_path / "work" / "normalized.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"0" * 64)
    app = create_app(settings, orchestrator=SweepStub(tmp_path / "output", video))
    return TestClient(app), video


def finished(client: TestClient, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body["state"] in {"done", "failed"}:
            return body
        time.sleep(0.01)
    raise AssertionError("job never finished")


def test_every_line_comes_back_with_its_verdict(setup: tuple[TestClient, Path]) -> None:
    client, _ = setup
    body = finished(
        client, client.post("/extract", json={"url": "https://example.com/v"}).json()["job_id"]
    )
    lines = body["lines"]
    assert [ln["status"] for ln in lines] == ["success", "not_on_screen", "error"]
    assert lines[2]["reason"] == "service_unavailable"
    assert lines[0]["start"] == pytest.approx(1.0)  # the line itself, not the padded window
    assert lines[0]["frame_number"] == 25


def test_the_video_survives_the_working_file_being_overwritten(
    setup: tuple[TestClient, Path],
) -> None:
    client, video = setup
    body = finished(
        client, client.post("/extract", json={"url": "https://example.com/v"}).json()["job_id"]
    )
    original = video.read_bytes()
    video.write_bytes(b"next run's video")  # what the next job does to the work file
    response = client.get(body["video"])
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert response.content == original


def test_target_jobs_keep_no_video(setup: tuple[TestClient, Path]) -> None:
    client, _ = setup
    body = finished(
        client,
        client.post("/extract", json={"url": "https://example.com/v", "dialogue": "a line"}).json()[
            "job_id"
        ],
    )
    assert "video" not in body


def test_unknown_job_video_is_404(setup: tuple[TestClient, Path]) -> None:
    client, _ = setup
    response = client.get("/jobs/nope/video")
    assert response.status_code == 404
    assert response.json()["detail"]["reason"] == "no_video"


def test_listing_stays_light(setup: tuple[TestClient, Path]) -> None:
    client, _ = setup
    finished(
        client, client.post("/extract", json={"url": "https://example.com/v"}).json()["job_id"]
    )
    listed = client.get("/jobs").json()["jobs"][0]
    assert "lines" not in listed and "partial" not in listed
