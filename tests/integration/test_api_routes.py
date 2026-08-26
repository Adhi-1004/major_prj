"""The HTTP layer, with the pipeline stubbed.

The API is a thin shell over `Orchestrator.run`, so what is worth testing here
is the shell: does a bad request get refused at the edge, does a long job report
progress while it runs, does a correct "no" come back as 200 rather than an
error, and can a caller actually fetch the frame the result names.

No network, no model, no credentials -- the orchestrator is substituted whole.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dialogue_extractor.api.app import create_app
from dialogue_extractor.config.settings import Settings
from dialogue_extractor.domain.enums import StageName
from dialogue_extractor.pipeline.orchestrator import (
    Orchestrator,
    PipelineOutcome,
    PipelineRequest,
)

SUCCESS = {
    "status": "success",
    "mode": "target",
    "timestamp": "00:00:39.373",
    "frame_number": 955,
    "text": "a line of dialogue",
    "confidence": 0.87,
    "image": "frame_955.png",
}


class StubOrchestrator(Orchestrator):
    """Answers with a scripted payload, optionally after reporting progress."""

    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        *,
        delay: float = 0.0,
        stages: tuple[StageName, ...] = (),
    ) -> None:
        self.payload = payload if payload is not None else dict(SUCCESS)
        self.delay = delay
        self.stages = stages
        self.seen: list[PipelineRequest] = []

    def run(
        self,
        request: PipelineRequest,
        *,
        on_progress: Any = None,
        on_candidate: Any = None,
    ) -> PipelineOutcome:
        self.seen.append(request)
        self.saw_candidate_hook = on_candidate is not None
        for stage in self.stages:
            if on_progress is not None:
                on_progress(stage, f"doing {stage.value}")
            time.sleep(self.delay)
        return PipelineOutcome(
            payload=self.payload, diagnostics={"fps": 23.976, "words": 325}
        )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings that cannot see the developer's real .env."""
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        cache_dir=tmp_path / "cache",
    )


def client_for(settings: Settings, orchestrator: Orchestrator) -> TestClient:
    return TestClient(create_app(settings, orchestrator=orchestrator))


def poll(client: TestClient, job_id: str, *, timeout: float = 5.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body["state"] in {"done", "failed"}:
            return body
        time.sleep(0.01)
    raise AssertionError("job never finished")


class TestJobLifecycle:
    def test_submit_poll_done(self, settings: Settings) -> None:
        client = client_for(settings, StubOrchestrator())

        accepted = client.post(
            "/extract", json={"url": "https://example.com/v", "dialogue": "a line"}
        )
        assert accepted.status_code == 202
        job_id = accepted.json()["job_id"]

        body = poll(client, job_id)
        assert body["state"] == "done"
        assert body["result"]["frame_number"] == 955

    def test_the_response_says_where_to_poll(self, settings: Settings) -> None:
        client = client_for(settings, StubOrchestrator())
        body = client.post("/extract", json={"url": "https://example.com/v"}).json()
        assert body["poll"] == f"/jobs/{body['job_id']}"
        assert client.get(body["poll"]).status_code == 200

    def test_202_not_200(self, settings: Settings) -> None:
        """The work was accepted, not completed. 200 would claim otherwise."""
        client = client_for(settings, StubOrchestrator())
        assert (
            client.post("/extract", json={"url": "https://example.com/v"}).status_code
            == 202
        )

    def test_an_unknown_job_is_404_with_the_documented_shape(
        self, settings: Settings
    ) -> None:
        client = client_for(settings, StubOrchestrator())
        response = client.get("/jobs/deadbeef")
        assert response.status_code == 404
        assert response.json()["detail"]["reason"] == "unknown_job"

    def test_diagnostics_come_back_with_the_result(self, settings: Settings) -> None:
        client = client_for(settings, StubOrchestrator())
        job_id = client.post("/extract", json={"url": "https://example.com/v"}).json()[
            "job_id"
        ]
        assert poll(client, job_id)["diagnostics"]["words"] == 325


class TestProgressWhileRunning:
    def test_the_current_stage_is_reported(self, settings: Settings) -> None:
        """A five-minute job that only ever says "running" is not much use."""
        stub = StubOrchestrator(
            delay=0.05,
            stages=(
                StageName.INGESTION,
                StageName.TRANSCRIPTION,
                StageName.DIARIZATION,
            ),
        )
        client = client_for(settings, stub)
        job_id = client.post("/extract", json={"url": "https://example.com/v"}).json()[
            "job_id"
        ]

        seen: set[str] = set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            body = client.get(f"/jobs/{job_id}").json()
            if body.get("stage"):
                seen.add(body["stage"])
            if body["state"] in {"done", "failed"}:
                break
            time.sleep(0.01)

        assert seen  # at least one stage was observed mid-run
        assert seen <= {s.value for s in StageName}


class TestAnswersAreNotHttpErrors:
    @pytest.mark.parametrize("status", ["not_found", "not_on_screen"])
    def test_a_correct_no_is_200_and_done(
        self, status: str, settings: Settings
    ) -> None:
        """Returning 404 for "this line is spoken off camera" would conflate the
        answer with the job being missing."""
        client = client_for(settings, StubOrchestrator({"status": status}))
        job_id = client.post("/extract", json={"url": "https://example.com/v"}).json()[
            "job_id"
        ]
        body = poll(client, job_id)
        assert body["state"] == "done"
        assert body["result"]["status"] == status

    def test_a_pipeline_error_marks_the_job_failed_but_still_answers_200(
        self, settings: Settings
    ) -> None:
        client = client_for(
            settings,
            StubOrchestrator(
                {
                    "status": "error",
                    "stage": "ingestion",
                    "reason": "geo_restricted",
                    "detail": "blocked",
                }
            ),
        )
        job_id = client.post("/extract", json={"url": "https://example.com/v"}).json()[
            "job_id"
        ]
        body = poll(client, job_id)
        assert body["state"] == "failed"
        assert body["result"]["reason"] == "geo_restricted"


class TestRequestValidation:
    @pytest.mark.parametrize(
        ("payload", "why"),
        [
            ({}, "no url at all"),
            ({"url": ""}, "empty url"),
            ({"url": "   "}, "whitespace url"),
            ({"url": "javascript:alert(1)"}, "refused scheme"),
            ({"url": "https://e.com/v", "max_candidates": -1}, "negative budget"),
            ({"url": "https://e.com/v", "time_budget_seconds": -5}, "negative time"),
            ({"url": "https://e.com/v", "min_speaking_ms": 0}, "zero speaking floor"),
            ({"url": "https://e.com/v", "match_score_floor": 101}, "score over 100"),
            ({"url": "https://e.com/v", "window_pad_seconds": -1}, "negative pad"),
        ],
    )
    def test_bad_requests_are_refused_at_the_edge(
        self, payload: dict[str, Any], why: str, settings: Settings
    ) -> None:
        client = client_for(settings, StubOrchestrator())
        assert client.post("/extract", json=payload).status_code == 422, why

    def test_every_rejection_uses_the_same_code(self, settings: Settings) -> None:
        """A bad scheme and an empty string are one class of mistake.

        They used to return 400 and 422 respectively, which a client would have
        had to special-case.
        """
        client = client_for(settings, StubOrchestrator())
        codes = {
            client.post("/extract", json=body).status_code
            for body in ({"url": ""}, {"url": "javascript:alert(1)"}, {})
        }
        assert codes == {422}

    def test_a_refusal_explains_itself(self, settings: Settings) -> None:
        client = client_for(settings, StubOrchestrator())
        body = client.post("/extract", json={"url": "javascript:alert(1)"}).json()
        assert "detail" in body
        assert str(body["detail"])


class TestModeSelection:
    def test_a_phrase_selects_target(self, settings: Settings) -> None:
        client = client_for(settings, StubOrchestrator())
        body = client.post(
            "/extract", json={"url": "https://e.com/v", "dialogue": "a line"}
        ).json()
        assert body["mode"] == "target"

    def test_omitting_it_selects_discovery(self, settings: Settings) -> None:
        client = client_for(settings, StubOrchestrator())
        assert (
            client.post("/extract", json={"url": "https://e.com/v"}).json()["mode"]
            == "discovery"
        )

    def test_a_blank_phrase_is_not_a_target(self, settings: Settings) -> None:
        """It would otherwise search for nothing and report not_found."""
        client = client_for(settings, StubOrchestrator())
        assert (
            client.post(
                "/extract", json={"url": "https://e.com/v", "dialogue": "   "}
            ).json()["mode"]
            == "discovery"
        )

    def test_the_tuning_values_reach_the_pipeline(self, settings: Settings) -> None:
        stub = StubOrchestrator()
        client = client_for(settings, stub)
        job_id = client.post(
            "/extract",
            json={
                "url": "https://e.com/v",
                "max_candidates": 7,
                "min_speaking_ms": 200,
                "detect_scene_cuts": False,
            },
        ).json()["job_id"]
        poll(client, job_id)

        assert stub.seen[0].max_candidates == 7
        assert stub.seen[0].min_speaking_ms == 200
        assert stub.seen[0].detect_scene_cuts is False


class TestListing:
    def test_it_lists_jobs_without_their_results(self, settings: Settings) -> None:
        """The list is for scanning. Inlining every result would make a page of
        it unreadable, and each result is one GET away."""
        client = client_for(settings, StubOrchestrator())
        for _ in range(3):
            client.post("/extract", json={"url": "https://e.com/v"})

        body = client.get("/jobs").json()
        assert body["count"] == 3
        assert all("result" not in job for job in body["jobs"])

    def test_an_empty_store_lists_nothing(self, settings: Settings) -> None:
        client = client_for(settings, StubOrchestrator())
        assert client.get("/jobs").json() == {"jobs": [], "count": 0}


class TestFrames:
    def test_a_named_frame_can_be_fetched(self, settings: Settings) -> None:
        """A result naming an image the caller cannot fetch is half an answer."""
        frames = settings.output_dir / "frames"
        frames.mkdir(parents=True)
        (frames / "frame_955.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        client = client_for(settings, StubOrchestrator())
        response = client.get("/frames/frame_955.png")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_only_the_frames_directory_is_served(
        self, settings: Settings
    ) -> None:
        """A stray PNG in the output root is not a frame, and is not served.

        Both modes write to output/frames now. Serving the root as a fallback
        is exactly what created the collision the next test guards against.
        """
        settings.output_dir.mkdir(parents=True)
        (settings.output_dir / "frame_144.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        client = client_for(settings, StubOrchestrator())
        assert client.get("/frames/frame_144.png").status_code == 404

    def test_a_root_png_can_never_shadow_a_real_frame(
        self, settings: Settings
    ) -> None:
        """Regression guard for a real bug.

        Target mode used to write output/frame_N.png while discovery wrote
        output/frames/frame_N.png, and the resolver checked output/frames
        first. Once both held the same frame number, a target-mode result was
        served discovery's frame -- potentially from an entirely different
        video, for the one output the brief actually asks for.
        """
        frames = settings.output_dir / "frames"
        frames.mkdir(parents=True)
        (settings.output_dir / "frame_144.png").write_bytes(b"WRONG-FILE")
        (frames / "frame_144.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"RIGHT")

        client = client_for(settings, StubOrchestrator())
        response = client.get("/frames/frame_144.png")

        assert response.status_code == 200
        assert b"RIGHT" in response.content
        assert b"WRONG-FILE" not in response.content

    def test_a_missing_frame_is_404(self, settings: Settings) -> None:
        settings.output_dir.mkdir(parents=True)
        client = client_for(settings, StubOrchestrator())
        assert client.get("/frames/frame_1.png").status_code == 404

    @pytest.mark.parametrize(
        "name",
        [
            "..%2f..%2f.env",
            "..%2F.env",
            "frames%2f..%2f..%2f.env",
            "secrets.txt",
            "config.json",
        ],
    )
    def test_it_refuses_to_serve_anything_else(
        self, name: str, settings: Settings
    ) -> None:
        """This endpoint hands out files named by a value that arrived over the
        network, which is exactly when traversal matters."""
        settings.output_dir.mkdir(parents=True)
        (settings.output_dir.parent / ".env").write_text("HF_TOKEN=real")

        client = client_for(settings, StubOrchestrator())
        assert client.get(f"/frames/{name}").status_code == 404


class TestHealth:
    def test_it_answers_without_credentials(self, settings: Settings) -> None:
        """The endpoint whose job is to report a missing key must not need one."""
        client = client_for(settings, StubOrchestrator())
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_it_reports_presence_not_values(self, settings: Settings) -> None:
        with_key = Settings(
            _env_file=None,  # type: ignore[call-arg]
            nvidia_api_key="nvapi-secret-value-here",
            output_dir=settings.output_dir,
        )
        client = client_for(with_key, StubOrchestrator())
        body = client.get("/health").json()

        assert body["credentials"]["nvidia_api_key"] is True
        assert "nvapi-secret-value-here" not in str(body)

    def test_it_lists_the_providers(self, settings: Settings) -> None:
        client = client_for(settings, StubOrchestrator())
        assert "nvidia" in client.get("/health").json()["providers"]

    def test_it_counts_running_jobs(self, settings: Settings) -> None:
        client = client_for(settings, StubOrchestrator())
        for _ in range(2):
            client.post("/extract", json={"url": "https://e.com/v"})
        assert client.get("/health").json()["jobs"]["total"] == 2


class TestOpenAPI:
    def test_the_schema_is_generated(self, settings: Settings) -> None:
        client = client_for(settings, StubOrchestrator())
        schema = client.get("/openapi.json").json()
        assert "/extract" in schema["paths"]
        assert "/jobs/{job_id}" in schema["paths"]
        assert "/health" in schema["paths"]


class TestLiveDiscoveryProgress:
    """A polling client sees a sweep advance instead of a frozen "running".

    This is the whole reason `on_candidate` exists: without it a four-minute
    discovery run reports one stage message and then nothing until it finishes.
    """

    def test_progress_and_partials_appear_while_running(
        self, settings: Settings
    ) -> None:
        import threading

        from dialogue_extractor.domain.enums import Status as PipelineStatus
        from dialogue_extractor.domain.models import ExtractionResult
        from dialogue_extractor.pipeline.mode_b import CandidateProgress

        release = threading.Event()
        reported = threading.Event()

        class SweepingOrchestrator(StubOrchestrator):
            def run(self, request: PipelineRequest, *, on_progress=None, **kw):  # type: ignore[no-untyped-def]
                on_candidate = kw.get("on_candidate")
                assert on_candidate is not None, "the API must pass on_candidate"
                on_candidate(
                    CandidateProgress(
                        index=0,
                        total=8,
                        processed=1,
                        start=5.6,
                        end=9.3,
                        speaker_id="SPEAKER_05",
                        text="a line found early",
                        result=ExtractionResult(
                            status=PipelineStatus.SUCCESS,
                            text="a line found early",
                            timestamp=6.006,
                            frame_number=144,
                            confidence=0.88,
                        ),
                    )
                )
                reported.set()
                release.wait(timeout=5)
                return PipelineOutcome(payload={"status": "success", "results": []})

        client = client_for(settings, SweepingOrchestrator())
        job_id = client.post("/extract", json={"url": "https://e.com/v"}).json()["job_id"]

        assert reported.wait(timeout=5)
        body = client.get(f"/jobs/{job_id}").json()

        assert body["state"] == "running"
        assert body["progress"] == {"current": 1, "total": 8}
        assert len(body["partial"]) == 1
        assert body["partial"][0]["frame_number"] == 144

        release.set()
        poll(client, job_id)

    def test_a_job_that_reports_nothing_omits_both_keys(
        self, settings: Settings
    ) -> None:
        """Target mode never sweeps, so it must not grow empty scaffolding."""
        client = client_for(settings, StubOrchestrator())
        job_id = client.post(
            "/extract", json={"url": "https://e.com/v", "dialogue": "a line"}
        ).json()["job_id"]

        body = poll(client, job_id)
        assert "progress" not in body
        assert "partial" not in body

    def test_the_listing_stays_light(self, settings: Settings) -> None:
        """`partial` can hold every result found so far; it has no place in a
        listing whose job is to be scannable."""
        client = client_for(settings, StubOrchestrator())
        client.post("/extract", json={"url": "https://e.com/v"})

        entry = client.get("/jobs").json()["jobs"][0]
        assert "result" not in entry
        assert "diagnostics" not in entry
