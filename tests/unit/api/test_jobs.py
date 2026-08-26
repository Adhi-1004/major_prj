"""The in-process job store.

A full extraction takes minutes, so the HTTP layer hands the work to a thread
and lets clients poll. Two properties matter and neither is obvious from the
code: a job always reaches a terminal state, and concurrent readers never see a
half-written record. Both are tested here without any pipeline at all.
"""

from __future__ import annotations

import threading
import time

import pytest

from dialogue_extractor.api.jobs import Job, JobState, JobStore
from dialogue_extractor.domain.enums import StageName, Status
from dialogue_extractor.pipeline.orchestrator import PipelineRequest

TARGET = PipelineRequest(url="https://example.com/v", dialogue="a line")
DISCOVERY = PipelineRequest(url="https://example.com/v")


def wait_for(store: JobStore, job_id: str, *, timeout: float = 5.0) -> Job:
    """Block until a job is terminal, or fail the test."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = store.get(job_id)
        assert job is not None
        if job.state in {JobState.DONE, JobState.FAILED}:
            return job
        time.sleep(0.005)
    raise AssertionError(f"job {job_id} never finished")


class TestLifecycle:
    def test_a_submitted_job_gets_an_id_and_finishes(self) -> None:
        store = JobStore(run=lambda s, j: s.record_result(j, {"status": "success"}, {}))
        job = store.submit(TARGET)

        assert job.id
        assert wait_for(store, job.id).state is JobState.DONE

    def test_the_result_is_kept(self) -> None:
        store = JobStore(
            run=lambda s, j: s.record_result(j, {"status": "success", "frame_number": 7}, {})
        )
        job = wait_for(store, store.submit(TARGET).id)
        assert job.result == {"status": "success", "frame_number": 7}

    def test_progress_is_visible_before_the_result(self) -> None:
        released = threading.Event()

        def run(store: JobStore, job: Job) -> None:
            store.record_progress(job, StageName.TRANSCRIPTION, "transcribing")
            released.wait(timeout=5)
            store.record_result(job, {"status": "success"}, {})

        store = JobStore(run=run)
        job = store.submit(TARGET)

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            live = store.get(job.id)
            assert live is not None
            if live.stage == StageName.TRANSCRIPTION.value:
                break
            time.sleep(0.005)

        live = store.get(job.id)
        assert live is not None
        assert live.state is JobState.RUNNING
        assert live.result is None
        released.set()
        wait_for(store, job.id)

    def test_an_unknown_id_is_none_not_an_error(self) -> None:
        assert JobStore(run=lambda s, j: None).get("nope") is None


class TestTerminalStateIsGuaranteed:
    """A client polling a job that stopped updating would wait forever."""

    def test_a_crashing_runner_still_ends_the_job(self) -> None:
        def run(store: JobStore, job: Job) -> None:
            raise RuntimeError("the runner itself broke")

        store = JobStore(run=run)
        job = wait_for(store, store.submit(TARGET).id)

        assert job.state is JobState.FAILED
        assert job.result is not None
        assert job.result["reason"] == "job_crashed"

    def test_a_crash_document_is_the_documented_shape(self) -> None:
        store = JobStore(run=lambda s, j: 1 / 0)
        job = wait_for(store, store.submit(TARGET).id)

        assert job.result is not None
        assert set(job.result) >= {"status", "stage", "reason", "detail"}
        assert job.result["status"] == Status.ERROR.value

    def test_a_runner_that_does_nothing_still_ends_the_job(self) -> None:
        """Recording no result is not the same as never finishing."""
        store = JobStore(run=lambda s, j: None)
        finished = wait_for(store, store.submit(TARGET).id)
        assert finished.state is JobState.DONE
        assert finished.result is None

    def test_finished_at_is_always_set(self) -> None:
        store = JobStore(run=lambda s, j: 1 / 0)
        assert wait_for(store, store.submit(TARGET).id).finished_at is not None


class TestErrorResultsFailTheJob:
    def test_a_pipeline_error_marks_the_job_failed(self) -> None:
        """The pipeline caught it, so the runner returned normally -- but the
        job did not succeed and must not claim to."""
        store = JobStore(
            run=lambda s, j: s.record_result(
                j, {"status": "error", "stage": "ingestion", "reason": "http_404"}, {}
            )
        )
        assert wait_for(store, store.submit(TARGET).id).state is JobState.FAILED

    @pytest.mark.parametrize("status", ["success", "ambiguous", "not_on_screen", "not_found"])
    def test_a_correct_no_is_still_a_done_job(self, status: str) -> None:
        """"Spoken off camera" is an answer. Only a failure is a failure."""
        store = JobStore(run=lambda s, j: s.record_result(j, {"status": status}, {}))
        assert wait_for(store, store.submit(TARGET).id).state is JobState.DONE


class TestConcurrency:
    def test_many_jobs_run_without_losing_any(self) -> None:
        store = JobStore(run=lambda s, j: s.record_result(j, {"status": "success"}, {}))
        ids = [store.submit(TARGET).id for _ in range(25)]
        for job_id in ids:
            wait_for(store, job_id)
        assert len({j.id for j in store.list()}) == 25

    def test_ids_are_unique(self) -> None:
        store = JobStore(run=lambda s, j: None)
        assert len({store.submit(TARGET).id for _ in range(50)}) == 50

    def test_reading_while_writing_never_sees_a_torn_record(self) -> None:
        """The reason the store holds a lock rather than a plain dict."""
        stop = threading.Event()

        def run(store: JobStore, job: Job) -> None:
            for _ in range(200):
                store.record_progress(job, StageName.TRANSCRIPTION, "x" * 200)
                store.record_progress(job, StageName.DIARIZATION, "y" * 200)
            store.record_result(job, {"status": "success"}, {})

        store = JobStore(run=run)
        job = store.submit(TARGET)

        seen: list[dict[str, object]] = []
        while not stop.is_set() and len(seen) < 300:
            live = store.get(job.id)
            if live is None:
                break
            seen.append(live.to_dict())
            if live.state in {JobState.DONE, JobState.FAILED}:
                break

        for snapshot in seen:
            assert snapshot["job_id"] == job.id
            assert snapshot["state"] in {s.value for s in JobState}


class TestEviction:
    def test_the_store_does_not_grow_without_bound(self) -> None:
        store = JobStore(
            run=lambda s, j: s.record_result(j, {"status": "success"}, {}), max_jobs=5
        )
        for _ in range(12):
            wait_for(store, store.submit(TARGET).id)
        assert len(store.list()) <= 5

    def test_a_running_job_is_never_evicted(self) -> None:
        """Evicting one would leave its worker writing into a record no client
        can reach."""
        release = threading.Event()

        def run(store: JobStore, job: Job) -> None:
            release.wait(timeout=5)
            store.record_result(job, {"status": "success"}, {})

        store = JobStore(run=run, max_jobs=3)
        held = [store.submit(TARGET).id for _ in range(6)]

        live = {j.id for j in store.list()}
        assert set(held) <= live  # nothing dropped while all are running

        release.set()
        for job_id in held:
            wait_for(store, job_id)

    def test_the_newest_survives(self) -> None:
        store = JobStore(
            run=lambda s, j: s.record_result(j, {"status": "success"}, {}), max_jobs=3
        )
        ids = []
        for _ in range(8):
            job = store.submit(TARGET)
            wait_for(store, job.id)
            ids.append(job.id)
        assert ids[-1] in {j.id for j in store.list()}


class TestSerialization:
    def test_the_mode_comes_from_the_request(self) -> None:
        store = JobStore(run=lambda s, j: None)
        assert store.submit(DISCOVERY).to_dict()["mode"] == "discovery"
        assert store.submit(TARGET).to_dict()["mode"] == "target"

    def test_a_target_job_reports_the_phrase(self) -> None:
        store = JobStore(run=lambda s, j: None)
        assert store.submit(TARGET).to_dict()["dialogue"] == "a line"

    def test_a_discovery_job_has_no_dialogue_field(self) -> None:
        store = JobStore(run=lambda s, j: None)
        assert "dialogue" not in store.submit(DISCOVERY).to_dict()

    def test_elapsed_is_reported_once_finished(self) -> None:
        store = JobStore(run=lambda s, j: s.record_result(j, {"status": "success"}, {}))
        payload = wait_for(store, store.submit(TARGET).id).to_dict()
        assert payload["elapsed_seconds"] >= 0

    def test_it_is_json_serializable(self) -> None:
        import json

        store = JobStore(run=lambda s, j: s.record_result(j, {"status": "success"}, {}))
        payload = wait_for(store, store.submit(TARGET).id).to_dict()
        assert json.loads(json.dumps(payload))["state"] == "done"
