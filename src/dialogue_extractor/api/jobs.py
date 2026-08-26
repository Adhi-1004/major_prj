"""In-process job store for the HTTP layer.

A full extraction takes minutes -- download, transcribe, diarize, then one
network round-trip per candidate -- so `POST /extract` cannot answer inside a
request. It accepts the work, returns an id, and runs the pipeline on a worker
thread while `GET /jobs/{id}` reports progress.

**Deliberately not a queue.** No Redis, no Celery, no broker. The design
document scopes this to a single-process service, and a real queue would add
infrastructure that nothing here needs; the cost of the choice -- jobs do not
survive a restart -- is stated in the README rather than hidden.

Threads rather than asyncio: the pipeline is synchronous, CPU- and GPU-bound
work in libraries with no async API. Running it on the event loop would block
every other request, including /health.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from dialogue_extractor.domain.enums import StageName, Status
from dialogue_extractor.infra.logging import get_logger
from dialogue_extractor.pipeline.orchestrator import PipelineRequest

log = get_logger(__name__)


class JobState(StrEnum):
    """Lifecycle of a submitted job, distinct from the pipeline's own Status.

    A job that finished cleanly is DONE even when its result says
    `not_on_screen`: the work succeeded, and the answer was no. Collapsing the
    two would make "the line is spoken off camera" indistinguishable from "the
    download failed".
    """

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    """One submitted extraction and everything known about it so far."""

    id: str
    request: PipelineRequest
    state: JobState = JobState.QUEUED
    stage: str | None = None
    message: str | None = None
    result: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": self.id,
            "state": self.state.value,
            "url": self.request.url,
            "mode": self.request.mode.value,
            "created_at": self.created_at.isoformat(),
        }
        if self.request.dialogue:
            payload["dialogue"] = self.request.dialogue
        if self.stage is not None:
            payload["stage"] = self.stage
        if self.message is not None:
            payload["message"] = self.message
        if self.started_at is not None:
            payload["started_at"] = self.started_at.isoformat()
        if self.finished_at is not None:
            payload["finished_at"] = self.finished_at.isoformat()
            payload["elapsed_seconds"] = round(
                (self.finished_at - (self.started_at or self.created_at)).total_seconds(), 2
            )
        if self.result is not None:
            payload["result"] = self.result
        if self.diagnostics:
            payload["diagnostics"] = self.diagnostics
        return payload


RunFn = Callable[["JobStore", Job], None]
"""Does the actual work for one job. Injected so tests need no pipeline.

Takes the store as its first argument rather than closing over it. The runner
needs the store to report progress, and the store needs the runner to do the
work; passing it in breaks what would otherwise be a construction cycle that
could only be closed by reaching into a private attribute.
"""


class JobStore:
    """Thread-safe registry of jobs, with a worker thread per job.

    Every read and write of the shared dict goes through one lock. Progress
    callbacks fire from the worker thread while an HTTP handler may be reading
    the same job, and an unsynchronized dict would eventually hand a client a
    half-written record.
    """

    def __init__(self, run: RunFn, *, max_jobs: int = 100) -> None:
        self._run = run
        self._max_jobs = max_jobs
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    def submit(self, request: PipelineRequest) -> Job:
        """Register a job and start working on it."""
        job = Job(id=uuid.uuid4().hex[:12], request=request)
        with self._lock:
            self._evict_if_full()
            self._jobs[job.id] = job

        thread = threading.Thread(
            target=self._work, args=(job,), name=f"job-{job.id}", daemon=True
        )
        self._threads.append(thread)
        thread.start()
        log.info("api.job_submitted", job_id=job.id, mode=job.request.mode.value)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> tuple[Job, ...]:
        with self._lock:
            return tuple(
                sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            )

    def _evict_if_full(self) -> None:
        """Drop the oldest finished job once the store is full.

        Called with the lock held. Only finished jobs are evicted -- discarding
        a running one would leave its worker writing into a record no client can
        reach.
        """
        if len(self._jobs) < self._max_jobs:
            return
        finished = [
            j for j in self._jobs.values()
            if j.state in {JobState.DONE, JobState.FAILED}
        ]
        if not finished:
            return
        oldest = min(finished, key=lambda j: j.created_at)
        del self._jobs[oldest.id]
        log.debug("api.job_evicted", job_id=oldest.id)

    def _work(self, job: Job) -> None:
        with self._lock:
            job.state = JobState.RUNNING
            job.started_at = datetime.now(UTC)
        try:
            self._run(self, job)
        except Exception as exc:
            # The orchestrator already has an error boundary, so reaching here
            # means something outside it broke. The job must still reach a
            # terminal state: a client polling a job that silently stopped
            # updating would wait forever.
            log.error(
                "api.job_crashed", job_id=job.id, error=type(exc).__name__
            )
            with self._lock:
                job.state = JobState.FAILED
                job.result = {
                    "status": Status.ERROR.value,
                    "stage": StageName.CONFIG.value,
                    "reason": "job_crashed",
                    "detail": f"The job failed with an unexpected {type(exc).__name__}.",
                }
        finally:
            with self._lock:
                job.finished_at = datetime.now(UTC)
                if job.state is JobState.RUNNING:
                    job.state = JobState.DONE

    # --- called from the worker thread --------------------------------------

    def record_progress(self, job: Job, stage: StageName, message: str) -> None:
        with self._lock:
            job.stage = stage.value
            job.message = message

    def record_result(
        self, job: Job, result: dict[str, Any], diagnostics: dict[str, Any]
    ) -> None:
        with self._lock:
            job.result = result
            job.diagnostics = diagnostics
            job.state = (
                JobState.FAILED
                if result.get("status") == Status.ERROR.value
                else JobState.DONE
            )
