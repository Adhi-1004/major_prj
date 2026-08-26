"""HTTP endpoints.

Three of them, which is what the design document's right-sized-scope goal asks
for: submit work, poll it, check the service is alive. The frame images are
served too, because a result naming an image the caller cannot fetch is only
half an answer over HTTP.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from dialogue_extractor.api.jobs import JobState, JobStore
from dialogue_extractor.api.schemas import (
    ExtractRequest,
    HealthResponse,
    JobAccepted,
)
from dialogue_extractor.config.settings import Settings
from dialogue_extractor.domain.enums import Status
from dialogue_extractor.infra.logging import get_logger
from dialogue_extractor.vision.providers.registry import available_providers

log = get_logger(__name__)


def _store(request: Request) -> JobStore:
    store: JobStore = request.app.state.store
    return store


def _settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def resolve_frame_path(root: Path, filename: str) -> Path | None:
    """Resolve a frame name under `root/frames`, or None if it is not servable.

    The one implementation of the traversal rule, so the route and its tests
    cannot check different things. `is_relative_to` runs on the *resolved* path
    -- comparing before resolution would accept `frames/../../.env`, which is
    the whole attack this exists to stop.

    Exactly one directory is searched. It used to be two -- `root/frames` then
    `root` -- because the two modes wrote to different places while naming files
    only by frame number. That made the answer depend on lookup order: with a
    `frame_144.png` in both, a target-mode result was served discovery's frame,
    possibly from a different video entirely. Both modes now write here, so
    there is nothing left to disambiguate.
    """
    root = root.resolve()
    if Path(filename).suffix.lower() != ".png":
        return None
    resolved = (root / "frames" / filename).resolve()
    if resolved.is_relative_to(root) and resolved.is_file():
        return resolved
    return None


def build_router() -> APIRouter:
    router = APIRouter()

    @router.post(
        "/extract",
        response_model=JobAccepted,
        status_code=202,
        summary="Submit an extraction",
        tags=["extraction"],
    )
    def extract(payload: ExtractRequest, request: Request) -> JobAccepted:
        """Accept the work and return immediately.

        202 rather than 200: a full run takes minutes, and the response says the
        request was accepted, not that it is finished. Answering 200 with a
        result the caller must then poll for would misdescribe both.
        """
        job = _store(request).submit(payload.to_pipeline_request())
        return JobAccepted(
            job_id=job.id,
            state=job.state.value,
            mode=job.request.mode.value,
            poll=f"/jobs/{job.id}",
        )

    @router.get(
        "/jobs/{job_id}",
        summary="Poll a job",
        tags=["extraction"],
    )
    def get_job(job_id: str, request: Request) -> JSONResponse:
        """Report a job's state, and its result once there is one.

        A finished job returns 200 whatever the pipeline concluded. `not_found`
        and `not_on_screen` are answers, not HTTP failures -- returning 404 for
        "this line is spoken off camera" would conflate the answer with the job
        itself being missing.
        """
        job = _store(request).get(job_id)
        if job is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "status": Status.ERROR.value,
                    "stage": "api",
                    "reason": "unknown_job",
                    "detail": f"No job with id {job_id!r}.",
                    "remediation": (
                        "Job ids are not persisted across restarts. Submit the "
                        "work again with POST /extract."
                    ),
                },
            )
        return JSONResponse(status_code=200, content=job.to_dict())

    @router.get("/jobs", summary="List jobs", tags=["extraction"])
    def list_jobs(request: Request) -> JSONResponse:
        jobs = _store(request).list()
        return JSONResponse(
            status_code=200,
            content={
                "jobs": [
                    {
                        k: v
                        for k, v in job.to_dict().items()
                        if k not in {"result", "diagnostics"}
                    }
                    for job in jobs
                ],
                "count": len(jobs),
            },
        )

    @router.get(
        "/frames/{filename}",
        summary="Fetch an extracted frame",
        tags=["extraction"],
        response_class=FileResponse,
    )
    def get_frame(filename: str, request: Request) -> FileResponse:
        """Serve one extracted PNG.

        The filename is resolved and then checked to be inside the output
        directory. Without that, `../../.env` in the path would read whatever
        the server process can reach -- and this endpoint exists precisely to
        hand out files named by a value that arrived over the network.
        """
        settings = _settings(request)
        resolved = resolve_frame_path(settings.output_dir, filename)
        if resolved is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "status": Status.ERROR.value,
                    "stage": "api",
                    "reason": "unknown_frame",
                    "detail": f"No extracted frame named {filename!r}.",
                },
            )
        return FileResponse(resolved, media_type="image/png")

    @router.get(
        "/health",
        response_model=HealthResponse,
        summary="Liveness and configuration",
        tags=["ops"],
    )
    def health(request: Request) -> HealthResponse:
        """Report liveness plus the facts that usually explain a failed run.

        Whether each credential is *present* is reported; the values never are.
        A missing key is the most common setup mistake, and learning it here
        beats learning it after a five-minute transcription.
        """
        from dialogue_extractor.api.app import API_VERSION

        settings = _settings(request)
        jobs = _store(request).list()

        ffmpeg_version: str | None = None
        try:
            from dialogue_extractor.infra.ffmpeg_runner import resolve_binary

            ffmpeg_version = str(resolve_binary("ffmpeg"))
        except Exception:
            # /health must answer even when ffmpeg is missing -- that is one of
            # the things it is being asked.
            ffmpeg_version = None

        return HealthResponse(
            status="ok",
            version=API_VERSION,
            ffmpeg=ffmpeg_version,
            providers=list(available_providers()),
            credentials={
                "nvidia_api_key": settings.nvidia_api_key is not None,
                "asd_function_id": bool(settings.asd_function_id),
                "hf_token": settings.hf_token is not None,
            },
            jobs={
                "total": len(jobs),
                "running": sum(
                    1 for j in jobs if j.state in {JobState.RUNNING, JobState.QUEUED}
                ),
            },
        )

    return router
