"""FastAPI application factory.

A factory rather than a module-level `app` so tests can build an instance with a
stubbed pipeline. Importing a module that has already constructed the real thing
would make the HTTP layer untestable without credentials and a GPU.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.requests import Request
from starlette.staticfiles import StaticFiles

from dialogue_extractor.api.jobs import Job, JobStore, RunFn
from dialogue_extractor.api.routes import build_router
from dialogue_extractor.config.settings import Settings
from dialogue_extractor.domain.enums import Status
from dialogue_extractor.domain.errors import StageError
from dialogue_extractor.infra.logging import configure_logging, get_logger
from dialogue_extractor.pipeline.orchestrator import Orchestrator

log = get_logger(__name__)

API_VERSION = "1.0.0"


def default_runner(orchestrator: Orchestrator) -> RunFn:
    """Wire a job to the real pipeline.

    Progress flows back through the store as the run proceeds rather than being
    returned at the end, so `GET /jobs/{id}` can name the stage a long run is
    currently in instead of only saying "running".
    """

    def run(store: JobStore, job: Job) -> None:
        outcome = orchestrator.run(
            job.request,
            on_progress=lambda stage, message: store.record_progress(
                job, stage, message
            ),
            on_candidate=lambda progress: store.record_candidate(job, progress),
        )
        store.record_result(job, outcome.payload, outcome.diagnostics)

    return run


def deferred_runner(settings: Settings) -> RunFn:
    """Build the pipeline on first use, not at import.

    Constructing it eagerly would resolve credentials at startup, so a server
    with no API key configured could not even answer /health -- the one endpoint
    whose job is to report exactly that.
    """

    def run(store: JobStore, job: Job) -> None:
        from dialogue_extractor.composition import build_orchestrator

        default_runner(build_orchestrator(settings))(store, job)

    return run


def create_app(
    settings: Settings | None = None,
    *,
    orchestrator: Orchestrator | None = None,
) -> FastAPI:
    """Build the application.

    Passing `orchestrator` substitutes the pipeline wholesale; that is how the
    route tests run without a network, a model, or an API key.
    """
    settings = settings or Settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Dialogue Frame Extractor",
        version=API_VERSION,
        description=(
            "Find the exact frame where a line of dialogue is spoken on screen, "
            "distinguishing it from voice-over and off-camera speech."
        ),
    )

    runner = (
        default_runner(orchestrator) if orchestrator is not None
        else deferred_runner(settings)
    )
    store = JobStore(run=runner)

    app.state.store = store
    app.state.settings = settings

    origins = settings.cors_origin_list()
    if origins:
        # Installed only when origins are named. In production the frontend is
        # served by this same process (below), so nothing is cross-origin and
        # this stays absent entirely -- the safest configuration being the one
        # that requires no configuration.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )
        log.info("api.cors_enabled", origins=origins)

    app.include_router(build_router())

    @app.exception_handler(StageError)
    async def _stage_error(_: Request, exc: StageError) -> JSONResponse:
        """Turn a pipeline error raised outside a job into the documented shape.

        Requests can fail before a job exists -- a missing credential during
        provider construction, for one -- and those must not come back as a
        FastAPI traceback when every other failure in the system is structured
        JSON.
        """
        return JSONResponse(status_code=400, content=exc.to_dict())

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        log.error("api.unexpected_error", error=type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={
                "status": Status.ERROR.value,
                "stage": "api",
                "reason": "unexpected_error",
                "detail": f"An unexpected {type(exc).__name__} occurred.",
            },
        )

    _mount_frontend(app, settings)
    return app


def _mount_frontend(app: FastAPI, settings: Settings) -> None:
    """Serve the built frontend at / when one has been built.

    Mounted last, after every API route is registered, because a mount at "/"
    is a catch-all: registered first it would shadow /extract, /jobs, /health
    and /docs. Starlette matches routes in order, so last means the API always
    wins and only genuinely unmatched paths fall through to the static files.

    The existence check is not optional politeness. Someone running the API with
    no frontend built -- every backend test, every CLI-only user, the Docker
    image if the node stage were skipped -- must get a working service, not a
    startup crash about a missing directory.
    """
    dist = Path(settings.frontend_dist)
    if not dist.is_dir():
        log.debug("api.frontend_absent", path=str(dist))
        return

    # html=True serves index.html for the bare path, which is what makes the
    # single-page app resolve at "/" rather than 404.
    app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    log.info("api.frontend_mounted", path=str(dist.resolve()))
