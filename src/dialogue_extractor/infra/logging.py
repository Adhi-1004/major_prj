"""Structured logging with per-stage timing.

Console-friendly by default, JSON when LOG_FORMAT=json, so the same run is
readable during development and machine-parseable in a container.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from typing import Any

import structlog

_configured = False

_REDACT_KEYS = {"api_key", "nvidia_api_key", "hf_token", "token", "authorization", "password"}


def _redact(
    _logger: Any, _name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Last line of defence against a credential reaching a log sink."""
    for key in list(event_dict):
        if key.lower() in _REDACT_KEYS:
            event_dict[key] = "***redacted***"
    return event_dict


def configure_logging(level: str = "INFO", json_output: bool | None = None) -> None:
    global _configured
    if json_output is None:
        json_output = os.getenv("LOG_FORMAT", "").lower() == "json"

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=not json_output)
    )

    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _redact,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> Any:
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)


@contextmanager
def stage(name: str, **context: Any) -> Iterator[dict[str, Any]]:
    """Time one pipeline stage and log its start, end, and any failure.

    Yields a mutable dict; anything put in it is included in the completion log,
    so a stage can report what it produced without a second log call.
    """
    log = get_logger("pipeline")
    payload: dict[str, Any] = {}
    log.info("stage.start", stage=name, **context)
    started = time.perf_counter()
    try:
        yield payload
    except Exception as exc:
        log.error(
            "stage.failed",
            stage=name,
            elapsed_s=round(time.perf_counter() - started, 3),
            error=type(exc).__name__,
            detail=str(exc),
            **context,
        )
        raise
    else:
        log.info(
            "stage.done",
            stage=name,
            elapsed_s=round(time.perf_counter() - started, 3),
            **context,
            **payload,
        )
