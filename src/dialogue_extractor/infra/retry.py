"""Exponential backoff for the two network-dependent stages.

Deliberately small: retrying is only correct for *transient* failures. A
geo-block, a DRM refusal, or a rejected API key will fail identically on every
attempt, so callers pass the specific exception types worth retrying rather
than blanket-retrying everything and turning a fast clear error into a slow one.
"""

from __future__ import annotations

import functools
import random
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from dialogue_extractor.infra.logging import get_logger

log = get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def retry(
    *,
    attempts: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    give_up_on: tuple[type[BaseException], ...] = (),
    jitter: bool = True,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Retry `attempts` times with exponential backoff.

    Args:
        retry_on: Exception types worth another attempt.
        give_up_on: Types that must fail immediately even if they also match
            `retry_on` -- e.g. a DRM error that subclasses a general download
            error. Checked first.
        jitter: Spread retries so concurrent callers do not synchronize.
    """

    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last: BaseException | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except give_up_on:
                    raise
                except retry_on as exc:
                    last = exc
                    if attempt == attempts:
                        break
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    if jitter:
                        delay *= 0.5 + random.random()
                    log.warning(
                        "retry.attempt_failed",
                        function=fn.__name__,
                        attempt=attempt,
                        of=attempts,
                        retry_in_s=round(delay, 2),
                        error=type(exc).__name__,
                    )
                    time.sleep(delay)
            assert last is not None
            raise last

        return wrapper

    return decorator
