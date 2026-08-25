"""Retry policy.

Retrying is only correct for transient failures. A permanent one retried three
times just produces the same answer more slowly, so `give_up_on` must win over
`retry_on` even when a type matches both.
"""

from __future__ import annotations

import pytest

from dialogue_extractor.infra.retry import retry


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dialogue_extractor.infra.retry.time.sleep", lambda _: None)


class Transient(Exception):
    pass


class Permanent(Transient):
    """Subclasses Transient on purpose -- give_up_on must still short-circuit."""


def test_returns_immediately_on_success() -> None:
    calls = {"n": 0}

    @retry(attempts=3, retry_on=(Transient,))
    def ok() -> str:
        calls["n"] += 1
        return "done"

    assert ok() == "done"
    assert calls["n"] == 1


def test_retries_then_succeeds() -> None:
    calls = {"n": 0}

    @retry(attempts=3, base_delay=0.0, retry_on=(Transient,))
    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise Transient("blip")
        return "done"

    assert flaky() == "done"
    assert calls["n"] == 3


def test_exhausts_attempts_then_reraises() -> None:
    calls = {"n": 0}

    @retry(attempts=3, base_delay=0.0, retry_on=(Transient,))
    def always_fails() -> None:
        calls["n"] += 1
        raise Transient("blip")

    with pytest.raises(Transient):
        always_fails()
    assert calls["n"] == 3


def test_give_up_on_wins_over_retry_on() -> None:
    """Even though Permanent is a Transient, it must not be retried."""
    calls = {"n": 0}

    @retry(attempts=3, base_delay=0.0, retry_on=(Transient,), give_up_on=(Permanent,))
    def permanent() -> None:
        calls["n"] += 1
        raise Permanent("geo-blocked")

    with pytest.raises(Permanent):
        permanent()
    assert calls["n"] == 1


def test_unlisted_exceptions_propagate_immediately() -> None:
    calls = {"n": 0}

    @retry(attempts=3, base_delay=0.0, retry_on=(Transient,))
    def other() -> None:
        calls["n"] += 1
        raise ValueError("unrelated")

    with pytest.raises(ValueError):
        other()
    assert calls["n"] == 1


def test_backoff_grows_and_is_capped() -> None:
    delays: list[float] = []

    @retry(attempts=5, base_delay=1.0, max_delay=4.0, retry_on=(Transient,), jitter=False)
    def failing() -> None:
        raise Transient("blip")

    import dialogue_extractor.infra.retry as module

    original = module.time.sleep
    module.time.sleep = delays.append  # type: ignore[assignment]
    try:
        with pytest.raises(Transient):
            failing()
    finally:
        module.time.sleep = original  # type: ignore[assignment]

    assert delays == [1.0, 2.0, 4.0, 4.0], "should double, then clamp at max_delay"


def test_preserves_function_metadata() -> None:
    @retry(attempts=1)
    def documented() -> None:
        """A docstring."""

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "A docstring."


def test_passes_arguments_through() -> None:
    @retry(attempts=2, base_delay=0.0, retry_on=(Transient,))
    def add(a: int, *, b: int) -> int:
        return a + b

    assert add(1, b=2) == 3
