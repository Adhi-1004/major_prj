"""Discovery budgets.

The only thing a budget must never do is quietly change the meaning of the
answer. A run that examined 5 of 37 candidates has not found "every on-screen
line" -- it has found every one in the first 5, and the output has to say so.
"""

from __future__ import annotations

import time

import pytest

from dialogue_extractor.pipeline.mode_b import Budget, BudgetTracker


class TestUnlimitedByDefault:
    def test_no_limits_set(self) -> None:
        assert Budget().unlimited is True

    def test_a_candidate_cap_makes_it_limited(self) -> None:
        assert Budget(max_candidates=5).unlimited is False

    def test_a_time_cap_makes_it_limited(self) -> None:
        assert Budget(time_budget_seconds=30).unlimited is False

    def test_zero_means_unlimited_not_none(self) -> None:
        """0 is the documented "no limit" value, not "stop immediately"."""
        tracker = BudgetTracker(Budget(max_candidates=0), total=10)
        assert tracker.may_continue() is True

    def test_negative_is_also_treated_as_unlimited(self) -> None:
        tracker = BudgetTracker(Budget(max_candidates=-1), total=10)
        assert tracker.may_continue() is True


class TestCandidateCap:
    def test_allows_exactly_the_budget(self) -> None:
        tracker = BudgetTracker(Budget(max_candidates=3), total=10)
        for _ in range(3):
            assert tracker.may_continue() is True
            tracker.record()
        assert tracker.may_continue() is False

    def test_records_why_it_stopped(self) -> None:
        tracker = BudgetTracker(Budget(max_candidates=2), total=10)
        for _ in range(2):
            tracker.may_continue()
            tracker.record()
        tracker.may_continue()
        assert tracker.stopped_reason is not None
        assert "candidate budget" in tracker.stopped_reason

    def test_a_budget_larger_than_the_work_never_trips(self) -> None:
        tracker = BudgetTracker(Budget(max_candidates=100), total=3)
        for _ in range(3):
            assert tracker.may_continue() is True
            tracker.record()
        assert tracker.stopped_reason is None


class TestTruncationIsHonest:
    def test_stopping_early_is_reported_as_truncated(self) -> None:
        tracker = BudgetTracker(Budget(max_candidates=2), total=10)
        for _ in range(2):
            tracker.record()
        assert tracker.truncated is True

    def test_a_complete_sweep_is_not_truncated(self) -> None:
        tracker = BudgetTracker(Budget(), total=3)
        for _ in range(3):
            tracker.record()
        assert tracker.truncated is False

    def test_an_empty_run_is_not_truncated(self) -> None:
        assert BudgetTracker(Budget(), total=0).truncated is False


class TestTimeCap:
    def test_stops_once_the_clock_runs_out(self) -> None:
        """The clock is stubbed rather than slept against: a real sleep makes
        the test slow *and* flaky on a loaded machine."""
        clock = [1000.0]
        tracker = BudgetTracker(Budget(time_budget_seconds=30.0), total=100)
        tracker._started = clock[0]  # type: ignore[attr-defined]

        import dialogue_extractor.pipeline.mode_b as module

        real = module.time.monotonic
        module.time.monotonic = lambda: clock[0]  # type: ignore[assignment]
        try:
            assert tracker.may_continue() is True
            clock[0] += 31.0
            assert tracker.may_continue() is False
        finally:
            module.time.monotonic = real  # type: ignore[assignment]

        assert tracker.stopped_reason is not None
        assert "time budget" in tracker.stopped_reason

    def test_elapsed_advances(self) -> None:
        tracker = BudgetTracker(Budget(), total=1)
        time.sleep(0.02)
        assert tracker.elapsed > 0.0


class TestBothCapsTogether:
    def test_whichever_binds_first_wins(self) -> None:
        tracker = BudgetTracker(
            Budget(max_candidates=2, time_budget_seconds=3600), total=10
        )
        for _ in range(2):
            tracker.may_continue()
            tracker.record()
        assert tracker.may_continue() is False
        assert tracker.stopped_reason is not None
        assert "candidate budget" in tracker.stopped_reason

    @pytest.mark.parametrize("processed", [0, 1, 5])
    def test_processed_count_is_accurate(self, processed: int) -> None:
        tracker = BudgetTracker(Budget(), total=10)
        for _ in range(processed):
            tracker.record()
        assert tracker.processed == processed
