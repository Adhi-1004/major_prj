"""The numbers the paper reports. Each formula is pinned against a hand count."""

from __future__ import annotations

import math

import pytest

from dialogue_extractor.eval import metrics as M


def pairs(spec: list[tuple[bool, str]], clip: str = "c1") -> list[M.Pair]:
    return [M.Pair(clip, truth, status) for truth, status in spec]


SAMPLE = pairs(
    [
        (True, "success"),  # TP
        (True, "success"),  # TP
        (True, "not_on_screen"),  # FN
        (True, "ambiguous"),  # policy-dependent
        (False, "not_on_screen"),  # TN
        (False, "not_on_screen"),  # TN
        (False, "success"),  # FP
        (False, "ambiguous"),  # policy-dependent
        (True, "error"),  # never decided
    ]
)


class TestDecide:
    def test_success_and_not_on_screen_are_fixed(self) -> None:
        for policy in M.POLICIES:
            assert M.decide("success", policy) is True
            assert M.decide("not_on_screen", policy) is False
            assert M.decide("error", policy) is None

    def test_ambiguous_follows_the_policy(self) -> None:
        assert M.decide("ambiguous", "strict") is False
        assert M.decide("ambiguous", "lenient") is True
        assert M.decide("ambiguous", "abstain") is None


class TestScore:
    def test_strict_counts(self) -> None:
        s = M.score(SAMPLE, "strict")
        assert (s.tp, s.fp, s.fn, s.tn) == (2, 1, 2, 3)
        assert s.precision == pytest.approx(2 / 3)
        assert s.recall == pytest.approx(2 / 4)
        assert s.f1 == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))
        assert s.off_precision == pytest.approx(3 / 5)
        assert s.off_recall == pytest.approx(3 / 4)
        assert s.coverage == pytest.approx(8 / 9)

    def test_lenient_counts(self) -> None:
        s = M.score(SAMPLE, "lenient")
        assert (s.tp, s.fp, s.fn, s.tn) == (3, 2, 1, 2)

    def test_abstain_drops_ambiguous_from_counts_and_coverage(self) -> None:
        s = M.score(SAMPLE, "abstain")
        assert (s.tp, s.fp, s.fn, s.tn) == (2, 1, 1, 2)
        assert s.coverage == pytest.approx(6 / 9)

    def test_empty_metrics_are_nan_not_zero(self) -> None:
        """A missing number must look missing, never like a measured 0."""
        s = M.score([], "strict")
        assert math.isnan(s.precision) and math.isnan(s.f1)


class TestBaseline:
    def test_transcript_only_precision_is_the_on_screen_rate(self) -> None:
        s = M.transcript_only_baseline(SAMPLE)
        on_rate = sum(p.truth_on for p in SAMPLE) / len(SAMPLE)
        assert s.precision == pytest.approx(on_rate)
        assert s.recall == pytest.approx(1.0)


class TestKappa:
    def test_perfect_agreement(self) -> None:
        assert M.cohen_kappa([True, False, True], [True, False, True]) == pytest.approx(1.0)

    def test_hand_computed(self) -> None:
        a = [True, True, False, False]
        b = [True, False, False, False]
        # observed 0.75; pa=0.5, pb=0.25 -> expected 0.5*0.25+0.5*0.75=0.5
        assert M.cohen_kappa(a, b) == pytest.approx(0.5)


class TestOnset:
    def test_statistics_and_misses(self) -> None:
        st = M.onset_stats([(2, 25.0), (-4, 25.0), (0, 25.0), (None, 25.0)])
        assert st.n == 4 and st.found == 3
        assert st.mean_abs_frames == pytest.approx(2.0)
        assert st.median_abs_frames == pytest.approx(2.0)
        assert st.mean_signed_frames == pytest.approx(-2 / 3)
        assert st.mean_abs_ms == pytest.approx(80.0)
        # a miss counts against every tolerance
        assert st.within[0] == pytest.approx(1 / 4)
        assert st.within[3] == pytest.approx(2 / 4)
        assert st.within[6] == pytest.approx(3 / 4)


class TestBootstrap:
    def test_single_clip_has_no_interval(self) -> None:
        lo, hi = M.cluster_bootstrap(SAMPLE, lambda s: M.score(s, "strict").f1)
        assert math.isnan(lo) and math.isnan(hi)

    def test_interval_contains_the_point_estimate_and_is_seeded(self) -> None:
        many = [p for i in range(6) for p in pairs([(True, "success"), (False, "success"),
                                                    (True, "not_on_screen")], clip=f"c{i}")]
        stat = lambda s: M.score(s, "strict").f1  # noqa: E731
        first = M.cluster_bootstrap(many, stat, resamples=300)
        second = M.cluster_bootstrap(many, stat, resamples=300)
        assert first == second
        assert first[0] <= stat(many) <= first[1]


class TestReliability:
    def test_bins_report_precision(self) -> None:
        data = [
            M.Pair("c", True, "success", 0.95),
            M.Pair("c", False, "success", 0.92),
            M.Pair("c", True, "ambiguous", 0.5),
            M.Pair("c", True, "not_on_screen", 0.0),  # no confidence bin
        ]
        rows = {(round(lo, 2), n): p for lo, _, n, p in M.reliability(data)}
        assert rows[(0.9, 2)] == pytest.approx(0.5)
        assert rows[(0.45, 1)] == pytest.approx(1.0)
