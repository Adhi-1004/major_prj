"""Task metrics. PURE: no I/O.

Line-level verdict metrics treat "on screen" as the positive class. The system's
three-way verdict is mapped to a binary decision under an explicit policy for
`ambiguous`, and every policy is reported, because which one is "right" depends
on how a downstream user treats an uncertain answer:

    strict    ambiguous counts as "not on screen"   (precision-oriented)
    lenient   ambiguous counts as "on screen"       (recall-oriented)
    abstain   ambiguous lines are excluded; coverage reports how many remain

Confidence intervals use a cluster bootstrap over clips. Lines from one clip
are not independent -- same speakers, same audio conditions -- so resampling
lines would understate the uncertainty.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise

POLICIES = ("strict", "lenient", "abstain")


@dataclass(frozen=True, slots=True)
class Pair:
    """One scored line: the human verdict and the system's."""

    clip_id: str
    truth_on: bool
    status: str  # success | ambiguous | not_on_screen | error
    confidence: float | None = None


def decide(status: str, policy: str) -> bool | None:
    """Map a system status to on-screen (True), off (False) or abstain (None)."""
    if status == "success":
        return True
    if status == "not_on_screen":
        return False
    if status == "ambiguous":
        return {"strict": False, "lenient": True, "abstain": None}[policy]
    return None  # error: no verdict


@dataclass(frozen=True, slots=True)
class Scores:
    n: int
    tp: int
    fp: int
    fn: int
    tn: int
    coverage: float

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else float("nan")

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r and not math.isnan(p + r) else float("nan")

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total else float("nan")

    @property
    def off_precision(self) -> float:
        """Of the lines called off camera, how many truly are."""
        return self.tn / (self.tn + self.fn) if self.tn + self.fn else float("nan")

    @property
    def off_recall(self) -> float:
        return self.tn / (self.tn + self.fp) if self.tn + self.fp else float("nan")


def score(pairs: Sequence[Pair], policy: str) -> Scores:
    tp = fp = fn = tn = 0
    decided = 0
    for pair in pairs:
        verdict = decide(pair.status, policy)
        if verdict is None:
            continue
        decided += 1
        if verdict and pair.truth_on:
            tp += 1
        elif verdict and not pair.truth_on:
            fp += 1
        elif not verdict and pair.truth_on:
            fn += 1
        else:
            tn += 1
    coverage = decided / len(pairs) if pairs else float("nan")
    return Scores(len(pairs), tp, fp, fn, tn, coverage)


def transcript_only_baseline(pairs: Sequence[Pair]) -> Scores:
    """Baseline B1: caption/transcript search calls every heard line on screen."""
    return score([Pair(p.clip_id, p.truth_on, "success") for p in pairs], "strict")


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Agreement beyond chance between two annotators' on/off verdicts."""
    if len(a) != len(b) or not a:
        return float("nan")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


@dataclass(frozen=True, slots=True)
class OnsetStats:
    n: int
    found: int
    mean_abs_frames: float
    median_abs_frames: float
    mean_signed_frames: float
    mean_abs_ms: float
    within: dict[int, float]  # tolerance in frames -> fraction of queries


def onset_stats(
    errors: Sequence[tuple[int | None, float]], tolerances: Sequence[int] = (0, 3, 6, 12)
) -> OnsetStats:
    """Onset error statistics.

    Each item is (signed error in frames, fps), or (None, fps) when the system
    produced no frame for an on-screen query -- a miss, which counts against
    every tolerance but is excluded from the error averages.
    """
    found = [(e, fps) for e, fps in errors if e is not None]
    abs_frames = sorted(abs(e) for e, _ in found)
    n = len(errors)

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else float("nan")

    def median(values: list[int]) -> float:
        if not values:
            return float("nan")
        mid = len(values) // 2
        return float(values[mid]) if len(values) % 2 else (values[mid - 1] + values[mid]) / 2

    return OnsetStats(
        n=n,
        found=len(found),
        mean_abs_frames=mean([float(a) for a in abs_frames]),
        median_abs_frames=median(abs_frames),
        mean_signed_frames=mean([float(e) for e, _ in found]),
        mean_abs_ms=mean([abs(e) / fps * 1000.0 for e, fps in found]),
        within={
            t: (sum(1 for a in abs_frames if a <= t) / n if n else float("nan")) for t in tolerances
        },
    )


def cluster_bootstrap(
    pairs: Sequence[Pair],
    statistic: Callable[[Sequence[Pair]], float],
    *,
    resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI for `statistic`, resampling whole clips with replacement."""
    by_clip: dict[str, list[Pair]] = {}
    for pair in pairs:
        by_clip.setdefault(pair.clip_id, []).append(pair)
    clips = sorted(by_clip)
    if len(clips) < 2:
        return (float("nan"), float("nan"))

    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(resamples):
        sample = [p for c in (rng.choice(clips) for _ in clips) for p in by_clip[c]]
        value = statistic(sample)
        if not math.isnan(value):
            values.append(value)
    if not values:
        return (float("nan"), float("nan"))
    values.sort()
    lo = values[int(alpha / 2 * (len(values) - 1))]
    hi = values[int((1 - alpha / 2) * (len(values) - 1))]
    return (lo, hi)


def reliability(
    pairs: Sequence[Pair], edges: Sequence[float] = (0.0, 0.45, 0.6, 0.7, 0.8, 0.9, 1.0001)
) -> list[tuple[float, float, int, float]]:
    """Precision of on-screen calls per confidence bin: (lo, hi, n, precision).

    Only lines the system called on screen or ambiguous carry a confidence.
    Shows whether a higher score actually means a more trustworthy answer.
    """
    rows = []
    for lo, hi in pairwise(edges):
        inside = [
            p
            for p in pairs
            if p.confidence is not None
            and p.status in ("success", "ambiguous")
            and lo <= p.confidence < hi
        ]
        precision = sum(p.truth_on for p in inside) / len(inside) if inside else float("nan")
        rows.append((lo, min(hi, 1.0), len(inside), precision))
    return rows
