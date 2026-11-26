"""Human ground truth: loading, resolving between annotators, agreement.

Label files are written by eval/labeler/index.html and saved as
``eval/labels/<clip_id>__<annotator>.json``. A file named
``<clip_id>__adjudicated.json`` takes precedence when present.

Label vocabulary, per line:

    on          the speaker of the line is visible and speaking it
    mixed       on camera for part of the line (e.g. a cut mid-sentence)
    off         spoken, but the speaker is not visibly speaking it
    unclear     the annotator could not decide
    not_speech  not a real line (ASR hallucination, song lyrics, crowd noise)

`on` and `mixed` both count as on-screen: the task asks for the earliest frame
at which the line is spoken on camera, and a mixed line has one. `unclear` and
`not_speech` are excluded from scoring and reported as counts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dialogue_extractor.eval.dataset import LABELS_DIR

VALID_LABELS = ("on", "mixed", "off", "unclear", "not_speech")
SCORED_LABELS = ("on", "mixed", "off")


@dataclass(frozen=True, slots=True)
class GroundTruth:
    line_id: str
    label: str
    onset_frame: int | None = None
    query: str | None = None

    @property
    def scored(self) -> bool:
        return self.label in SCORED_LABELS

    @property
    def is_on(self) -> bool:
        return self.label in ("on", "mixed")


def _read(path: Path) -> tuple[str, dict[str, GroundTruth]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    annotator = str(data.get("annotator") or path.stem.split("__", 1)[-1])
    out: dict[str, GroundTruth] = {}
    for item in data["lines"]:
        label = item.get("label")
        if label is None:
            continue  # not yet labelled
        if label not in VALID_LABELS:
            raise ValueError(f"{path.name}: line {item['line_id']} has label {label!r}")
        onset = item.get("onset_frame")
        out[item["line_id"]] = GroundTruth(
            line_id=item["line_id"],
            label=label,
            onset_frame=int(onset) if onset is not None else None,
            query=(item.get("query") or None),
        )
    return annotator, out


def annotator_files(clip_id: str, root: Path = LABELS_DIR) -> dict[str, dict[str, GroundTruth]]:
    """Every annotator's labels for a clip, adjudication excluded."""
    files = sorted(root.glob(f"{clip_id}__*.json"))
    result: dict[str, dict[str, GroundTruth]] = {}
    for path in files:
        if path.stem.endswith("__adjudicated"):
            continue
        annotator, labels = _read(path)
        result[annotator] = labels
    return result


def load_ground_truth(clip_id: str, root: Path = LABELS_DIR) -> dict[str, GroundTruth]:
    """The labels scoring uses.

    Adjudicated file if present; otherwise the only annotator; otherwise the
    lines on which every annotator agrees about on-screen vs off. Disagreements
    are left out rather than resolved by fiat -- `disagreements()` reports them
    so they can be adjudicated.
    """
    adjudicated = root / f"{clip_id}__adjudicated.json"
    if adjudicated.exists():
        return _read(adjudicated)[1]

    per_annotator = annotator_files(clip_id, root)
    if not per_annotator:
        return {}
    if len(per_annotator) == 1:
        return next(iter(per_annotator.values()))

    names = sorted(per_annotator)
    primary = per_annotator[names[0]]
    agreed: dict[str, GroundTruth] = {}
    for line_id, first in primary.items():
        others = [per_annotator[n].get(line_id) for n in names[1:]]
        if any(o is None for o in others):
            continue
        if not first.scored or not all(o is not None and o.scored for o in others):
            continue
        if all(o is not None and o.is_on == first.is_on for o in others):
            onsets = [g.onset_frame for g in (first, *others) if g and g.onset_frame is not None]
            onset = round(sum(onsets) / len(onsets)) if onsets else None
            agreed[line_id] = GroundTruth(line_id, first.label, onset, first.query)
    return agreed


def disagreements(clip_id: str, root: Path = LABELS_DIR) -> list[str]:
    per_annotator = annotator_files(clip_id, root)
    if len(per_annotator) < 2:
        return []
    names = sorted(per_annotator)
    shared = set.intersection(*(set(per_annotator[n]) for n in names))
    out = []
    for line_id in sorted(shared):
        verdicts = {
            per_annotator[n][line_id].is_on for n in names if per_annotator[n][line_id].scored
        }
        if len(verdicts) > 1:
            out.append(line_id)
    return out
