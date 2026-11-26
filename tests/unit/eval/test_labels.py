"""Resolving human labels into ground truth, and never inventing agreement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dialogue_extractor.eval.labels import disagreements, load_ground_truth


def write(root: Path, clip: str, annotator: str, lines: list[dict[str, object]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{clip}__{annotator}.json").write_text(
        json.dumps({"clip_id": clip, "annotator": annotator, "lines": lines}), encoding="utf-8"
    )


def test_single_annotator_is_used_directly(tmp_path: Path) -> None:
    write(tmp_path, "c", "ann1", [{"line_id": "c-L000", "label": "on", "onset_frame": 10}])
    truth = load_ground_truth("c", tmp_path)
    assert truth["c-L000"].is_on and truth["c-L000"].onset_frame == 10


def test_unlabelled_lines_are_skipped(tmp_path: Path) -> None:
    write(tmp_path, "c", "ann1", [{"line_id": "c-L000", "label": None}])
    assert load_ground_truth("c", tmp_path) == {}


def test_unknown_label_is_rejected(tmp_path: Path) -> None:
    write(tmp_path, "c", "ann1", [{"line_id": "c-L000", "label": "maybe"}])
    with pytest.raises(ValueError, match="maybe"):
        load_ground_truth("c", tmp_path)


def test_two_annotators_keep_only_agreement(tmp_path: Path) -> None:
    write(tmp_path, "c", "a", [
        {"line_id": "c-L000", "label": "on", "onset_frame": 10},
        {"line_id": "c-L001", "label": "off"},
        {"line_id": "c-L002", "label": "on"},
    ])
    write(tmp_path, "c", "b", [
        {"line_id": "c-L000", "label": "mixed", "onset_frame": 14},
        {"line_id": "c-L001", "label": "off"},
        {"line_id": "c-L002", "label": "off"},
    ])
    truth = load_ground_truth("c", tmp_path)
    assert set(truth) == {"c-L000", "c-L001"}
    assert truth["c-L000"].onset_frame == 12  # mean of the two marks
    assert disagreements("c", tmp_path) == ["c-L002"]


def test_adjudicated_file_wins(tmp_path: Path) -> None:
    write(tmp_path, "c", "a", [{"line_id": "c-L000", "label": "on"}])
    write(tmp_path, "c", "b", [{"line_id": "c-L000", "label": "off"}])
    write(tmp_path, "c", "adjudicated", [{"line_id": "c-L000", "label": "off"}])
    assert load_ground_truth("c", tmp_path)["c-L000"].is_on is False


def test_unclear_is_not_scored(tmp_path: Path) -> None:
    write(tmp_path, "c", "a", [{"line_id": "c-L000", "label": "unclear"}])
    assert load_ground_truth("c", tmp_path)["c-L000"].scored is False
