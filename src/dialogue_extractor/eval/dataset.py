"""The evaluation manifest, the configurations under test, and file layout."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dialogue_extractor.config import constants

EVAL_ROOT = Path("eval")
DATA_DIR = EVAL_ROOT / "data"  # media: gitignored, never redistributed
CLIPS_DIR = DATA_DIR / "clips"
LABEL_VIDEO_DIR = DATA_DIR / "label_videos"
RUNS_DIR = EVAL_ROOT / "runs"  # system output: small JSON, kept
LABELLING_DIR = EVAL_ROOT / "labelling"  # blind files for annotators
LABELS_DIR = EVAL_ROOT / "labels"  # human ground truth
REPORT_DIR = EVAL_ROOT / "report"

GENRES = ("film", "tv", "interview", "lecture", "news", "documentary")


@dataclass(frozen=True, slots=True)
class ClipSpec:
    """One evaluation clip, exactly as listed in the manifest."""

    id: str
    title: str
    genre: str
    source: str
    license: str
    license_url: str = ""
    start: float | None = None
    end: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.id or not self.id.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"clip id {self.id!r} must be alphanumeric, '-' or '_'")
        if self.genre not in GENRES:
            raise ValueError(f"clip {self.id}: genre {self.genre!r} not in {GENRES}")
        if self.start is not None and self.end is not None and self.end <= self.start:
            raise ValueError(f"clip {self.id}: end must be after start")

    @property
    def local_path(self) -> Path:
        return CLIPS_DIR / f"{self.id}.mp4"


def load_manifest(path: Path = EVAL_ROOT / "manifest.json") -> tuple[ClipSpec, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    clips = tuple(ClipSpec(**item) for item in data["clips"])
    ids = [c.id for c in clips]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate clip ids in manifest")
    return clips


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """A named pipeline configuration. `full` is the proposed system."""

    name: str
    description: str
    overrides: dict[str, Any] = field(default_factory=dict)


CONFIGS: tuple[EvalConfig, ...] = (
    EvalConfig("full", "Proposed system, all rules on"),
    EvalConfig(
        "A1_no_speaker_check",
        "Accept any speaking face (speaker-agnostic ASD baseline)",
        {"speaker_check": False},
    ),
    EvalConfig(
        "A2_no_persistence",
        "No minimum speaking duration (k = 1 frame)",
        {"min_speaking_ms": 1.0},
    ),
    EvalConfig("A3_no_scene_cuts", "Runs may cross shot cuts", {"detect_scene_cuts": False}),
    EvalConfig(
        "A4_no_min_clip",
        "Short windows are not padded to the detector's minimum clip length",
        {"min_clip_seconds": 0.0},
    ),
    EvalConfig("A5_no_onset", "Run may start before the line's onset", {"onset_constraint": False}),
)

CONFIG_BY_NAME = {c.name: c for c in CONFIGS}

DEFAULTS = {
    "speaker_check": True,
    "min_speaking_ms": constants.MIN_SPEAKING_MS,
    "detect_scene_cuts": True,
    "min_clip_seconds": constants.ASD_MIN_CLIP_SECONDS,
    "onset_constraint": True,
}
