"""Run every configuration over every clip, and export blind labelling files.

    python -m dialogue_extractor.eval.runner                 # discovery, all configs
    python -m dialogue_extractor.eval.runner --configs full  # one configuration
    python -m dialogue_extractor.eval.runner --targets       # target mode on labelled queries

Discovery runs drive the per-line metrics: every line in every clip gets a
verdict under every configuration. Target runs drive the onset metric and the
cost of a query; they need labels, because the queries and their true onset
frames come from the annotators.

Detection responses are cached (see vision/providers/cached_asd.py). Ablations
that only change how detector output is interpreted are therefore evaluated on
identical detector output, and cost no extra requests. The request counts
written here are the ones actually sent to the service.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dialogue_extractor.composition import build_asd, build_cache, build_stages
from dialogue_extractor.config import constants
from dialogue_extractor.config.settings import Settings
from dialogue_extractor.eval.dataset import (
    CONFIG_BY_NAME,
    CONFIGS,
    DATA_DIR,
    DEFAULTS,
    LABEL_VIDEO_DIR,
    LABELLING_DIR,
    RUNS_DIR,
    ClipSpec,
    EvalConfig,
    load_manifest,
)
from dialogue_extractor.eval.labels import load_ground_truth
from dialogue_extractor.infra.logging import configure_logging, get_logger
from dialogue_extractor.ingestion.media_cache import media_fingerprint
from dialogue_extractor.pipeline.mode_b import CandidateProgress
from dialogue_extractor.pipeline.orchestrator import (
    Orchestrator,
    PipelineOutcome,
    PipelineRequest,
)

log = get_logger(__name__)

ONSET_SAMPLES_PER_CLIP = 6
"""Lines per clip for which annotators are asked to mark the onset frame.

Marking an onset is the slow part of labelling, so it is requested for a fixed,
seeded sample rather than for every line. The sample is chosen before anyone
has seen the system's answer.
"""


class _ClipSession:
    """One clip's pipeline, with a shared counting detection provider."""

    def __init__(self, clip: ClipSpec) -> None:
        self.clip = clip
        self.settings = Settings(
            work_dir=DATA_DIR / "work" / clip.id,
            output_dir=DATA_DIR / "output" / clip.id,
        )
        cache = build_cache(self.settings)
        self.provider = build_asd(self.settings, None, cache)
        stages = dataclasses.replace(build_stages(self.settings), asd=lambda: self.provider)
        self.orchestrator = Orchestrator(
            stages,
            work_dir=self.settings.work_dir,
            output_dir=self.settings.output_dir,
            cache=cache,
            media_fingerprint=media_fingerprint(
                max_height=constants.MAX_NORMALIZED_HEIGHT,
                target_audio_rate=constants.TARGET_AUDIO_SAMPLE_RATE,
            ),
        )

    def _counts(self) -> tuple[int, int]:
        return (getattr(self.provider, "misses", 0), getattr(self.provider, "hits", 0))

    def run(
        self, config: EvalConfig, dialogue: str | None = None
    ) -> tuple[PipelineOutcome, dict[str, Any], list[CandidateProgress]]:
        params = {**DEFAULTS, **config.overrides}
        request = PipelineRequest(url=str(self.clip.local_path), dialogue=dialogue, **params)

        marks: list[tuple[str, float]] = []
        messages: list[str] = []
        lines: list[CandidateProgress] = []
        sent_before, hits_before = self._counts()
        started = time.monotonic()

        outcome = self.orchestrator.run(
            request,
            on_progress=lambda stage, msg: (
                marks.append((stage.value, time.monotonic())),
                messages.append(msg),
            ),
            on_candidate=lines.append,
        )
        finished = time.monotonic()
        sent_after, hits_after = self._counts()

        stage_seconds: dict[str, float] = {}
        for i, (stage, at) in enumerate(marks):
            until = marks[i + 1][1] if i + 1 < len(marks) else finished
            stage_seconds[stage] = round(stage_seconds.get(stage, 0.0) + until - at, 3)

        meta = {
            "elapsed_seconds": round(finished - started, 3),
            "stage_seconds": stage_seconds,
            "asd_requests_sent": sent_after - sent_before,
            "asd_cache_hits": hits_after - hits_before,
            # The orchestrator announces "reusing previously normalized media"
            # when acquisition and normalization come from the media cache.
            # Transcript and diarization caches are keyed by the same audio,
            # so a run that reused media is a warm run throughout.
            "cold_run": not any("reusing" in m for m in messages),
        }
        return outcome, meta, lines


def _line_row(progress: CandidateProgress) -> dict[str, Any]:
    result = progress.result.to_dict() if progress.result is not None else None
    return {
        "index": progress.index,
        "line_start": progress.line_start,
        "line_end": progress.line_end,
        "window": [progress.start, progress.end],
        "speaker": progress.speaker_id,
        "text": progress.text,
        "status": result["status"] if result else "error",
        "error_reason": progress.error_reason,
        "frame_number": result.get("frame_number") if result else None,
        "confidence": result.get("confidence") if result else None,
        "reason": result.get("ambiguity_reason") if result else None,
        "breakdown": result.get("confidence_breakdown") if result else None,
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_labelling(clip: ClipSpec, outcome: PipelineOutcome, rows: list[dict[str, Any]]) -> None:
    """Write the blind file an annotator works from: lines, never verdicts."""
    manifest = outcome.manifest
    if manifest is None:
        return
    LABEL_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    video = LABEL_VIDEO_DIR / f"{clip.id}.mp4"
    shutil.copyfile(manifest.video_path, video)

    rng = random.Random(f"onset-sample:{clip.id}")
    ask = set(rng.sample(range(len(rows)), min(ONSET_SAMPLES_PER_CLIP, len(rows))))

    _write(
        LABELLING_DIR / f"{clip.id}.json",
        {
            "clip_id": clip.id,
            "title": clip.title,
            "genre": clip.genre,
            "fps": manifest.fps,
            "frame_count": manifest.frame_count,
            "duration": manifest.duration,
            "video_file": video.name,
            "lines": [
                {
                    "line_id": f"{clip.id}-L{row['index']:03d}",
                    "line_start": row["line_start"],
                    "line_end": row["line_end"],
                    "speaker": row["speaker"],
                    "text": row["text"],
                    "ask_onset": row["index"] in ask,
                }
                for row in rows
            ],
        },
    )


def run_discovery(clips: tuple[ClipSpec, ...], configs: list[EvalConfig], export: bool) -> None:
    for clip in clips:
        if not clip.local_path.exists():
            log.warning("runner.clip_missing", clip=clip.id, hint="run eval.prepare first")
            continue
        session = _ClipSession(clip)
        for config in configs:
            log.info("runner.discovery", clip=clip.id, config=config.name)
            outcome, meta, lines = session.run(config)
            rows = [_line_row(p) for p in lines]
            _write(
                RUNS_DIR / config.name / f"{clip.id}.discovery.json",
                {
                    "clip_id": clip.id,
                    "genre": clip.genre,
                    "config": config.name,
                    "overrides": config.overrides,
                    "run_at": datetime.now(UTC).isoformat(),
                    "status": outcome.status,
                    "error": None if outcome.status != "error" else outcome.payload,
                    "diagnostics": outcome.diagnostics,
                    **meta,
                    "lines": rows,
                },
            )
            if export and config.name == "full" and outcome.status != "error":
                export_labelling(clip, outcome, rows)


def run_targets(clips: tuple[ClipSpec, ...], configs: list[EvalConfig]) -> None:
    for clip in clips:
        truth = load_ground_truth(clip.id)
        queries = [g for g in truth.values() if g.is_on and g.onset_frame is not None and g.query]
        if not queries:
            log.info("runner.no_target_queries", clip=clip.id)
            continue
        session = _ClipSession(clip)
        for config in configs:
            results = []
            for gt in queries:
                outcome, meta, _ = session.run(config, dialogue=gt.query)
                payload = outcome.payload
                results.append(
                    {
                        "line_id": gt.line_id,
                        "query": gt.query,
                        "gt_onset_frame": gt.onset_frame,
                        "status": payload.get("status"),
                        "frame_number": payload.get("frame_number"),
                        "confidence": payload.get("confidence"),
                        "reason": payload.get("ambiguity_reason"),
                        "candidates": outcome.diagnostics.get("candidates"),
                        "candidates_verified": outcome.diagnostics.get("candidates_verified"),
                        "verified_seconds": outcome.diagnostics.get("verified_seconds"),
                        "duration": outcome.diagnostics.get("duration"),
                        "fps": outcome.diagnostics.get("fps"),
                        **meta,
                    }
                )
            _write(
                RUNS_DIR / config.name / f"{clip.id}.targets.json",
                {
                    "clip_id": clip.id,
                    "genre": clip.genre,
                    "config": config.name,
                    "queries": results,
                },
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", nargs="*", default=[c.name for c in CONFIGS])
    parser.add_argument("--only", nargs="*", help="Run only these clip ids.")
    parser.add_argument("--targets", action="store_true", help="Target mode on labelled queries.")
    parser.add_argument(
        "--no-export", action="store_true", help="Do not (re)write labelling files."
    )
    args = parser.parse_args(argv)
    configure_logging("INFO")

    unknown = [c for c in args.configs if c not in CONFIG_BY_NAME]
    if unknown:
        parser.error(f"unknown configs {unknown}; choose from {list(CONFIG_BY_NAME)}")
    # `full` runs first: it fills the detection cache the ablations then reuse.
    configs = sorted((CONFIG_BY_NAME[c] for c in args.configs), key=lambda c: c.name != "full")

    clips = tuple(c for c in load_manifest() if not args.only or c.id in args.only)
    if args.targets:
        run_targets(clips, configs)
    else:
        run_discovery(clips, configs, export=not args.no_export)
    return 0


if __name__ == "__main__":
    sys.exit(main())
