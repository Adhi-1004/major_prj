"""Manual Phase 7 check: discovery mode over the whole video.

    # Bounded run first -- discovery costs one detection request per candidate
    python -m dialogue_extractor.tools.discover_check --max-candidates 5

    # Everything
    python -m dialogue_extractor.tools.discover_check

Verifies every speech segment and returns each one that is genuinely on screen,
with the uncertain ones kept in a separate bucket rather than dropped or
promoted.

Start bounded. On the sample video a full sweep is ~30 requests; a feature-length
film is hundreds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dialogue_extractor.audio.aligner import build_segments
from dialogue_extractor.config import constants
from dialogue_extractor.config.settings import Settings
from dialogue_extractor.domain.errors import StageError
from dialogue_extractor.domain.timing import format_timestamp
from dialogue_extractor.infra.logging import configure_logging
from dialogue_extractor.matching.candidate_builder import candidates_from_segments
from dialogue_extractor.pipeline import mode_b
from dialogue_extractor.pipeline.mode_b import Budget
from dialogue_extractor.tools.extract_check import load_manifest, load_words
from dialogue_extractor.vision.providers.registry import get_provider
from dialogue_extractor.vision.pyav_extractor import PyAVFrameExtractor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 7: discovery mode")
    parser.add_argument("--manifest", type=Path, default=Path("output/work/manifest.json"))
    parser.add_argument(
        "--transcript", type=Path, default=Path("output/aligned_transcript.json")
    )
    parser.add_argument("--max-candidates", type=int, default=0, help="0 = no limit")
    parser.add_argument("--time-budget", type=float, default=0.0, help="seconds; 0 = no limit")
    parser.add_argument("--work-dir", type=Path, default=Path("output/discovery"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/frames"))
    parser.add_argument("--out", type=Path, default=Path("output/discovery.json"))
    parser.add_argument("--provider", default="nvidia")
    parser.add_argument("--min-speaking-ms", type=float, default=constants.MIN_SPEAKING_MS)
    parser.add_argument("--no-scene-cuts", action="store_true")
    parser.add_argument("--no-frames", action="store_true", help="Skip image extraction")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; no requests")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    settings = Settings()

    try:
        manifest = load_manifest(args.manifest)
        words = load_words(args.transcript)
        segments = build_segments(words)
        candidates = candidates_from_segments(segments, manifest.duration)

        planned = (
            candidates[: args.max_candidates] if args.max_candidates > 0 else candidates
        )
        footage = sum(c.duration for c in planned)

        print(f"video      : {manifest.resolution} @ {manifest.fps:.3f} fps, "
              f"{manifest.duration:.1f}s")
        print(f"segments   : {len(segments)}")
        print(f"candidates : {len(candidates)}")
        print(f"planned    : {len(planned)} requests covering {footage:.1f}s "
              f"({footage / manifest.duration:.1%} of the video)")

        below = [c for c in candidates if c.duration < constants.ASD_MIN_CLIP_SECONDS]
        print(f"below the {constants.ASD_MIN_CLIP_SECONDS}s service floor: {len(below)} "
              f"(these would return nothing if not widened)")

        if args.dry_run:
            print("\n--- plan (dry run, no requests sent) ---")
            for index, candidate in enumerate(planned):
                speaker = candidate.speaker_id or "(unassigned)"
                print(f"  {index:>3}. {format_timestamp(candidate.start)} "
                      f"({candidate.duration:5.2f}s)  {speaker:>12}  "
                      f"{candidate.text[:52]}")
            return 0

        provider = get_provider(args.provider, settings)
        print(f"\nprovider   : {provider.name}\n")

        result = mode_b.run(
            candidates,
            manifest,
            words,
            provider,
            PyAVFrameExtractor(),
            work_dir=args.work_dir,
            output_dir=args.output_dir,
            budget=Budget(
                max_candidates=args.max_candidates,
                time_budget_seconds=args.time_budget,
            ),
            min_speaking_ms=args.min_speaking_ms,
            detect_scene_cuts=not args.no_scene_cuts,
            extract_frames=not args.no_frames,
        )

        print(f"\n=== {len(result.results)} on screen, "
              f"{len(result.ambiguous)} uncertain, "
              f"{len(result.not_on_screen)} off camera ===")
        print(f"processed {result.candidates_processed} of {result.candidates_total}"
              f"{'  (TRUNCATED)' if result.truncated else ''}")

        if result.results:
            print("\n--- verified on screen ---")
            for entry in result.results:
                print(f"  {format_timestamp(entry.timestamp or 0)}  "
                      f"frame {entry.frame_number:<7} conf {entry.confidence:.3f}  "
                      f"{(entry.speaker_id or '?'):>12}  {(entry.text or '')[:46]}")

        if result.ambiguous:
            print("\n--- uncertain (reported, not discarded) ---")
            for entry in result.ambiguous:
                reason = entry.ambiguity_reason.value if entry.ambiguity_reason else "?"
                print(f"  {format_timestamp(entry.timestamp or 0)}  "
                      f"frame {entry.frame_number:<7} conf {entry.confidence:.3f}  "
                      f"{reason:<24} {(entry.text or '')[:36]}")

        if result.errors:
            print(f"\n--- {len(result.errors)} candidate(s) failed ---")
            for error in result.errors:
                print(f"  candidate {error['candidate']}: {error['reason']}")

        if result.not_on_screen:
            print(f"\n--- {len(result.not_on_screen)} ruled off camera ---")
            for entry in result.not_on_screen:
                reason = entry.ambiguity_reason.value if entry.ambiguity_reason else "?"
                print(f"  {format_timestamp(entry.timestamp or 0)}  "
                      f"{(entry.speaker_id or '(unassigned)'):>12}  "
                      f"{reason:<22} {(entry.text or '')[:38]}")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        print(f"\nwritten to {args.out}")
        if not args.no_frames:
            print(f"frames in  {args.output_dir}")

        return 0

    except StageError as exc:
        print("\n--- FAILED ---", file=sys.stderr)
        print(json.dumps(exc.to_dict(), indent=2), file=sys.stderr)
        return 1
    except NotImplementedError as exc:
        print(f"\n--- NOT IMPLEMENTED ---\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
