"""Manual Phase 6 check: the core deliverable, end to end.

    python -m dialogue_extractor.tools.extract_check --dialogue "a line from the video"

Takes the artifacts produced by the earlier stages -- the normalized media, the
speaker-tagged transcript -- matches the phrase, verifies it on screen, and
writes the timestamp, frame number, text, confidence, and the frame image.

This is Mode A: it stops at the first occurrence that verifies, so a
feature-length film normally costs one detection request.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dialogue_extractor.config import constants
from dialogue_extractor.config.settings import Settings
from dialogue_extractor.domain.errors import StageError
from dialogue_extractor.domain.models import MediaManifest, Word
from dialogue_extractor.domain.timing import format_timestamp
from dialogue_extractor.infra.logging import configure_logging
from dialogue_extractor.matching.candidate_builder import candidates_from_matches
from dialogue_extractor.matching.fuzzy_matcher import find_matches
from dialogue_extractor.pipeline import mode_a
from dialogue_extractor.vision.providers.registry import get_provider
from dialogue_extractor.vision.pyav_extractor import PyAVFrameExtractor


def load_manifest(path: Path) -> MediaManifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    work = path.parent
    width, height = (int(x) for x in str(data["resolution"]).split("x"))
    return MediaManifest(
        video_path=work / "normalized.mp4",
        audio_path=work / "audio.wav",
        fps=float(data["fps"]),
        frame_count=int(data["frame_count"]),
        duration=float(data["duration"]),
        width=width,
        height=height,
        cfr_forced=bool(data["cfr_forced"]),
        source_was_vfr=bool(data["source_was_vfr"]),
        video_sha256=str(data["video_sha256"]),
        audio_sha256=str(data["audio_sha256"]),
    )


def load_words(path: Path) -> tuple[Word, ...]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    raw = payload if isinstance(payload, list) else payload["words"]
    return tuple(
        Word(
            text=str(w["text"]),
            start=float(w["start"]),
            end=float(w["end"]),
            probability=float(w.get("probability", 1.0)),
            speaker_id=w.get("speaker_id"),
        )
        for w in raw
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 6: Mode A end to end")
    parser.add_argument("--dialogue", required=True, help="The line to find")
    parser.add_argument("--manifest", type=Path, default=Path("output/work/manifest.json"))
    parser.add_argument(
        "--transcript", type=Path, default=Path("output/aligned_transcript.json")
    )
    parser.add_argument("--work-dir", type=Path, default=Path("output/asd"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--out", type=Path, default=Path("output/result.json"))
    parser.add_argument("--provider", default="nvidia")
    parser.add_argument("--floor", type=float, default=constants.MATCH_SCORE_FLOOR)
    parser.add_argument("--pad", type=float, default=constants.WINDOW_PAD_SECONDS)
    parser.add_argument("--min-speaking-ms", type=float, default=constants.MIN_SPEAKING_MS)
    parser.add_argument("--no-scene-cuts", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    settings = Settings()

    try:
        manifest = load_manifest(args.manifest)
        words = load_words(args.transcript)

        print(f"video     : {manifest.video_path.name}  "
              f"{manifest.resolution} @ {manifest.fps:.3f} fps")
        print(f"transcript: {len(words)} words")
        print(f"target    : {args.dialogue!r}")

        matches = find_matches(words, args.dialogue, score_floor=args.floor)
        print(f"\nmatches   : {len(matches)}")
        for i, match in enumerate(matches, 1):
            print(f"  {i}. {format_timestamp(match.start)}  score={match.score:5.1f}  "
                  f"speaker={match.dominant_speaker() or '(none)'}  {match.text[:52]}")

        candidates = candidates_from_matches(matches, manifest.duration, pad=args.pad)
        if not candidates:
            print("\nno candidates -- the phrase is not in the transcript above the floor.")

        provider = get_provider(args.provider, settings)
        print(f"\nprovider  : {provider.name}")
        print(f"verifying {len(candidates)} candidate(s) in time order, "
              f"stopping at the first on screen...\n")

        result, attempts = mode_a.run(
            candidates,
            manifest,
            words,
            provider,
            PyAVFrameExtractor(),
            target=args.dialogue,
            work_dir=args.work_dir,
            output_dir=args.output_dir,
            min_speaking_ms=args.min_speaking_ms,
            detect_scene_cuts=not args.no_scene_cuts,
        )

        print(f"--- attempts: {len(attempts)} ---")
        for i, attempt in enumerate(attempts, 1):
            window = f"{format_timestamp(attempt.candidate.start)}"
            if attempt.error is not None:
                print(f"  {i}. {window}  ERROR {attempt.error.reason}")
            elif attempt.outcome is not None:
                o = attempt.outcome
                verdict = "VERIFIED" if o.found else (o.reason.value if o.reason else "no")
                print(f"  {i}. {window}  {verdict:<22} "
                      f"speaking={o.frames_with_target_speaking}/{o.frames_examined} "
                      f"longest_run={o.longest_run} (need {o.required_run})")

        print("\n=== RESULT ===")
        print(json.dumps(result.to_dict(), indent=2))

        if result.image:
            print(f"\nframe image: {args.output_dir / result.image}")
            print("Open it and confirm the speaker is on camera mid-line.")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        print(f"result written to {args.out}")

        return 0 if result.status.value in {"success", "ambiguous"} else 1

    except StageError as exc:
        print("\n--- FAILED ---", file=sys.stderr)
        print(json.dumps(exc.to_dict(), indent=2), file=sys.stderr)
        return 1
    except NotImplementedError as exc:
        print(f"\n--- NOT IMPLEMENTED ---\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
