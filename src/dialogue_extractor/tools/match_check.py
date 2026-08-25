"""Manual Phase 4 check: find a phrase in a transcript and show the candidates.

    python -m dialogue_extractor.tools.match_check \
        --transcript output/aligned_transcript.json \
        --dialogue "a line you heard in the video"

    python -m dialogue_extractor.tools.match_check \
        --transcript output/aligned_transcript.json --mode b

Accepts either the raw transcript from transcribe_check or the speaker-tagged
one from diarize_check. The aligned file is more useful: without it every
candidate's speaker is null, and it is the speaker identity that Phase 5 will
cross-check against the face reported as speaking.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dialogue_extractor.audio.aligner import build_segments
from dialogue_extractor.config import constants
from dialogue_extractor.domain.errors import StageError
from dialogue_extractor.domain.models import Candidate, Word
from dialogue_extractor.domain.timing import format_timestamp
from dialogue_extractor.infra.logging import configure_logging
from dialogue_extractor.matching.candidate_builder import (
    apply_budget,
    candidates_from_matches,
    candidates_from_segments,
)
from dialogue_extractor.matching.fuzzy_matcher import find_matches
from dialogue_extractor.matching.text_normalizer import normalize


def load_words(path: Path) -> tuple[tuple[Word, ...], float]:
    """Read either transcript shape into words plus a duration."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, list):
        # Older aligned_transcript.json: a bare word list with no duration, so
        # the media length has to be inferred from the last word. That is short
        # of the truth and clamps windows near the end of the video too early.
        raw = payload
        duration = max((float(w["end"]) for w in raw), default=0.0)
    else:
        raw = payload["words"]
        duration = float(payload.get("duration") or 0.0)

    words = tuple(
        Word(
            text=str(w["text"]),
            start=float(w["start"]),
            end=float(w["end"]),
            probability=float(w.get("probability", 1.0)),
            speaker_id=w.get("speaker_id"),
        )
        for w in raw
    )
    return words, duration or max((w.end for w in words), default=0.0)


def describe(candidate: Candidate, index: int) -> None:
    speaker = candidate.speaker_id or "(unassigned)"
    score = f"{candidate.match_score:5.1f}" if candidate.match_score is not None else "  n/a"
    text = candidate.text if len(candidate.text) <= 62 else candidate.text[:59] + "..."
    print(
        f"  {index:>3}. {format_timestamp(candidate.start)} -> "
        f"{format_timestamp(candidate.end)}  ({candidate.duration:5.2f}s)  "
        f"score={score}  {speaker:>12}  {text}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4 matching check")
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--dialogue", default=None, help="Target phrase (Mode A)")
    parser.add_argument(
        "--mode",
        choices=("a", "b"),
        default=None,
        help="Defaults to a when --dialogue is given, b otherwise.",
    )
    parser.add_argument("--pad", type=float, default=constants.WINDOW_PAD_SECONDS)
    parser.add_argument("--floor", type=float, default=constants.MATCH_SCORE_FLOOR)
    parser.add_argument("--slack", type=int, default=constants.MATCH_WINDOW_SLACK)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--show", type=int, default=20, help="Candidates to print")
    parser.add_argument("--out", type=Path, default=Path("output/candidates.json"))
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    mode = args.mode or ("a" if args.dialogue else "b")

    try:
        words, duration = load_words(args.transcript)
        tagged = sum(1 for w in words if w.speaker_id)
        print(f"transcript : {args.transcript}")
        print(f"words      : {len(words)}  ({tagged} speaker-tagged)")
        print(f"duration   : {duration:.2f}s")
        if not tagged:
            print("  note: no speaker tags -- run diarize_check --transcript first,")
            print("        or every candidate's speaker will be null.")

        if mode == "a":
            if not args.dialogue:
                print("\n--dialogue is required for mode a", file=sys.stderr)
                return 2

            print(f"\ntarget     : {args.dialogue!r}")
            print(f"normalized : {normalize(args.dialogue)!r}")
            print(f"floor      : {args.floor}   slack: +/-{args.slack} tokens")

            matches = find_matches(
                words, args.dialogue, score_floor=args.floor, slack=args.slack
            )
            print(f"\n--- {len(matches)} match(es) ---")
            for i, match in enumerate(matches, 1):
                context = words[max(0, match.start_index - 4) : match.end_index + 4]
                print(
                    f"  {i:>3}. {format_timestamp(match.start)}  score={match.score:5.1f}  "
                    f"speaker={match.dominant_speaker() or '(none)'}"
                )
                print(f"       matched : {match.text}")
                print(f"       context : ...{' '.join(w.text for w in context)}...")

            if not matches:
                print("  nothing above the floor.")
                print(f"  try a lower --floor (current {args.floor}), or check the")
                print("  transcript actually contains the line.")
                return 1

            candidates = candidates_from_matches(matches, duration, pad=args.pad)
        else:
            segments = build_segments(words)
            print(f"\nsegments   : {len(segments)}")
            candidates = candidates_from_segments(segments, duration)

        candidates, truncated = apply_budget(candidates, max_candidates=args.max_candidates)

        suffix = " (truncated)" if truncated else ""
        print(f"\n--- {len(candidates)} candidate window(s){suffix} ---")
        for i, candidate in enumerate(candidates[: args.show], 1):
            describe(candidate, i)
        if len(candidates) > args.show:
            print(f"  ... {len(candidates) - args.show} more")

        total = sum(c.duration for c in candidates)
        print(f"\n  total footage to verify: {total:.1f}s of {duration:.1f}s "
              f"({total / duration:.1%})")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "mode": mode,
                    "target": args.dialogue,
                    "truncated": truncated,
                    "candidates": [
                        {
                            "start": c.start,
                            "end": c.end,
                            "timestamp": format_timestamp(c.start),
                            "speaker_id": c.speaker_id,
                            "match_score": c.match_score,
                            "text": c.text,
                        }
                        for c in candidates
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\ncandidates written to {args.out}")
        return 0

    except StageError as exc:
        print("\n--- FAILED ---", file=sys.stderr)
        print(json.dumps(exc.to_dict(), indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
