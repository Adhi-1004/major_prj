"""Manual Phase 2 check: transcribe an audio file and print word-level output.

    python -m dialogue_extractor.tools.transcribe_check --audio output/work/audio.wav

Add --find "some phrase" to see whether a phrase is present and where -- a plain
substring scan over the normalized transcript, not the real fuzzy matcher (that
arrives in Phase 4). Enough to confirm the target line was heard at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dialogue_extractor.audio.cached_transcriber import (
    CachedTranscriber,
    transcript_to_dict,
    whisper_fingerprint,
)
from dialogue_extractor.audio.whisper_transcriber import WhisperTranscriber
from dialogue_extractor.config import constants
from dialogue_extractor.domain.errors import StageError
from dialogue_extractor.domain.timing import format_timestamp
from dialogue_extractor.infra.disk_cache import DiskCache, NullCache
from dialogue_extractor.infra.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 transcription check")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("output/transcript.json"))
    parser.add_argument("--model", default=constants.DEFAULT_WHISPER_MODEL)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--language", default="en", help='"auto" to detect')
    parser.add_argument("--find", default=None, help="Phrase to look for in the transcript")
    parser.add_argument("--preview", type=int, default=25, help="Words to print")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    language = None if args.language == "auto" else args.language

    try:
        transcriber = WhisperTranscriber(args.model, device=args.device, language=language)
        print(f"model   : {args.model}")
        print(f"device  : {transcriber.device} ({transcriber.compute_type})")

        cache = NullCache() if args.no_cache else DiskCache(Path(".cache"))
        pipeline = CachedTranscriber(
            transcriber,
            cache,
            fingerprint=whisper_fingerprint(
                args.model, language, constants.WHISPER_BEAM_SIZE, transcriber.compute_type
            ),
        )

        transcript = pipeline.transcribe(args.audio)

        print(f"language: {transcript.language}")
        print(f"duration: {transcript.duration:.2f}s")
        print(f"words   : {len(transcript.words)}")
        mean_prob = (
            sum(w.probability for w in transcript.words) / len(transcript.words)
            if transcript.words
            else 0.0
        )
        print(f"mean word probability: {mean_prob:.3f}")

        print(f"\n--- first {args.preview} words ---")
        for w in transcript.words[: args.preview]:
            print(f"  {format_timestamp(w.start)}  {w.probability:.2f}  {w.text}")

        if args.find:
            print(f"\n--- searching for {args.find!r} ---")
            hits = find_phrase(transcript, args.find)
            if not hits:
                print("  not found (a plain substring scan; Phase 4 adds fuzzy matching)")
            for start, end, snippet in hits:
                print(f"  {format_timestamp(start)} -> {format_timestamp(end)}  {snippet}")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(transcript_to_dict(transcript), indent=2), encoding="utf-8"
        )
        print(f"\ntranscript written to {args.out}")
        return 0

    except StageError as exc:
        print("\n--- FAILED ---", file=sys.stderr)
        print(json.dumps(exc.to_dict(), indent=2), file=sys.stderr)
        return 1


def find_phrase(transcript, phrase: str) -> list[tuple[float, float, str]]:  # type: ignore[no-untyped-def]
    """Crude substring scan over the word stream, for eyeballing only."""
    import re

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

    target = norm(phrase)
    if not target:
        return []
    target_len = len(target.split())

    words = transcript.words
    hits: list[tuple[float, float, str]] = []
    for i in range(len(words) - target_len + 1):
        window = words[i : i + target_len]
        if norm(" ".join(w.text for w in window)) == target:
            context = words[max(0, i - 3) : i + target_len + 3]
            hits.append(
                (window[0].start, window[-1].end, " ".join(w.text for w in context))
            )
    return hits


if __name__ == "__main__":
    raise SystemExit(main())
