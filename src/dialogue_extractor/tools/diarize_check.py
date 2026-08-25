"""Manual Phase 3 check: diarize audio, align it to a transcript, print the result.

    python -m dialogue_extractor.tools.diarize_check --audio output/work/audio.wav

With --transcript, the words are tagged with speakers and the speaker-attributed
dialogue is printed, plus the exact JSON payload the Active Speaker Detection
service will receive.

Requires HF_TOKEN in .env, and the gated pipeline's user conditions accepted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dialogue_extractor.audio.aligner import assign_speakers, build_segments, speaker_statistics
from dialogue_extractor.audio.cached_diarizer import CachedDiarizer, pyannote_fingerprint
from dialogue_extractor.audio.cached_transcriber import transcript_from_dict
from dialogue_extractor.audio.diarization_export import (
    looks_like_milliseconds,
    to_nvidia_diarization,
    validate_diarization_payload,
)
from dialogue_extractor.audio.pyannote_diarizer import PyannoteDiarizer, turns_to_dict
from dialogue_extractor.config import constants
from dialogue_extractor.config.settings import Settings
from dialogue_extractor.domain.errors import StageError
from dialogue_extractor.domain.timing import format_timestamp
from dialogue_extractor.infra.disk_cache import DiskCache, NullCache
from dialogue_extractor.infra.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 3 diarization + alignment check")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="Transcript JSON from transcribe_check; enables alignment output.",
    )
    parser.add_argument("--out", type=Path, default=Path("output/diarization.json"))
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=15, help="Turns to print")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    settings = Settings()

    try:
        token = settings.require_hf_token()

        diarizer = CachedDiarizer(
            PyannoteDiarizer(
                token,
                device=args.device,
                num_speakers=args.num_speakers,
            ),
            NullCache() if args.no_cache else DiskCache(Path(".cache")),
            fingerprint=pyannote_fingerprint(
                constants.PYANNOTE_PIPELINE, num_speakers=args.num_speakers
            ),
        )

        print(f"pipeline: {constants.PYANNOTE_PIPELINE}")
        turns = diarizer.diarize(args.audio)

        speakers = sorted({t.speaker_id for t in turns})
        print(f"turns   : {len(turns)}")
        print(f"speakers: {len(speakers)}  ({', '.join(speakers)})")

        print(f"\n--- first {args.max_turns} turns ---")
        for t in turns[: args.max_turns]:
            print(
                f"  {format_timestamp(t.start)} -> {format_timestamp(t.end)}  "
                f"({t.end - t.start:5.2f}s)  {t.speaker_id}"
            )

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(turns_to_dict(turns), indent=2), encoding="utf-8")
        print(f"\nturns written to {args.out}")

        if args.transcript is None:
            print("\n(pass --transcript to see alignment and the ASD payload)")
            return 0

        transcript = transcript_from_dict(
            json.loads(args.transcript.read_text(encoding="utf-8"))
        )
        aligned = assign_speakers(transcript.words, turns)
        segments = build_segments(aligned)

        stats = speaker_statistics(aligned)
        assigned = sum(v for k, v in stats.items() if k)
        unassigned = stats.get("", 0)
        print("\n--- alignment ---")
        print(f"  words          : {len(aligned)}")
        print(f"  assigned       : {assigned}")
        print(f"  unassigned     : {unassigned} (kept, not dropped)")
        for speaker in sorted(k for k in stats if k):
            print(f"    {speaker}: {stats[speaker]} words")
        print(f"  segments       : {len(segments)}")

        print(f"\n--- first {args.max_turns} segments ---")
        for seg in segments[: args.max_turns]:
            speaker = seg.speaker_id or "(unassigned)"
            text = seg.text if len(seg.text) <= 70 else seg.text[:67] + "..."
            print(f"  {format_timestamp(seg.start)}  {speaker:>12}  {text}")

        # The exact payload the ASD service will receive, for a sample window.
        window = [w_ for w_ in aligned if 0.0 <= w_.start <= 30.0]
        if window:
            payload = to_nvidia_diarization(tuple(window), clip_start=window[0].start)
            validate_diarization_payload(payload)
            unit_warning = looks_like_milliseconds(payload, clip_duration=30.0)
            print("\n--- ASD diarization payload (first 30s window) ---")
            print("  validates          : yes")
            print(f"  millisecond warning: {unit_warning}")
            print(f"  words in payload   : {len(payload['words'])}")
            print("  first entry        :", json.dumps(payload["words"][0]))

            payload_path = args.out.with_name("asd_diarization_sample.json")
            payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"  written to         : {payload_path}")

        aligned_path = args.out.with_name("aligned_transcript.json")
        aligned_path.write_text(
            json.dumps(
                [
                    {
                        "text": w_.text,
                        "start": w_.start,
                        "end": w_.end,
                        "probability": w_.probability,
                        "speaker_id": w_.speaker_id,
                    }
                    for w_ in aligned
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\naligned transcript written to {aligned_path}")
        return 0

    except StageError as exc:
        print("\n--- FAILED ---", file=sys.stderr)
        print(json.dumps(exc.to_dict(), indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
