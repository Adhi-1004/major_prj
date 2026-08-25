"""Manual Phase 1 check: acquire a video, validate it, normalize it, print the manifest.

    python -m dialogue_extractor.tools.ingest_check --url "https://ok.ru/video/248244667877"
    python -m dialogue_extractor.tools.ingest_check --url "D:/clips/sample.mp4"

Exists so ingestion can be exercised end-to-end against a real URL without the
later stages (which need credentials and a GPU) being built yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dialogue_extractor.domain.errors import StageError
from dialogue_extractor.infra.ffmpeg_runner import resolve_binary
from dialogue_extractor.infra.logging import configure_logging
from dialogue_extractor.ingestion.ffprobe_validator import probe_audio, validate_and_probe
from dialogue_extractor.ingestion.local_file_source import LocalFileSource
from dialogue_extractor.ingestion.normalizer import is_faststart, normalize
from dialogue_extractor.ingestion.url_validator import SourceKind, classify_source
from dialogue_extractor.ingestion.ytdlp_source import YtDlpSource


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 1 ingestion check")
    parser.add_argument("--url", required=True, help="Public video URL or local file path")
    parser.add_argument("--work-dir", default="output/work", type=Path)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--dns",
        choices=("auto", "doh", "system"),
        default="auto",
        help="DNS resolution: auto falls back to DNS-over-HTTPS if interference "
             "is detected, doh always uses it, system never does.",
    )
    parser.add_argument(
        "--skip-normalize",
        action="store_true",
        help="Acquire and probe only; do not re-encode.",
    )
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    try:
        print(f"ffmpeg  : {resolve_binary('ffmpeg')}")
        print(f"ffprobe : {resolve_binary('ffprobe')}")

        kind = classify_source(args.url)
        print(f"source  : {kind.value}")

        source = (
            LocalFileSource()
            if kind is SourceKind.LOCAL
            else YtDlpSource(dns_mode=args.dns)
        )
        raw = source.acquire(args.url, args.work_dir)
        size_mb = raw.path.stat().st_size / 1_048_576
        print(f"acquired: {raw.path}  ({size_mb:.1f} MB)")
        if raw.title:
            print(f"title   : {raw.title}")

        probe = validate_and_probe(raw.path)
        print("\n--- source (as downloaded) ---")
        print(f"  duration      : {probe.duration:.3f}s")
        print(f"  resolution    : {probe.width}x{probe.height}")
        print(f"  video codec   : {probe.video_codec}")
        print(f"  audio codec   : {probe.audio_codec} "
              f"({probe.audio_sample_rate} Hz, {probe.audio_channels}ch)")
        print(f"  r_frame_rate  : {probe.r_frame_rate:.6f}")
        print(f"  avg_frame_rate: {probe.avg_frame_rate:.6f}")
        print(f"  variable fps  : {probe.is_vfr}")

        if args.skip_normalize:
            return 0

        manifest = normalize(raw, args.work_dir)
        print("\n--- normalized ---")
        print(f"  video       : {manifest.video_path}")
        print(f"  audio       : {manifest.audio_path}")
        print(f"  fps         : {manifest.fps:.6f}  (CFR forced: {manifest.cfr_forced})")
        print(f"  frames      : {manifest.frame_count}")
        print(f"  duration    : {manifest.duration:.3f}s")
        print(f"  resolution  : {manifest.resolution}")
        print(f"  source VFR  : {manifest.source_was_vfr}")
        print(f"  faststart   : {is_faststart(manifest.video_path)}")

        audio_probe = probe_audio(manifest.audio_path)
        print(f"  audio out   : {audio_probe.sample_rate} Hz, "
              f"{audio_probe.channels}ch, {audio_probe.codec}")

        # The relationship every later stage depends on.
        implied = manifest.frame_count / manifest.fps
        print(f"\n  frame_count / fps = {implied:.3f}s  vs  duration {manifest.duration:.3f}s")
        print(f"  invariant 2 holds within {abs(implied - manifest.duration):.3f}s")

        out = args.work_dir / "manifest.json"
        out.write_text(
            json.dumps(
                {
                    "fps": manifest.fps,
                    "frame_count": manifest.frame_count,
                    "duration": manifest.duration,
                    "resolution": manifest.resolution,
                    "cfr_forced": manifest.cfr_forced,
                    "source_was_vfr": manifest.source_was_vfr,
                    "video_sha256": manifest.video_sha256,
                    "audio_sha256": manifest.audio_sha256,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nmanifest written to {out}")
        return 0

    except StageError as exc:
        print("\n--- FAILED ---", file=sys.stderr)
        print(json.dumps(exc.to_dict(), indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
