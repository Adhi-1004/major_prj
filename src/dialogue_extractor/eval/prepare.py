"""Download and trim the evaluation clips.

    python -m dialogue_extractor.eval.prepare [--only ID ...]

Each clip is acquired with the same sources the pipeline uses (yt-dlp for URLs,
a plain copy for local files), trimmed to the manifest's [start, end] by
re-encoding so the cut is frame-exact, and fingerprinted. The fingerprints go to
eval/runs/prepared.json so a reported result can be tied to the exact bytes it
was computed on. The media itself stays under eval/data/, which is gitignored:
clips are processed for research, never redistributed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from urllib.parse import urlparse

from dialogue_extractor.composition import source_for
from dialogue_extractor.eval.dataset import CLIPS_DIR, DATA_DIR, RUNS_DIR, load_manifest
from dialogue_extractor.infra.ffmpeg_runner import run_ffmpeg
from dialogue_extractor.infra.logging import configure_logging, get_logger
from dialogue_extractor.ingestion.ffprobe_validator import validate_and_probe
from dialogue_extractor.ingestion.normalizer import sha256_of
from dialogue_extractor.ingestion.url_validator import validate_remote_url

log = get_logger(__name__)

_MEDIA_SUFFIXES = (".mp4", ".m4v", ".mov", ".mpeg", ".mpg", ".mkv", ".webm", ".avi", ".ogv")


def is_direct_media_url(source: str) -> bool:
    """True for an http(s) link straight to a media file (not a watch page)."""
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https") and parsed.path.lower().endswith(_MEDIA_SUFFIXES)


def prepare(only: set[str] | None = None, force: bool = False) -> dict[str, dict[str, object]]:
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    record_path = RUNS_DIR / "prepared.json"
    record: dict[str, dict[str, object]] = (
        json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
    )

    for clip in load_manifest():
        if only and clip.id not in only:
            continue
        if clip.local_path.exists() and not force:
            log.info("prepare.exists", clip=clip.id)
            continue

        if is_direct_media_url(clip.source):
            # A direct file link: let ffmpeg seek over HTTP and read only the
            # segment, instead of downloading a feature-length file to cut
            # seven minutes out of it.
            source_path = validate_remote_url(clip.source, resolve_dns=True)
        else:
            raw_dir = DATA_DIR / "raw" / clip.id
            raw_dir.mkdir(parents=True, exist_ok=True)
            source_path = str(source_for(clip.source).acquire(clip.source, raw_dir).path)

        args: list[str] = []
        if clip.start is not None:
            args += ["-ss", f"{clip.start:.3f}"]
        if clip.end is not None:
            args += ["-to", f"{clip.end:.3f}"]
        args += [
            "-i",
            source_path,
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(clip.local_path),
        ]
        if clip.start is None and clip.end is None and not is_direct_media_url(clip.source):
            shutil.copyfile(source_path, clip.local_path)
        else:
            run_ffmpeg(args, timeout=3600)

        probe = validate_and_probe(clip.local_path)
        record[clip.id] = {
            "source": clip.source,
            "start": clip.start,
            "end": clip.end,
            "duration": round(probe.duration, 3),
            "fps": round(probe.fps, 6),
            "resolution": f"{probe.width}x{probe.height}",
            "sha256": sha256_of(clip.local_path),
        }
        log.info("prepare.done", clip=clip.id, duration=probe.duration)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", help="Prepare only these clip ids.")
    parser.add_argument("--force", action="store_true", help="Re-download and re-trim.")
    args = parser.parse_args(argv)
    configure_logging("INFO")
    record = prepare(set(args.only) if args.only else None, force=args.force)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
