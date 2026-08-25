"""Manual Phase 5 check: send one candidate window to the detection service.

    # Isolating check first -- NVIDIA's own assets, so a failure is ours or theirs
    python -m dialogue_extractor.tools.asd_check --nvidia-sample

    # Then our own pipeline output
    python -m dialogue_extractor.tools.asd_check \
        --manifest output/work/manifest.json \
        --candidates output/candidates.json \
        --transcript output/aligned_transcript.json

Run `--nvidia-sample` before anything else. If it passes and a run on our own
clip fails, the fault is in what we produced; if both fail, it is the client or
the credentials. Without that separation a failure has two causes and no way to
tell them apart.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

from dialogue_extractor.audio.diarization_export import (
    SpeakerCodec,
    to_nvidia_diarization,
    validate_diarization_payload,
)
from dialogue_extractor.config.settings import Settings
from dialogue_extractor.domain.enums import Mode
from dialogue_extractor.domain.errors import StageError
from dialogue_extractor.domain.models import ASDResult, Candidate, MediaManifest, Word
from dialogue_extractor.domain.timing import clip_frame_to_global, format_timestamp
from dialogue_extractor.infra.logging import configure_logging
from dialogue_extractor.vision.clip_builder import build_clip
from dialogue_extractor.vision.providers.registry import get_provider

_REPO_PATH = "NVIDIA-Maxine/nim-clients/master/active-speaker-detection/assets"
RAW_BASE = f"https://raw.githubusercontent.com/{_REPO_PATH}"
LFS_BASE = f"https://media.githubusercontent.com/media/{_REPO_PATH}"

_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def fetch_asset(filename: str, destination: Path) -> Path:
    """Download one sample asset, following Git LFS when needed.

    The two hosts are not interchangeable. Large media lives in LFS, so the raw
    host serves a 132-byte *pointer* rather than the file -- and the service
    then rejects that with "not a valid MP4 atom", which reads like our clip
    builder is at fault rather than the download. The media host resolves LFS
    but 404s for ordinary files like the diarization JSON.

    So: fetch from raw, and only if the result is a pointer, fetch again from
    the media host. That works for both kinds without having to know in advance
    which is which.
    """
    if destination.exists() and destination.stat().st_size > 0:
        head = destination.read_bytes()[: len(_LFS_POINTER_PREFIX)]
        if not head.startswith(_LFS_POINTER_PREFIX):
            return destination

    urllib.request.urlretrieve(f"{RAW_BASE}/{filename}", destination)

    head = destination.read_bytes()[: len(_LFS_POINTER_PREFIX)]
    if head.startswith(_LFS_POINTER_PREFIX):
        urllib.request.urlretrieve(f"{LFS_BASE}/{filename}", destination)

    if destination.stat().st_size == 0:
        raise StageError(
            "asset_download_failed",
            f"{filename} downloaded empty.",
            remediation="Check network access to github.com.",
        )
    return destination


def load_manifest(path: Path) -> MediaManifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    work = path.parent
    return MediaManifest(
        video_path=work / "normalized.mp4",
        audio_path=work / "audio.wav",
        fps=float(data["fps"]),
        frame_count=int(data["frame_count"]),
        duration=float(data["duration"]),
        width=int(data["resolution"].split("x")[0]),
        height=int(data["resolution"].split("x")[1]),
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


def summarize(result: ASDResult, codec: SpeakerCodec | None, expected: str | None) -> None:
    print(f"\n--- {len(result.frames)} frames returned ---")
    if not result.frames:
        print("  the service returned nothing")
        return

    faces = [f for frame in result.frames for f in frame.faces]
    speaking_frames = [f for f in result.frames if f.speaking_faces]
    print(f"  frames with a face   : {sum(1 for f in result.frames if f.faces)}")
    print(f"  frames with speaking : {len(speaking_frames)}")
    print(f"  total face detections: {len(faces)}")

    if faces:
        confidences = [f.face_detection_confidence for f in faces]
        print(f"  face confidence      : {min(confidences):.3f} - {max(confidences):.3f}")

    codes = {f.diarized_speaker_id for f in faces}
    print(f"  speaker codes seen   : {sorted(c for c in codes if c is not None)}")
    if codec is not None:
        labels = sorted(
            {(codec.decode(c) if c is not None else None) or "(none)" for c in codes}
        )
        print(f"  decoded to           : {labels}")
        if expected:
            matching = [
                f for f in faces
                if f.is_speaking
                and f.diarized_speaker_id is not None
                and codec.decode(f.diarized_speaker_id) == expected
            ]
            print(f"  expected speaker     : {expected}")
            print(f"  frames where THAT speaker is speaking: {len(matching)}")

    print("\n  first 12 frames (clip-relative id -> global frame):")
    for frame in result.frames[:12]:
        globally = clip_frame_to_global(frame.frame_id, result.clip_start_frame)
        stamp = format_timestamp(globally / result.fps)
        if not frame.faces:
            print(f"    {frame.frame_id:>4} -> {globally:>6}  {stamp}   (no face)")
            continue
        for face in frame.faces:
            speaking = "SPEAKING" if face.is_speaking else "silent  "
            label = (
                codec.decode(face.diarized_speaker_id)
                if codec is not None and face.diarized_speaker_id is not None
                else None
            )
            who = label or f"code={face.diarized_speaker_id}"
            print(
                f"    {frame.frame_id:>4} -> {globally:>6}  {stamp}   {speaking}  "
                f"conf={face.face_detection_confidence:.2f}  {who}"
            )


def run_nvidia_sample(settings: Settings, provider_name: str, work: Path) -> int:
    """Isolating check against NVIDIA's own assets."""
    work.mkdir(parents=True, exist_ok=True)
    print("--- downloading NVIDIA's sample assets ---")
    paths: dict[str, Path] = {}
    for filename, kind in (
        ("sample_video_streamable.mp4", "video"),
        ("sample_audio.wav", "audio"),
        ("sample_diarization.json", "diarization"),
    ):
        destination = fetch_asset(filename, work / filename)
        print(f"  {filename}  ({destination.stat().st_size / 1024:.0f} KB)")
        paths[kind] = destination

    diarization = json.loads(paths["diarization"].read_text(encoding="utf-8"))
    print(f"  diarization words: {len(diarization.get('words', []))}")

    provider = get_provider(provider_name, settings)
    print(f"\nprovider: {provider.name} -> {settings.asd_target}")
    result = provider.detect(paths["video"], paths["audio"], diarization, 0, 25.0)

    summarize(result, codec=None, expected=None)
    print("\nIf this worked, the client and credentials are fine. Any failure on")
    print("our own clip after this is in what we produced, not in the client.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 5 active-speaker-detection check")
    parser.add_argument("--nvidia-sample", action="store_true",
                        help="Run against NVIDIA's own assets and exit.")
    parser.add_argument("--manifest", type=Path, default=Path("output/work/manifest.json"))
    parser.add_argument("--candidates", type=Path, default=Path("output/candidates.json"))
    parser.add_argument("--transcript", type=Path,
                        default=Path("output/aligned_transcript.json"))
    parser.add_argument("--index", type=int, default=0, help="Which candidate to send")
    parser.add_argument("--work-dir", type=Path, default=Path("output/asd"))
    parser.add_argument("--provider", default="nvidia")
    parser.add_argument("--out", type=Path, default=Path("output/asd_result.json"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    settings = Settings()

    try:
        if args.nvidia_sample:
            return run_nvidia_sample(settings, args.provider, args.work_dir / "nvidia")

        manifest = load_manifest(args.manifest)
        words = load_words(args.transcript)
        payload = json.loads(args.candidates.read_text(encoding="utf-8"))
        entries = payload["candidates"]

        if not entries:
            print("no candidates in the file", file=sys.stderr)
            return 2
        if not 0 <= args.index < len(entries):
            print(f"--index must be 0..{len(entries) - 1}", file=sys.stderr)
            return 2

        entry = entries[args.index]
        candidate = Candidate(
            start=float(entry["start"]),
            end=float(entry["end"]),
            speaker_id=entry.get("speaker_id"),
            text=str(entry.get("text", "")),
            mode=Mode.TARGET if entry.get("match_score") is not None else Mode.DISCOVERY,
            match_score=entry.get("match_score"),
        )

        print(f"candidate {args.index} of {len(entries)}")
        print(f"  window  : {format_timestamp(candidate.start)} -> "
              f"{format_timestamp(candidate.end)}  ({candidate.duration:.2f}s)")
        print(f"  speaker : {candidate.speaker_id or '(unassigned)'}")
        print(f"  text    : {candidate.text[:70]}")

        clip = build_clip(candidate, manifest, args.work_dir, name=f"cand{args.index}")
        print(f"\nclip      : {clip.path.name}  {clip.size_mb:.2f} MB")
        print(f"  fps               : {clip.fps:.6f}")
        print(f"  clip_start_frame  : {clip.clip_start_frame}  (invariant 1 offset)")

        window = tuple(w for w in words if candidate.start <= w.start <= candidate.end)
        codec = SpeakerCodec.from_words(window)
        diarization = to_nvidia_diarization(window, clip_start=candidate.start, codec=codec)
        validate_diarization_payload(diarization)

        print(f"  diarization words : {len(diarization['words'])}")
        print(f"  speaker codes     : {codec.to_code}")

        provider = get_provider(args.provider, settings)
        print(f"\nprovider  : {provider.name} -> {settings.asd_target}")

        result = provider.detect(
            clip.path, clip.audio_path, diarization, clip.clip_start_frame, clip.fps
        )
        summarize(result, codec, candidate.speaker_id)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "candidate_index": args.index,
                    "window": [candidate.start, candidate.end],
                    "expected_speaker": candidate.speaker_id,
                    "clip_start_frame": clip.clip_start_frame,
                    "fps": clip.fps,
                    "frames": [
                        {
                            "frame_id": frame.frame_id,
                            "global_frame": clip_frame_to_global(
                                frame.frame_id, result.clip_start_frame
                            ),
                            "faces": [
                                {
                                    "face_id": face.face_id,
                                    "is_speaking": face.is_speaking,
                                    "face_detection_confidence": face.face_detection_confidence,
                                    "diarized_speaker_id": face.diarized_speaker_id,
                                    "speaker_label": (
                                        codec.decode(face.diarized_speaker_id)
                                        if face.diarized_speaker_id is not None
                                        else None
                                    ),
                                }
                                for face in frame.faces
                            ],
                        }
                        for frame in result.frames
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nresult written to {args.out}")
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
