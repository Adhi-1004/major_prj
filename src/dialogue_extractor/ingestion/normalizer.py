"""Normalize arbitrary input to the one shape the rest of the pipeline assumes.

Output contract, relied on by every downstream stage:

  * video -- H.264 in MP4, yuv420p, **constant frame rate**, at most 1080p tall,
    with the moov atom at the front (`+faststart`)
  * audio -- PCM signed 16-bit, 16 kHz, mono WAV

Three of those are load-bearing rather than cosmetic:

  * **Forced CFR** is what makes `timestamp == frame_number / fps` true. Merely
    measuring the source's frame rate is not enough -- on VFR input that
    relationship does not hold, so a reported frame number could not be
    verified against the reported timestamp.

  * **+faststart** is required by the active-speaker-detection service, whose
    documentation states non-streamable MP4 files "will fail to process
    correctly". It is also what makes the service select its streaming path.

  * **16 kHz mono** is what faster-whisper and pyannote both expect.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from dialogue_extractor.config.constants import (
    MAX_NORMALIZED_HEIGHT,
    TARGET_AUDIO_CHANNELS,
    TARGET_AUDIO_SAMPLE_RATE,
)
from dialogue_extractor.domain.errors import NormalizationError
from dialogue_extractor.domain.models import MediaManifest, RawMedia
from dialogue_extractor.infra.ffmpeg_runner import run_ffmpeg
from dialogue_extractor.infra.logging import get_logger, stage
from dialogue_extractor.ingestion.ffprobe_validator import MediaProbe, validate_and_probe

log = get_logger(__name__)

_HASH_CHUNK = 1024 * 1024


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def choose_target_fps(probe: MediaProbe) -> float:
    """Frame rate to force the CFR output to.

    Uses the measured average rate: on VFR input that reflects the frames
    actually present, so the re-encode neither invents frames nor drops them.
    """
    fps = probe.fps
    if fps <= 0:
        raise NormalizationError(
            "no_frame_rate",
            f"Could not determine a usable frame rate for {probe.path.name}.",
        )
    return fps


def _scale_filter(probe: MediaProbe) -> str | None:
    """Downscale only when taller than the cap, preserving aspect ratio.

    `-2` keeps the width even, which yuv420p requires.
    """
    if probe.height <= MAX_NORMALIZED_HEIGHT:
        return None
    return f"scale=-2:{MAX_NORMALIZED_HEIGHT}"


def normalize(
    raw: RawMedia,
    work_dir: Path,
    *,
    ffmpeg_binary: str | None = None,
    ffprobe_binary: str | None = None,
) -> MediaManifest:
    """Normalize `raw` into the pipeline's standard video + audio pair."""
    work_dir.mkdir(parents=True, exist_ok=True)
    video_out = work_dir / "normalized.mp4"
    audio_out = work_dir / "audio.wav"

    with stage("normalization", source=raw.path.name):
        probe = validate_and_probe(raw.path, ffprobe_binary=ffprobe_binary)
        target_fps = choose_target_fps(probe)

        args: list[str] = ["-i", str(raw.path)]

        # --- video output ---
        video_filters = [f for f in (_scale_filter(probe),) if f]
        args += ["-map", "0:v:0"]
        if video_filters:
            args += ["-vf", ",".join(video_filters)]
        args += [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            # Force CFR. Both flags: -fps_mode governs frame duplication and
            # dropping, -r pins the output rate.
            "-fps_mode", "cfr",
            "-r", f"{target_fps:.6f}",
            "-movflags", "+faststart",
            "-map", "0:a:0?",
            "-c:a", "aac",
            "-b:a", "128k",
            str(video_out),
        ]

        # --- audio output, from the same decode pass ---
        args += [
            "-map", "0:a:0",
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(TARGET_AUDIO_SAMPLE_RATE),
            "-ac", str(TARGET_AUDIO_CHANNELS),
            str(audio_out),
        ]

        run_ffmpeg(args, binary=ffmpeg_binary)

        if not video_out.exists() or video_out.stat().st_size == 0:
            raise NormalizationError(
                "empty_video_output", "Normalization produced no video output."
            )
        if not audio_out.exists() or audio_out.stat().st_size == 0:
            raise NormalizationError(
                "empty_audio_output", "Normalization produced no audio output."
            )

        # Re-probe the *output*, never trusting what we asked ffmpeg for. The
        # manifest's fps is the one the extractor and the ASD offset maths use,
        # so it has to be measured, not assumed.
        out_probe = validate_and_probe(video_out, ffprobe_binary=ffprobe_binary)
        frame_count = out_probe.nb_frames or round(out_probe.duration * out_probe.fps)

        manifest = MediaManifest(
            video_path=video_out,
            audio_path=audio_out,
            fps=out_probe.fps,
            frame_count=frame_count,
            duration=out_probe.duration,
            width=out_probe.width,
            height=out_probe.height,
            cfr_forced=True,
            source_was_vfr=probe.is_vfr,
            video_sha256=sha256_of(video_out),
            audio_sha256=sha256_of(audio_out),
        )

    log.info(
        "normalize.done",
        fps=round(manifest.fps, 4),
        frames=manifest.frame_count,
        duration_s=round(manifest.duration, 3),
        resolution=manifest.resolution,
        source_was_vfr=manifest.source_was_vfr,
    )
    return manifest


def is_faststart(path: Path) -> bool:
    """True if the moov atom precedes mdat, i.e. the file is streamable.

    Read directly from the atom table rather than inferred from the ffmpeg flag,
    because the flag silently does nothing on some inputs and the ASD service
    fails *quietly* on non-streamable input rather than reporting an error.
    """
    with path.open("rb") as handle:
        offset = 0
        while True:
            handle.seek(offset)
            header = handle.read(8)
            if len(header) < 8:
                return False
            size = int.from_bytes(header[:4], "big")
            kind = header[4:8]
            if kind == b"moov":
                return True
            if kind == b"mdat":
                return False
            if size == 1:  # 64-bit extended size follows the header
                ext = handle.read(8)
                if len(ext) < 8:
                    return False
                size = int.from_bytes(ext, "big")
            if size <= 0:
                return False
            offset += size
