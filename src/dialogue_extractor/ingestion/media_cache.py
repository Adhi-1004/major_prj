"""Caching the download and normalization, keyed by source URL.

Transcription and diarization have been cached since Phase 2, but acquisition
never was -- so every run re-downloaded the whole video and re-encoded it, even
when nothing about the request had changed but the phrase being searched for.

That is expensive, and on YouTube it is worse than expensive. Repeatedly pulling
the same video from one address trips bot detection ("Sign in to confirm you're
not a bot"), and once tripped it stays tripped for a while. Four or five runs
through the UI was enough to do it. The download is also the single slowest
stage: roughly a hundred seconds of ffmpeg on the sample video, paid again on
every run.

So this caches the pair of stages together. It has to be the pair: caching only
normalization would still re-download, and the download is the part that gets
blocked.

Keyed by URL rather than by content, because the content is not knowable until
after the download that the key exists to avoid.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from dialogue_extractor.domain.models import MediaManifest
from dialogue_extractor.infra.disk_cache import make_key
from dialogue_extractor.infra.logging import get_logger
from dialogue_extractor.interfaces.ports import Cache

log = get_logger(__name__)

VIDEO_SUFFIX = ".mp4"
AUDIO_SUFFIX = ".wav"


def media_fingerprint(*, max_height: int, target_audio_rate: int) -> str:
    """Everything about normalization that changes the bytes it produces.

    A cached entry made under different settings is not the same media, and
    serving it would silently answer a question nobody asked -- the same reason
    the transcript cache folds in the model and its quantization.
    """
    return f"h{max_height}-a{target_audio_rate}"


_TRACKING_PARAMS = frozenset(
    {"si", "feature", "app", "utm_source", "utm_medium", "utm_campaign", "pp"}
)


def canonical_url(raw: str) -> str:
    """Reduce the spellings of one video to a single cache key.

    `youtu.be/ID`, `youtube.com/watch?v=ID`, and the same with a `?si=` share
    tag are one video, and keying on the raw string stored a separate 286 MB
    copy for each. Worse, a user who pasted a share link would re-download a
    video already cached under its clean URL -- which is exactly the repeated
    downloading that trips bot detection.

    Deliberately conservative: it only strips tracking parameters and unifies
    the two YouTube host forms. Anything it does not recognise is returned with
    just its case and trailing slash tidied, because wrongly treating two
    different videos as one would serve the wrong media entirely.
    """
    url = raw.strip()
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url  # a local path; caller keys on it as-is

    host = parsed.netloc.lower().removeprefix("www.")

    # youtu.be/<id> and youtube.com/watch?v=<id> are the same video.
    if host in {"youtu.be", "youtube.com", "m.youtube.com"}:
        video_id = ""
        if host == "youtu.be":
            video_id = parsed.path.lstrip("/").split("/")[0]
        else:
            video_id = dict(parse_qsl(parsed.query)).get("v", "")
        if video_id:
            return f"https://youtube.com/watch?v={video_id}"

    query = urlencode(
        [(k, v) for k, v in parse_qsl(parsed.query) if k.lower() not in _TRACKING_PARAMS]
    )
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((parsed.scheme.lower(), host, path, "", query, ""))


def media_key(url: str, fingerprint: str) -> str:
    return make_key("media", canonical_url(url), fingerprint)


def _to_dict(manifest: MediaManifest) -> dict[str, Any]:
    return {
        "fps": manifest.fps,
        "frame_count": manifest.frame_count,
        "duration": manifest.duration,
        "width": manifest.width,
        "height": manifest.height,
        "cfr_forced": manifest.cfr_forced,
        "source_was_vfr": manifest.source_was_vfr,
        "video_sha256": manifest.video_sha256,
        "audio_sha256": manifest.audio_sha256,
    }


def _from_dict(data: dict[str, Any], video: Path, audio: Path) -> MediaManifest:
    return MediaManifest(
        video_path=video,
        audio_path=audio,
        fps=float(data["fps"]),
        frame_count=int(data["frame_count"]),
        duration=float(data["duration"]),
        width=int(data["width"]),
        height=int(data["height"]),
        cfr_forced=bool(data["cfr_forced"]),
        source_was_vfr=bool(data["source_was_vfr"]),
        video_sha256=str(data["video_sha256"]),
        audio_sha256=str(data["audio_sha256"]),
    )


def lookup(
    cache: Cache, url: str, fingerprint: str, work_dir: Path
) -> MediaManifest | None:
    """Return normalized media for `url` from the cache, or None on a miss.

    The files are copied into `work_dir` rather than referenced where they lie,
    so every later stage sees the same layout whether the media was cached or
    freshly produced. A local copy costs a second or two against the hundred
    seconds of re-encoding it replaces.
    """
    key = media_key(url, fingerprint)
    data = cache.get(key)
    if not isinstance(data, dict):
        return None

    cached_video = cache.path_for(key, VIDEO_SUFFIX)
    cached_audio = cache.path_for(key, AUDIO_SUFFIX)
    if not (cached_video.is_file() and cached_audio.is_file()):
        # Metadata without its media is a half-entry, most likely a cache
        # directory partially cleared by hand. Treat it as a miss.
        log.debug("media_cache.incomplete", key=key)
        return None

    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        video = work_dir / f"normalized{VIDEO_SUFFIX}"
        audio = work_dir / f"audio{AUDIO_SUFFIX}"
        if cached_video.resolve() != video.resolve():
            shutil.copyfile(cached_video, video)
        if cached_audio.resolve() != audio.resolve():
            shutil.copyfile(cached_audio, audio)
        manifest = _from_dict(data, video, audio)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        # Never let a cache problem fail a run that could still be done the
        # slow way.
        log.warning("media_cache.unusable", key=key, error=type(exc).__name__)
        return None

    log.info(
        "media_cache.hit",
        key=key,
        duration_s=round(manifest.duration, 2),
        note="skipped download and normalization",
    )
    return manifest


def store(
    cache: Cache, url: str, fingerprint: str, manifest: MediaManifest
) -> None:
    """Remember this URL's normalized media for the next run."""
    key = media_key(url, fingerprint)
    store_file = getattr(cache, "store_file", None)
    if store_file is None:
        return  # a cache that keeps no files, e.g. NullCache
    try:
        store_file(key, manifest.video_path, VIDEO_SUFFIX)
        store_file(key, manifest.audio_path, AUDIO_SUFFIX)
        cache.put(key, _to_dict(manifest))
    except OSError as exc:
        log.warning("media_cache.store_failed", key=key, error=type(exc).__name__)
        return
    log.info("media_cache.stored", key=key)
