"""Remote acquisition via yt-dlp.

Uses the Python API rather than shelling out, so failures arrive as typed
exceptions carrying yt-dlp's own diagnostic instead of a parsed stderr blob.

The main work here is *classifying* failures. yt-dlp reports geo-blocking, DRM,
a dead link, and a transient network blip all as DownloadError; retrying is only
correct for the last of those, so the message is inspected and mapped to a
specific error type. Retrying a geo-block three times turns a clear, immediate
failure into a slow one that says the same thing.
"""

from __future__ import annotations

import re
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dialogue_extractor.config.constants import DOWNLOAD_MAX_RETRIES, DOWNLOAD_RETRY_BASE_DELAY
from dialogue_extractor.domain.errors import (
    DownloadError,
    DRMProtectedError,
    FFmpegNotFoundError,
    GeoRestrictedError,
    IngestionError,
)
from dialogue_extractor.domain.models import RawMedia
from dialogue_extractor.infra.dns import detect_interference, doh_resolution
from dialogue_extractor.infra.ffmpeg_runner import resolve_binary
from dialogue_extractor.infra.logging import get_logger, stage
from dialogue_extractor.infra.retry import retry
from dialogue_extractor.ingestion.url_validator import validate_remote_url
from dialogue_extractor.interfaces.ports import VideoSource

log = get_logger(__name__)

# Prefer a progressive/mergeable H.264 MP4 at or below 1080p. Normalization
# re-encodes anyway, so there is nothing to gain from a larger source, and a
# lot of download time to lose.
FORMAT_SELECTOR = (
    "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/"
    "best[height<=1080][ext=mp4]/"
    "best[height<=1080]/"
    "best"
)

_GEO_PATTERNS = (
    # Matches both "not available in your country" and YouTube's longer
    # "has not made this video available in your country".
    "available in your country",
    "available in your location",
    "geo restrict",
    "geo-restrict",
    "geoblock",
    "geo block",
    "blocked in your country",
    "not available from your location",
)
_DRM_PATTERNS = (
    "drm",
    "protected content",
    "encrypted",
)
_PERMANENT_PATTERNS = (
    "video unavailable",
    "private video",
    "removed by the user",
    "does not exist",
    "http error 404",
    "account has been terminated",
    "unsupported url",
    "no video formats found",
)


def classify_download_failure(message: str) -> IngestionError:
    """Map a yt-dlp error message to a specific, correctly-retryable error.

    Kept as a free function so it can be unit tested against real message
    strings without invoking a download.
    """
    lowered = message.lower()

    if any(p in lowered for p in _GEO_PATTERNS):
        return GeoRestrictedError(
            "geo_restricted",
            f"The source refused the request on geographic grounds: {message.strip()[:400]}",
            remediation="This video is not available from this location. Try a local file instead.",
        )
    if any(p in lowered for p in _DRM_PATTERNS):
        return DRMProtectedError(
            "drm_protected",
            f"The source is DRM-protected and cannot be decoded: {message.strip()[:400]}",
            remediation="DRM-protected media cannot be processed. Provide an unprotected source.",
        )
    if any(p in lowered for p in _PERMANENT_PATTERNS):
        return DownloadError(
            "unavailable",
            f"The video is not available: {message.strip()[:400]}",
            remediation="Check the URL is correct and the video is still public.",
        )
    return DownloadError(
        "download_failed",
        f"Download failed: {message.strip()[:400]}",
        remediation="This may be transient. Check your network, or pass a local file instead.",
    )


def _is_retryable(error: IngestionError) -> bool:
    """Only a generic failure is worth another attempt."""
    return isinstance(error, DownloadError) and error.reason == "download_failed"


class YtDlpSource(VideoSource):
    """Acquires media from a public URL."""

    def __init__(
        self,
        *,
        resolve_dns: bool = True,
        quiet: bool = True,
        dns_mode: str = "auto",
    ) -> None:
        """
        Args:
            dns_mode: "auto" retries via DoH when DNS interference is detected,
                "doh" always resolves via DoH, "system" never does.
        """
        self._resolve_dns = resolve_dns
        self._quiet = quiet
        self._dns_mode = dns_mode

    def supports(self, uri: str) -> bool:
        return bool(re.match(r"^https?://", uri.strip(), re.IGNORECASE))

    def _build_options(self, dest_dir: Path) -> dict[str, Any]:
        options: dict[str, Any] = {
            "format": FORMAT_SELECTOR,
            # Filename only. yt-dlp joins this onto paths["home"], so an
            # absolute or dir-qualified template here produces a doubled path
            # (output/work/output/work/source.mp4).
            "outtmpl": "source.%(ext)s",
            "merge_output_format": "mp4",
            "quiet": self._quiet,
            "no_warnings": self._quiet,
            "noprogress": self._quiet,
            "noplaylist": True,
            "retries": 2,
            "socket_timeout": 30,
            "nocheckcertificate": False,
            # Never let a URL cause writes outside dest_dir.
            "paths": {"home": str(dest_dir)},
        }

        # yt-dlp finds ffmpeg on PATH, which is not necessarily a usable build --
        # it needs one for HLS and for muxing separate streams. Point it at the
        # same binary the rest of the pipeline validated.
        with suppress(FFmpegNotFoundError):
            options["ffmpeg_location"] = str(resolve_binary("ffmpeg").parent)

        # Deliberately NOT setting "impersonate". curl_cffi resolves hostnames
        # inside libcurl, in C, so it never calls socket.getaddrinfo -- which
        # means it bypasses the DNS-over-HTTPS fallback below entirely. On a
        # network that interferes with DNS, turning impersonation on *breaks*
        # downloads that otherwise succeed. The plain handler is also enough for
        # the sites tested. If a site ever demands TLS impersonation, it needs
        # an ImpersonateTarget object (not the string "chrome", which asserts)
        # and a different answer to DNS.

        return options

    def acquire(self, uri: str, dest_dir: Path) -> RawMedia:
        url = validate_remote_url(uri, resolve_dns=self._resolve_dns)
        dest_dir.mkdir(parents=True, exist_ok=True)
        hostname = urlparse(url).hostname or ""

        with stage("ingestion", url=url):
            if self._dns_mode == "doh":
                with doh_resolution([hostname]):
                    info = self._download_with_retry(url, dest_dir)
            else:
                try:
                    info = self._download_with_retry(url, dest_dir)
                except DownloadError as exc:
                    if self._dns_mode != "auto":
                        raise
                    info = self._retry_via_doh_if_interfered(url, dest_dir, hostname, exc)

        path = self._locate_output(info, dest_dir)
        return RawMedia(path=path, source_uri=url, title=info.get("title"))

    def _retry_via_doh_if_interfered(
        self, url: str, dest_dir: Path, hostname: str, original: DownloadError
    ) -> dict[str, Any]:
        """Re-attempt over DoH, but only if plain DNS is demonstrably wrong.

        A hijacked lookup resets during the TLS handshake, which is
        indistinguishable from a flaky connection at the call site -- so the
        two resolvers are compared before concluding anything, and the finding
        is logged rather than silently worked around.
        """
        interfered, system_addresses, doh_addresses = detect_interference(hostname)
        if not interfered:
            raise original

        log.warning(
            "dns.interference_detected",
            hostname=hostname,
            system_resolved=list(system_addresses),
            actual=list(doh_addresses),
            action="retrying over DNS-over-HTTPS",
        )
        with doh_resolution([hostname]):
            return self._download_with_retry(url, dest_dir)

    def _download_with_retry(self, url: str, dest_dir: Path) -> dict[str, Any]:
        @retry(
            attempts=DOWNLOAD_MAX_RETRIES,
            base_delay=DOWNLOAD_RETRY_BASE_DELAY,
            retry_on=(DownloadError,),
            give_up_on=(GeoRestrictedError, DRMProtectedError),
        )
        def attempt() -> dict[str, Any]:
            return self._download_once(url, dest_dir)

        return attempt()

    def _download_once(self, url: str, dest_dir: Path) -> dict[str, Any]:
        import yt_dlp

        try:
            with yt_dlp.YoutubeDL(self._build_options(dest_dir)) as ydl:
                info = ydl.extract_info(url, download=True)
        except Exception as exc:  # yt-dlp raises several unrelated types
            # Some yt-dlp failures carry an empty message (a bare assertion, for
            # one), which would otherwise produce "Download failed: " and tell
            # the user nothing. Fall back to naming the exception type.
            message = str(exc).strip() or f"{type(exc).__name__} (no message)"
            error = classify_download_failure(message)
            if not _is_retryable(error):
                # Signal "do not retry" by raising a type the decorator gives up on,
                # or a non-retryable DownloadError subtype.
                raise error from exc
            raise error from exc

        if info is None:
            raise DownloadError("no_metadata", "yt-dlp returned no information for this URL.")
        return dict(info)

    def _locate_output(self, info: dict[str, Any], dest_dir: Path) -> Path:
        """Find what was actually written.

        yt-dlp's reported filename can predate a merge or a container change, so
        it is checked first and then the directory is searched.
        """
        candidates: list[Path] = []

        for key in ("filepath", "_filename"):
            value = info.get(key)
            if value:
                candidates.append(Path(value))

        for downloaded in info.get("requested_downloads") or []:
            value = downloaded.get("filepath")
            if value:
                candidates.append(Path(value))

        for candidate in candidates:
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate

        media = [
            p
            for p in dest_dir.glob("source.*")
            if p.is_file() and p.stat().st_size > 0 and p.suffix.lower() != ".part"
        ]
        if media:
            return max(media, key=lambda p: p.stat().st_size)

        raise DownloadError(
            "output_missing",
            "The download reported success but no media file was written.",
            remediation="Check available disk space and permissions on the output directory.",
        )
