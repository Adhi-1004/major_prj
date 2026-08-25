"""Input classification and URL safety checks.

Two jobs:

  1. Decide whether the input names a remote URL or a local file, so the right
     VideoSource handles it. The local path exists because the brief's
     acquisition step can fail for reasons outside our control -- a site
     change, a geo-block -- and a manual file fallback keeps the rest of the
     pipeline demonstrable.

  2. Refuse URLs we should not fetch. The system takes a URL from an untrusted
     caller and hands it to a downloader, so it must not be usable as a probe
     of the host's own network or filesystem.

Pure apart from DNS resolution, which is opt-in via `resolve_dns`.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from enum import StrEnum
from pathlib import Path
from urllib.parse import unquote, urlparse, urlunparse

from dialogue_extractor.config.constants import ALLOWED_URL_SCHEMES
from dialogue_extractor.domain.errors import InvalidURLError
from dialogue_extractor.infra.netaddr import describe_rejection, is_public_address


class SourceKind(StrEnum):
    REMOTE = "remote"
    LOCAL = "local"


# A bare Windows drive letter parses as a one-character URL scheme
# ("d:\\clips\\a.mp4" -> scheme "d"), which would otherwise be rejected as an
# unsupported scheme rather than recognized as a local path.
_WINDOWS_DRIVE = re.compile(r"^[a-zA-Z]:[\\/]")

_BLOCKED_HOST_SUFFIXES = (".localhost", ".local", ".internal")


def classify_source(raw: str) -> SourceKind:
    """Decide whether `raw` names a remote URL or a local file."""
    if not raw or not raw.strip():
        raise InvalidURLError(
            "empty_input", "No video URL or file path was provided.",
            remediation="Pass --url with a public video URL, or a path to a local file.",
        )

    candidate = raw.strip()

    if _WINDOWS_DRIVE.match(candidate):
        return SourceKind.LOCAL

    parsed = urlparse(candidate)

    if parsed.scheme == "file":
        return SourceKind.LOCAL
    if parsed.scheme in ALLOWED_URL_SCHEMES:
        return SourceKind.REMOTE
    if not parsed.scheme:
        return SourceKind.LOCAL

    raise InvalidURLError(
        "unsupported_scheme",
        f"Unsupported URL scheme {parsed.scheme!r}. Only http and https are fetched.",
        remediation="Provide an http(s) URL, or a path to a local video file.",
    )


def _reject_private_host(host: str) -> None:
    """Refuse hosts that would turn the downloader into an internal probe."""
    lowered = host.lower().strip("[]")

    if lowered in {"localhost", ""} or lowered.endswith(_BLOCKED_HOST_SUFFIXES):
        raise InvalidURLError(
            "private_host",
            f"Refusing to fetch from {host!r}: it names a local or internal host.",
            remediation="Provide a publicly accessible video URL.",
        )

    try:
        ipaddress.ip_address(lowered)
    except ValueError:
        return  # A hostname, not a literal IP. DNS is checked separately.

    if not is_public_address(lowered):
        raise InvalidURLError(
            "private_address",
            f"Refusing to fetch from {host!r}: it is a {describe_rejection(lowered)}.",
            remediation="Provide a publicly accessible video URL.",
        )


def _reject_private_dns(host: str) -> None:
    """Resolve `host` and refuse if it points anywhere non-public.

    Separate from the literal-IP check because a public-looking hostname can
    resolve to a private address.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise InvalidURLError(
            "dns_resolution_failed",
            f"Could not resolve host {host!r}.",
            remediation="Check the URL and your network connection.",
        ) from exc

    # A host is acceptable if *any* of its addresses is public. Requiring every
    # address to pass would reject dual-stack hosts on NAT64/DNS64 networks,
    # where the synthesized IPv6 form sits alongside the plain IPv4 one.
    addresses = [str(info[4][0]) for info in infos]
    if addresses and not any(is_public_address(a) for a in addresses):
        raise InvalidURLError(
            "private_address",
            f"Refusing to fetch from {host!r}: it resolves only to non-public addresses "
            f"({', '.join(addresses[:4])}).",
            remediation="Provide a publicly accessible video URL.",
        )


def validate_remote_url(raw: str, *, resolve_dns: bool = False) -> str:
    """Validate and normalize a remote URL.

    Args:
        resolve_dns: Also resolve the host and refuse private addresses. Off by
            default so unit tests stay offline and deterministic; the CLI turns
            it on.

    Returns:
        The normalized URL.
    """
    candidate = raw.strip()
    parsed = urlparse(candidate)

    if parsed.scheme not in ALLOWED_URL_SCHEMES:
        raise InvalidURLError(
            "unsupported_scheme",
            f"Unsupported URL scheme {parsed.scheme!r}. Only http and https are fetched.",
            remediation="Provide an http(s) URL.",
        )

    if not parsed.netloc:
        raise InvalidURLError(
            "missing_host",
            f"URL {raw!r} has no host component.",
            remediation="Provide a full URL, for example https://ok.ru/video/248244667877",
        )

    host = parsed.hostname or ""
    _reject_private_host(host)
    if resolve_dns:
        _reject_private_dns(host)

    # Drop any embedded credentials and the fragment; neither is useful to a
    # downloader and both are things we would rather not carry into logs.
    netloc = parsed.hostname or ""
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"

    return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, ""))


def resolve_local_path(raw: str) -> Path:
    """Turn a local input (plain path or file:// URL) into an existing file path."""
    candidate = raw.strip()

    if candidate.lower().startswith("file://"):
        parsed = urlparse(candidate)
        # Percent-decode first: a file:// URL escapes spaces as %20, and paths
        # under a user profile very often contain them.
        path_part = unquote(parsed.path)
        # file:///D:/clips/a.mp4 -> /D:/clips/a.mp4 on Windows; strip the slash.
        if _WINDOWS_DRIVE.match(path_part.lstrip("/")):
            path_part = path_part.lstrip("/")
        path = Path(path_part)
    else:
        path = Path(candidate)

    try:
        expanded = path.expanduser()
    except RuntimeError as exc:
        raise InvalidURLError("bad_path", f"Could not interpret path {raw!r}.") from exc

    if not expanded.exists():
        raise InvalidURLError(
            "file_not_found",
            f"Local file not found: {expanded}",
            remediation="Check the path, or pass a public video URL instead.",
        )
    if not expanded.is_file():
        raise InvalidURLError(
            "not_a_file",
            f"{expanded} is a directory, not a video file.",
            remediation="Point at a video file.",
        )
    return expanded.resolve()
