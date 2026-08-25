"""DNS-over-HTTPS resolution, for networks that interfere with plain DNS.

Some ISPs answer lookups for particular domains with an address of their own
rather than the real one. The connection then reaches that host instead, and is
reset during the TLS handshake -- which surfaces as a generic "connection
forcibly closed" and looks exactly like a flaky network, so it gets retried
pointlessly.

This module detects that case by comparing what the system resolver returns
against what a DoH resolver returns, and can substitute the DoH answer for the
duration of a download. Only the resolution step changes: the hostname is still
what gets sent as SNI and in the Host header, so certificate validation stays
fully in force. An attacker-controlled DoH answer would still have to present a
valid certificate for the hostname.

Ordinary networks are unaffected -- interference is detected, not assumed.
"""

from __future__ import annotations

import contextlib
import json
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Sequence
from typing import Any

from dialogue_extractor.infra.logging import get_logger
from dialogue_extractor.infra.netaddr import is_public_address

log = get_logger(__name__)

DOH_ENDPOINTS: tuple[str, ...] = (
    "https://cloudflare-dns.com/dns-query",
    "https://dns.google/resolve",
)

_DOH_TIMEOUT = 10.0
_A_RECORD = 1

_cache: dict[str, tuple[str, ...]] = {}


def _query_doh(endpoint: str, hostname: str) -> tuple[str, ...]:
    request = urllib.request.Request(
        f"{endpoint}?name={urllib.parse.quote(hostname)}&type=A",
        headers={"accept": "application/dns-json"},
    )
    with urllib.request.urlopen(
        request, timeout=_DOH_TIMEOUT, context=ssl.create_default_context()
    ) as response:
        payload: dict[str, Any] = json.load(response)

    return tuple(
        answer["data"]
        for answer in payload.get("Answer", [])
        if answer.get("type") == _A_RECORD and answer.get("data")
    )


def resolve_doh(hostname: str, *, use_cache: bool = True) -> tuple[str, ...]:
    """Resolve `hostname` to public A records via DoH.

    Returns an empty tuple if every endpoint fails, so callers can fall back to
    the system resolver rather than hard-failing on a network that blocks DoH
    too. Non-public answers are discarded: a resolver should never be able to
    redirect this downloader at an internal address.
    """
    if use_cache and hostname in _cache:
        return _cache[hostname]

    for endpoint in DOH_ENDPOINTS:
        try:
            addresses = tuple(a for a in _query_doh(endpoint, hostname) if is_public_address(a))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            log.debug("doh.endpoint_failed", endpoint=endpoint, error=type(exc).__name__)
            continue
        if addresses:
            _cache[hostname] = addresses
            log.debug("doh.resolved", hostname=hostname, addresses=list(addresses))
            return addresses

    log.debug("doh.unresolved", hostname=hostname)
    return ()


def resolve_system(hostname: str) -> tuple[str, ...]:
    """Resolve `hostname` with the system resolver."""
    try:
        infos = socket.getaddrinfo(hostname, None, family=socket.AF_INET)
    except socket.gaierror:
        return ()
    return tuple({str(info[4][0]) for info in infos})


def detect_interference(hostname: str) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
    """Compare system and DoH answers for `hostname`.

    Returns (interfered, system_addresses, doh_addresses). Interference means
    both resolvers answered and their answers share no address at all --
    a stronger signal than mere difference, which CDNs produce routinely.
    """
    system = resolve_system(hostname)
    doh = resolve_doh(hostname)
    if not system or not doh:
        return False, system, doh
    return (not set(system) & set(doh)), system, doh


@contextlib.contextmanager
def doh_resolution(hostnames: Sequence[str]) -> Iterator[None]:
    """Resolve the given hostnames (and their subdomains) via DoH inside this block.

    Patches socket.getaddrinfo, which is what every HTTP client in the process
    ultimately calls -- including the ones inside yt-dlp, which offers no
    resolver hook of its own. Everything else resolves normally.
    """
    suffixes = tuple(h.lower().lstrip(".") for h in hostnames if h)
    if not suffixes:
        yield
        return

    original = socket.getaddrinfo

    def patched(
        host: Any, port: Any, family: int = 0, type: int = 0, proto: int = 0, flags: int = 0
    ) -> list[Any]:
        if isinstance(host, str):
            lowered = host.lower()
            if any(lowered == s or lowered.endswith(f".{s}") for s in suffixes):
                for address in resolve_doh(lowered):
                    with contextlib.suppress(socket.gaierror):
                        # Resolve the literal IP to build a proper addrinfo tuple.
                        return original(address, port, family, type, proto, flags)
        return original(host, port, family, type, proto, flags)

    socket.getaddrinfo = patched  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.getaddrinfo = original  # type: ignore[assignment]


def clear_cache() -> None:
    _cache.clear()
