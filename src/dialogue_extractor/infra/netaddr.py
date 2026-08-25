"""Deciding whether an IP address is safe to fetch from.

Used by both the URL validator and the DoH resolver, so the two cannot drift
apart and start disagreeing about what counts as public.

The subtlety this module exists for: several IPv6 forms *embed* an IPv4 address,
and the embedded address is the one that actually determines where traffic goes.
Classifying the outer IPv6 address directly gives the wrong answer in both
directions:

  * `64:ff9b::5fa3:3d49` is a NAT64 address carrying 95.163.61.73 -- a perfectly
    ordinary public host. Python reports `is_reserved == True` for it (while
    also reporting `is_global == True`), so a blanket reserved-check refuses a
    legitimate address on any NAT64/DNS64 network.

  * `::ffff:127.0.0.1` is an IPv4-mapped loopback. Its IPv6 form is not
    loopback, private, or reserved -- so a naive check *accepts* it and the
    connection reaches localhost anyway.

So embedded addresses are unwrapped first, then judged.
"""

from __future__ import annotations

import ipaddress

# RFC 6052: the well-known prefix for IPv4/IPv6 translation.
_NAT64_WELL_KNOWN = ipaddress.ip_network("64:ff9b::/96")
# RFC 8215: local-use prefix for the same purpose.
_NAT64_LOCAL_USE = ipaddress.ip_network("64:ff9b:1::/48")

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


def unwrap_embedded_ipv4(ip: IPAddress) -> IPAddress:
    """Return the IPv4 address embedded in `ip`, or `ip` unchanged.

    Handles IPv4-mapped (::ffff:a.b.c.d), 6to4 (2002::/16), and NAT64
    (64:ff9b::/96 and 64:ff9b:1::/48) forms.
    """
    if not isinstance(ip, ipaddress.IPv6Address):
        return ip

    if ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    if ip.sixtofour is not None:
        return ip.sixtofour
    if ip in _NAT64_WELL_KNOWN or ip in _NAT64_LOCAL_USE:
        # The low 32 bits carry the IPv4 address.
        return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return ip


def is_public_address(value: str | IPAddress) -> bool:
    """True if `value` is an address we are willing to fetch from.

    Anything not parseable as an IP is False -- callers pass literals here, and
    a hostname reaching this function would be a bug, not a permission.
    """
    if isinstance(value, str):
        try:
            ip: IPAddress = ipaddress.ip_address(value)
        except ValueError:
            return False
    else:
        ip = value

    ip = unwrap_embedded_ipv4(ip)

    if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified:
        return False
    if ip.is_private:
        return False

    if isinstance(ip, ipaddress.IPv4Address):
        # 240.0.0.0/4 and friends -- not routable.
        return not ip.is_reserved

    # For IPv6, `is_reserved` covers large swathes that are in fact routable
    # (NAT64 among them), so `is_global` is the meaningful test. Any embedded
    # IPv4 was already unwrapped above.
    return bool(ip.is_global)


def describe_rejection(value: str) -> str:
    """Short reason an address was refused, for the error message."""
    try:
        ip: IPAddress = ipaddress.ip_address(value)
    except ValueError:
        return "not a valid IP address"

    unwrapped = unwrap_embedded_ipv4(ip)
    via = f" (embedded IPv4 {unwrapped})" if unwrapped != ip else ""

    # Most specific first: 169.254.0.0/16 is both link-local and private in
    # Python's classification, and "link-local" is the more useful answer.
    if unwrapped.is_loopback:
        return f"loopback address{via}"
    if unwrapped.is_link_local:
        return f"link-local address{via}"
    if unwrapped.is_multicast:
        return f"multicast address{via}"
    if unwrapped.is_unspecified:
        return f"unspecified address{via}"
    if unwrapped.is_private:
        return f"private address{via}"
    return f"non-public address{via}"
