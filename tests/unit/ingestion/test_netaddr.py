"""Address classification.

This is a security boundary in both directions, and both directions have real
failure modes:

  * Too strict rejects legitimate hosts. A NAT64 address is reported by Python
    as `is_reserved`, so a blanket reserved-check refuses ordinary public sites
    on any DNS64 network -- which is exactly what happened to ok.ru resolving
    to 64:ff9b::5fa3:3d49.

  * Too lax lets a request reach an internal host. `::ffff:127.0.0.1` is not
    loopback, private, or reserved in its IPv6 form, yet connects to localhost.
"""

from __future__ import annotations

import ipaddress

import pytest

from dialogue_extractor.infra.netaddr import (
    describe_rejection,
    is_public_address,
    unwrap_embedded_ipv4,
)


class TestPublicAddresses:
    @pytest.mark.parametrize(
        "address",
        [
            "8.8.8.8",
            "95.163.61.73",  # ok.ru
            "95.163.61.74",
            "93.184.216.34",
            "1.1.1.1",
            "2606:4700:4700::1111",  # Cloudflare IPv6
            "2001:4860:4860::8888",  # Google IPv6
        ],
    )
    def test_accepted(self, address: str) -> None:
        assert is_public_address(address) is True


class TestPrivateAddresses:
    @pytest.mark.parametrize(
        "address",
        [
            "127.0.0.1",
            "127.0.0.53",
            "0.0.0.0",
            "10.0.0.1",
            "192.168.1.1",
            "172.16.0.1",
            "169.254.169.254",  # cloud metadata endpoint
            "224.0.0.1",  # multicast
            "240.0.0.1",  # reserved
            "::1",  # IPv6 loopback
            "fe80::1",  # IPv6 link-local
            "fc00::1",  # IPv6 unique-local
            "::",  # unspecified
        ],
    )
    def test_rejected(self, address: str) -> None:
        assert is_public_address(address) is False

    @pytest.mark.parametrize("value", ["garbage", "", "ok.ru", "999.999.999.999"])
    def test_non_addresses_are_rejected(self, value: str) -> None:
        """A hostname reaching this function is a bug, not a permission."""
        assert is_public_address(value) is False


class TestNat64:
    """The bug that prompted this module.

    ok.ru resolved to 64:ff9b::5fa3:3d49 on a DNS64 network. Python reports
    is_reserved=True *and* is_global=True for that prefix, so the original
    reserved-check refused a legitimate public host.
    """

    def test_nat64_carrying_a_public_ipv4_is_accepted(self) -> None:
        assert is_public_address("64:ff9b::5fa3:3d49") is True

    def test_the_embedded_address_is_the_real_one(self) -> None:
        unwrapped = unwrap_embedded_ipv4(ipaddress.ip_address("64:ff9b::5fa3:3d49"))
        assert unwrapped == ipaddress.ip_address("95.163.61.73")

    def test_second_ok_ru_address(self) -> None:
        assert unwrap_embedded_ipv4(ipaddress.ip_address("64:ff9b::5fa3:3d4a")) == (
            ipaddress.ip_address("95.163.61.74")
        )

    def test_nat64_carrying_a_private_ipv4_is_rejected(self) -> None:
        """Unwrapping must judge the embedded address, not merely accept NAT64."""
        # 64:ff9b::c0a8:101 carries 192.168.1.1
        assert is_public_address("64:ff9b::c0a8:101") is False

    def test_nat64_carrying_loopback_is_rejected(self) -> None:
        # 64:ff9b::7f00:1 carries 127.0.0.1
        assert is_public_address("64:ff9b::7f00:1") is False

    def test_local_use_nat64_prefix_is_also_unwrapped(self) -> None:
        """RFC 8215 local-use prefix, same translation semantics."""
        assert is_public_address("64:ff9b:1::7f00:1") is False

    def test_python_alone_would_have_got_this_wrong(self) -> None:
        """Documents precisely why the blanket reserved-check was wrong."""
        ip = ipaddress.ip_address("64:ff9b::5fa3:3d49")
        assert ip.is_reserved is True
        assert ip.is_global is True
        assert is_public_address(ip) is True


class TestIpv4Mapped:
    """The opposite failure: an IPv6 form that is not loopback/private/reserved
    but still connects to an internal host."""

    def test_mapped_loopback_is_rejected(self) -> None:
        assert is_public_address("::ffff:127.0.0.1") is False

    def test_mapped_private_is_rejected(self) -> None:
        assert is_public_address("::ffff:192.168.1.1") is False

    def test_mapped_metadata_endpoint_is_rejected(self) -> None:
        assert is_public_address("::ffff:169.254.169.254") is False

    def test_mapped_public_is_accepted(self) -> None:
        assert is_public_address("::ffff:8.8.8.8") is True

    def test_naive_check_would_have_accepted_mapped_loopback(self) -> None:
        ip = ipaddress.ip_address("::ffff:127.0.0.1")
        assert ip.is_private is True or ip.ipv4_mapped is not None
        assert is_public_address(ip) is False


class TestSixToFour:
    def test_6to4_carrying_private_is_rejected(self) -> None:
        # 2002:c0a8:0101:: carries 192.168.1.1
        assert is_public_address("2002:c0a8:101::") is False

    def test_6to4_carrying_public_is_accepted(self) -> None:
        # 2002:0808:0808:: carries 8.8.8.8
        assert is_public_address("2002:808:808::") is True


class TestAcceptsAddressObjects:
    def test_ipv4_object(self) -> None:
        assert is_public_address(ipaddress.ip_address("8.8.8.8")) is True

    def test_ipv6_object(self) -> None:
        assert is_public_address(ipaddress.ip_address("::1")) is False


class TestDescribeRejection:
    @pytest.mark.parametrize(
        ("address", "expected"),
        [
            ("127.0.0.1", "loopback"),
            ("10.0.0.1", "private"),
            ("169.254.169.254", "link-local"),
            ("224.0.0.1", "multicast"),
            ("garbage", "not a valid IP"),
        ],
    )
    def test_names_the_reason(self, address: str, expected: str) -> None:
        assert expected in describe_rejection(address)

    def test_mentions_the_embedded_address(self) -> None:
        """An operator seeing this in a log needs to know what it really was."""
        message = describe_rejection("::ffff:127.0.0.1")
        assert "127.0.0.1" in message
