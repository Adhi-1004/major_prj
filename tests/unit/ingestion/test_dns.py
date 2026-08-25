"""DNS-over-HTTPS resolution and interference detection.

No network: every DoH query and system lookup is stubbed. The behaviour that
matters is *when* interference is concluded -- a false positive would route
ordinary traffic through a third-party resolver for no reason, and a false
negative leaves the original failure unexplained.
"""

from __future__ import annotations

import ipaddress
import json
import socket
import urllib.error
from typing import Any

import pytest

from dialogue_extractor.infra import dns


@pytest.fixture(autouse=True)
def clear_dns_cache() -> None:
    dns.clear_cache()


def stub_doh(monkeypatch: pytest.MonkeyPatch, answers: dict[str, list[str]]) -> None:
    def fake_query(endpoint: str, hostname: str) -> tuple[str, ...]:
        if hostname not in answers:
            raise urllib.error.URLError("NXDOMAIN")
        return tuple(answers[hostname])

    monkeypatch.setattr(dns, "_query_doh", fake_query)


def stub_system(monkeypatch: pytest.MonkeyPatch, answers: dict[str, list[str]]) -> None:
    """Stand in for the system resolver.

    Literal IP addresses pass straight through, as the real getaddrinfo does --
    doh_resolution relies on that to turn a DoH answer into an addrinfo tuple.
    """

    def fake_getaddrinfo(
        host: Any, port: Any = None, *args: Any, **kwargs: Any
    ) -> list[Any]:
        if isinstance(host, str):
            try:
                ipaddress.ip_address(host)
            except ValueError:
                pass
            else:
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (host, port or 0))]
        if host not in answers:
            raise socket.gaierror("NXDOMAIN")
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0))
            for ip in answers[host]
        ]

    monkeypatch.setattr(dns.socket, "getaddrinfo", fake_getaddrinfo)


class TestResolveDoh:
    def test_returns_public_addresses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub_doh(monkeypatch, {"ok.ru": ["95.163.61.73", "95.163.61.74"]})
        assert dns.resolve_doh("ok.ru") == ("95.163.61.73", "95.163.61.74")

    def test_discards_private_answers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A resolver must never be able to point this downloader inward."""
        stub_doh(monkeypatch, {"evil.test": ["127.0.0.1", "10.0.0.5", "192.168.1.1"]})
        assert dns.resolve_doh("evil.test") == ()

    def test_keeps_only_the_public_ones(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub_doh(monkeypatch, {"mixed.test": ["10.0.0.5", "93.184.216.34"]})
        assert dns.resolve_doh("mixed.test") == ("93.184.216.34",)

    def test_empty_tuple_when_every_endpoint_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Falls back to the system resolver rather than hard-failing."""
        stub_doh(monkeypatch, {})
        assert dns.resolve_doh("ok.ru") == ()

    def test_falls_through_to_the_second_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []

        def flaky(endpoint: str, hostname: str) -> tuple[str, ...]:
            calls.append(endpoint)
            if len(calls) == 1:
                raise TimeoutError("first endpoint down")
            return ("95.163.61.73",)

        monkeypatch.setattr(dns, "_query_doh", flaky)
        assert dns.resolve_doh("ok.ru") == ("95.163.61.73",)
        assert len(calls) == 2

    def test_malformed_json_is_survivable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def bad(endpoint: str, hostname: str) -> tuple[str, ...]:
            raise json.JSONDecodeError("bad", "", 0)

        monkeypatch.setattr(dns, "_query_doh", bad)
        assert dns.resolve_doh("ok.ru") == ()

    def test_result_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"n": 0}

        def counting(endpoint: str, hostname: str) -> tuple[str, ...]:
            calls["n"] += 1
            return ("95.163.61.73",)

        monkeypatch.setattr(dns, "_query_doh", counting)
        dns.resolve_doh("ok.ru")
        dns.resolve_doh("ok.ru")
        assert calls["n"] == 1

    def test_cache_can_be_bypassed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"n": 0}

        def counting(endpoint: str, hostname: str) -> tuple[str, ...]:
            calls["n"] += 1
            return ("95.163.61.73",)

        monkeypatch.setattr(dns, "_query_doh", counting)
        dns.resolve_doh("ok.ru")
        dns.resolve_doh("ok.ru", use_cache=False)
        assert calls["n"] == 2


class TestDetectInterference:
    def test_disjoint_answers_mean_interference(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The observed real case: ISP returns its own address instead."""
        stub_system(monkeypatch, {"ok.ru": ["202.83.21.14"]})
        stub_doh(monkeypatch, {"ok.ru": ["95.163.61.73", "95.163.61.74"]})

        interfered, system, doh = dns.detect_interference("ok.ru")
        assert interfered is True
        assert system == ("202.83.21.14",)
        assert set(doh) == {"95.163.61.73", "95.163.61.74"}

    def test_identical_answers_mean_no_interference(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_system(monkeypatch, {"example.com": ["93.184.216.34"]})
        stub_doh(monkeypatch, {"example.com": ["93.184.216.34"]})
        assert dns.detect_interference("example.com")[0] is False

    def test_overlapping_answers_mean_no_interference(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A CDN routinely returns different subsets to different resolvers.

        Requiring *zero* overlap avoids calling that interference.
        """
        stub_system(monkeypatch, {"cdn.test": ["1.2.3.4", "1.2.3.5"]})
        stub_doh(monkeypatch, {"cdn.test": ["1.2.3.5", "1.2.3.6"]})
        assert dns.detect_interference("cdn.test")[0] is False

    def test_no_conclusion_when_doh_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cannot compare against nothing, so must not claim interference."""
        stub_system(monkeypatch, {"ok.ru": ["202.83.21.14"]})
        stub_doh(monkeypatch, {})
        assert dns.detect_interference("ok.ru")[0] is False

    def test_no_conclusion_when_system_resolution_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_system(monkeypatch, {})
        stub_doh(monkeypatch, {"ok.ru": ["95.163.61.73"]})
        assert dns.detect_interference("ok.ru")[0] is False


class TestDohResolutionContext:
    def test_patches_only_the_named_hosts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub_system(monkeypatch, {"ok.ru": ["202.83.21.14"], "other.test": ["1.1.1.1"]})
        stub_doh(monkeypatch, {"ok.ru": ["95.163.61.73"]})

        with dns.doh_resolution(["ok.ru"]):
            patched = socket.getaddrinfo("ok.ru", 443)
            untouched = socket.getaddrinfo("other.test", 443)

        assert patched[0][4][0] == "95.163.61.73"
        assert untouched[0][4][0] == "1.1.1.1"

    def test_covers_subdomains(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ok.ru serves media from vd*.mycdn.me-style subdomains too."""
        stub_system(monkeypatch, {"www.ok.ru": ["202.83.21.14"]})
        stub_doh(monkeypatch, {"www.ok.ru": ["95.163.61.73"]})

        with dns.doh_resolution(["ok.ru"]):
            assert socket.getaddrinfo("www.ok.ru", 443)[0][4][0] == "95.163.61.73"

    def test_does_not_match_a_lookalike_suffix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """"notok.ru" must not be captured by a rule for "ok.ru"."""
        stub_system(monkeypatch, {"notok.ru": ["1.1.1.1"]})
        stub_doh(monkeypatch, {"notok.ru": ["9.9.9.9"]})

        with dns.doh_resolution(["ok.ru"]):
            assert socket.getaddrinfo("notok.ru", 443)[0][4][0] == "1.1.1.1"

    def test_restores_the_original_resolver(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub_system(monkeypatch, {"ok.ru": ["202.83.21.14"]})
        stub_doh(monkeypatch, {"ok.ru": ["95.163.61.73"]})
        before = socket.getaddrinfo

        with dns.doh_resolution(["ok.ru"]):
            assert socket.getaddrinfo is not before
        assert socket.getaddrinfo is before

    def test_restores_the_resolver_even_on_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_system(monkeypatch, {"ok.ru": ["202.83.21.14"]})
        stub_doh(monkeypatch, {"ok.ru": ["95.163.61.73"]})
        before = socket.getaddrinfo

        with pytest.raises(RuntimeError), dns.doh_resolution(["ok.ru"]):
            raise RuntimeError("download blew up")
        assert socket.getaddrinfo is before

    def test_empty_hostname_list_is_a_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        before = socket.getaddrinfo
        with dns.doh_resolution([]):
            assert socket.getaddrinfo is before

    def test_falls_back_to_system_when_doh_has_no_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_system(monkeypatch, {"ok.ru": ["202.83.21.14"]})
        stub_doh(monkeypatch, {})

        with dns.doh_resolution(["ok.ru"]):
            assert socket.getaddrinfo("ok.ru", 443)[0][4][0] == "202.83.21.14"
