"""URL classification and safety.

The system hands a caller-supplied URL to a downloader, so these tests cover
both "does it route to the right source" and "does it refuse what it should".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dialogue_extractor.domain.errors import InvalidURLError
from dialogue_extractor.ingestion.url_validator import (
    SourceKind,
    classify_source,
    resolve_local_path,
    validate_remote_url,
)


class TestClassifyRemote:
    @pytest.mark.parametrize(
        "url",
        [
            "https://ok.ru/video/248244667877",
            "http://ok.ru/video/248244667877",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://vimeo.com/123456",
            "  https://ok.ru/video/1  ",  # surrounding whitespace
        ],
    )
    def test_http_urls_are_remote(self, url: str) -> None:
        assert classify_source(url) is SourceKind.REMOTE


class TestClassifyLocal:
    @pytest.mark.parametrize(
        "value",
        [
            "D:/clips/sample.mp4",
            "D:\\clips\\sample.mp4",
            "c:/video.mp4",
            "./relative/clip.mp4",
            "../up/clip.mp4",
            "/absolute/unix/clip.mp4",
            "plain_name.mp4",
            "file:///D:/clips/sample.mp4",
        ],
    )
    def test_paths_and_file_urls_are_local(self, value: str) -> None:
        assert classify_source(value) is SourceKind.LOCAL

    def test_windows_drive_letter_is_not_mistaken_for_a_scheme(self) -> None:
        """"d:\\clip.mp4" parses as scheme "d"; it must not be rejected."""
        assert classify_source("d:\\clip.mp4") is SourceKind.LOCAL


class TestClassifyRejects:
    @pytest.mark.parametrize(
        "value",
        [
            "javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "ftp://example.com/video.mp4",
            "gopher://example.com/",
            "ssh://example.com/",
        ],
    )
    def test_dangerous_or_unsupported_schemes(self, value: str) -> None:
        with pytest.raises(InvalidURLError) as excinfo:
            classify_source(value)
        assert excinfo.value.reason == "unsupported_scheme"

    @pytest.mark.parametrize("value", ["", "   ", "\t\n"])
    def test_empty_input(self, value: str) -> None:
        with pytest.raises(InvalidURLError) as excinfo:
            classify_source(value)
        assert excinfo.value.reason == "empty_input"


class TestValidateRemoteUrl:
    def test_returns_normalized_url(self) -> None:
        assert (
            validate_remote_url("https://ok.ru/video/248244667877")
            == "https://ok.ru/video/248244667877"
        )

    def test_preserves_query_string(self) -> None:
        result = validate_remote_url("https://www.youtube.com/watch?v=abc123")
        assert result == "https://www.youtube.com/watch?v=abc123"

    def test_strips_fragment(self) -> None:
        assert validate_remote_url("https://ok.ru/video/1#t=30") == "https://ok.ru/video/1"

    def test_strips_embedded_credentials(self) -> None:
        """Credentials in a URL would otherwise reach logs and yt-dlp options."""
        result = validate_remote_url("https://user:pass@ok.ru/video/1")
        assert "user" not in result
        assert "pass" not in result
        assert result == "https://ok.ru/video/1"

    def test_preserves_explicit_port(self) -> None:
        assert validate_remote_url("https://example.com:8443/v.mp4") == (
            "https://example.com:8443/v.mp4"
        )

    def test_rejects_missing_host(self) -> None:
        with pytest.raises(InvalidURLError) as excinfo:
            validate_remote_url("https:///video/1")
        assert excinfo.value.reason == "missing_host"

    def test_rejects_non_http_scheme(self) -> None:
        with pytest.raises(InvalidURLError):
            validate_remote_url("ftp://example.com/v.mp4")


class TestSsrfProtection:
    """A caller-supplied URL must not become a probe of the host's own network."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost/video.mp4",
            "http://localhost:8080/video.mp4",
            "http://127.0.0.1/video.mp4",
            "http://127.0.0.53/video.mp4",
            "http://0.0.0.0/video.mp4",
            "http://10.0.0.5/video.mp4",
            "http://192.168.1.1/video.mp4",
            "http://172.16.0.1/video.mp4",
            "http://169.254.169.254/latest/meta-data/",  # cloud metadata endpoint
            "http://[::1]/video.mp4",
            "http://service.internal/video.mp4",
            "http://box.local/video.mp4",
        ],
    )
    def test_refuses_local_and_private_targets(self, url: str) -> None:
        with pytest.raises(InvalidURLError) as excinfo:
            validate_remote_url(url)
        assert excinfo.value.reason in {"private_host", "private_address"}

    def test_public_addresses_are_allowed(self) -> None:
        assert validate_remote_url("http://8.8.8.8/video.mp4") == "http://8.8.8.8/video.mp4"

    def test_dns_check_is_opt_in(self) -> None:
        """Off by default so unit tests stay offline and deterministic."""
        validate_remote_url("https://ok.ru/video/1")  # must not resolve DNS


class TestResolveLocalPath:
    def test_resolves_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "clip.mp4"
        target.write_bytes(b"data")
        assert resolve_local_path(str(target)) == target.resolve()

    def test_resolves_file_url(self, tmp_path: Path) -> None:
        target = tmp_path / "clip.mp4"
        target.write_bytes(b"data")
        as_url = target.resolve().as_uri()
        assert resolve_local_path(as_url) == target.resolve()

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidURLError) as excinfo:
            resolve_local_path(str(tmp_path / "nope.mp4"))
        assert excinfo.value.reason == "file_not_found"

    def test_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidURLError) as excinfo:
            resolve_local_path(str(tmp_path))
        assert excinfo.value.reason == "not_a_file"

    def test_error_carries_remediation(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidURLError) as excinfo:
            resolve_local_path(str(tmp_path / "nope.mp4"))
        assert excinfo.value.to_dict()["remediation"]
