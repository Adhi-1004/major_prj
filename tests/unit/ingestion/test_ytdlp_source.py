"""Remote acquisition: failure classification, retry policy, output location.

No network. The point of these tests is that yt-dlp reports geo-blocking, DRM,
a dead link, and a transient blip all as the same exception type, and only one
of those is worth retrying.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dialogue_extractor.domain.errors import (
    DownloadError,
    DRMProtectedError,
    GeoRestrictedError,
    IngestionError,
    InvalidURLError,
)
from dialogue_extractor.ingestion.ytdlp_source import (
    YtDlpSource,
    classify_download_failure,
)


@pytest.fixture(autouse=True)
def no_real_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test in this module offline.

    detect_interference does a real DNS lookup and a real DoH query. Without
    this, retry-policy tests would take the DoH fallback path on any network
    that actually interferes with the host they name -- passing or failing
    depending on whose ISP runs them.
    """
    monkeypatch.setattr(
        "dialogue_extractor.ingestion.ytdlp_source.detect_interference",
        lambda host: (False, (), ()),
    )


class TestClassifyDownloadFailure:
    @pytest.mark.parametrize(
        "message",
        [
            "ERROR: The uploader has not made this video available in your country",
            "Video is geo restricted",
            "This content is geoblocked",
            "Sorry, this video is blocked in your country.",
        ],
    )
    def test_geo_restriction(self, message: str) -> None:
        error = classify_download_failure(message)
        assert isinstance(error, GeoRestrictedError)
        assert error.reason == "geo_restricted"

    @pytest.mark.parametrize(
        "message",
        [
            "This video is DRM protected",
            "ERROR: protected content cannot be downloaded",
            "The stream is encrypted",
        ],
    )
    def test_drm(self, message: str) -> None:
        error = classify_download_failure(message)
        assert isinstance(error, DRMProtectedError)
        assert error.reason == "drm_protected"

    @pytest.mark.parametrize(
        "message",
        [
            "ERROR: Video unavailable",
            "ERROR: Private video. Sign in if you've been granted access",
            "This video has been removed by the user",
            "ERROR: unable to download video data: HTTP Error 404: Not Found",
            "ERROR: Unsupported URL: https://example.com/foo",
            "ERROR: No video formats found!",
        ],
    )
    def test_permanent_failures_are_not_retryable(self, message: str) -> None:
        error = classify_download_failure(message)
        assert isinstance(error, DownloadError)
        assert error.reason == "unavailable"

    @pytest.mark.parametrize(
        "message",
        [
            "ERROR: unable to download video data: <urlopen error timed out>",
            "Connection reset by peer",
            "ERROR: Unable to connect to proxy",
        ],
    )
    def test_transient_failures_are_retryable(self, message: str) -> None:
        error = classify_download_failure(message)
        assert isinstance(error, DownloadError)
        assert error.reason == "download_failed"

    def test_every_classification_carries_remediation(self) -> None:
        for message in ("geo restricted", "drm", "Video unavailable", "timed out"):
            assert classify_download_failure(message).to_dict()["remediation"]

    def test_classification_is_an_ingestion_error(self) -> None:
        assert isinstance(classify_download_failure("anything"), IngestionError)

    def test_long_messages_are_truncated(self) -> None:
        error = classify_download_failure("timeout " + "x" * 5000)
        assert len(error.detail) < 600


class TestSupports:
    @pytest.mark.parametrize(
        "uri", ["https://ok.ru/video/1", "http://example.com/v.mp4", "HTTPS://OK.RU/video/1"]
    )
    def test_accepts_http_urls(self, uri: str) -> None:
        assert YtDlpSource().supports(uri) is True

    @pytest.mark.parametrize("uri", ["D:/clip.mp4", "file:///tmp/a.mp4", "./rel.mp4"])
    def test_rejects_local_paths(self, uri: str) -> None:
        assert YtDlpSource().supports(uri) is False


class TestAcquireValidatesFirst:
    def test_rejects_private_target_before_downloading(self, tmp_path: Path) -> None:
        """Validation must run before any network call."""
        with pytest.raises(InvalidURLError):
            YtDlpSource(resolve_dns=False).acquire("http://127.0.0.1/v.mp4", tmp_path)

    def test_rejects_bad_scheme(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidURLError):
            YtDlpSource(resolve_dns=False).acquire("ftp://example.com/v.mp4", tmp_path)


class TestRetryPolicy:
    def _source(self) -> YtDlpSource:
        return YtDlpSource(resolve_dns=False)

    def test_transient_failure_is_retried_then_reraised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}

        def always_timeout(url: str, dest: Path) -> dict[str, Any]:
            calls["n"] += 1
            raise classify_download_failure("connection timed out")

        source = self._source()
        monkeypatch.setattr(source, "_download_once", always_timeout)
        monkeypatch.setattr("dialogue_extractor.infra.retry.time.sleep", lambda _: None)

        with pytest.raises(DownloadError):
            source.acquire("https://ok.ru/video/1", tmp_path)
        assert calls["n"] == 3, "transient failures should exhaust all attempts"

    @pytest.mark.parametrize(
        ("message", "expected"),
        [("geo restricted", GeoRestrictedError), ("drm protected", DRMProtectedError)],
    )
    def test_permanent_failure_is_not_retried(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        message: str,
        expected: type[Exception],
    ) -> None:
        """Retrying a geo-block three times just makes the same answer slower."""
        calls = {"n": 0}

        def always_blocked(url: str, dest: Path) -> dict[str, Any]:
            calls["n"] += 1
            raise classify_download_failure(message)

        source = self._source()
        monkeypatch.setattr(source, "_download_once", always_blocked)
        monkeypatch.setattr("dialogue_extractor.infra.retry.time.sleep", lambda _: None)

        with pytest.raises(expected):
            source.acquire("https://ok.ru/video/1", tmp_path)
        assert calls["n"] == 1, "permanent failures must fail on the first attempt"

    def test_succeeds_after_a_transient_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}
        media = tmp_path / "source.mp4"

        def flaky(url: str, dest: Path) -> dict[str, Any]:
            calls["n"] += 1
            if calls["n"] == 1:
                raise classify_download_failure("timed out")
            media.write_bytes(b"video data")
            return {"title": "Sample", "filepath": str(media)}

        source = self._source()
        monkeypatch.setattr(source, "_download_once", flaky)
        monkeypatch.setattr("dialogue_extractor.infra.retry.time.sleep", lambda _: None)

        raw = source.acquire("https://ok.ru/video/1", tmp_path)
        assert calls["n"] == 2
        assert raw.path == media
        assert raw.title == "Sample"


class TestLocateOutput:
    """yt-dlp's reported filename can predate a merge or container change."""

    def test_prefers_reported_filepath(self, tmp_path: Path) -> None:
        actual = tmp_path / "source.mp4"
        actual.write_bytes(b"data")
        found = YtDlpSource()._locate_output({"filepath": str(actual)}, tmp_path)
        assert found == actual

    def test_falls_back_to_requested_downloads(self, tmp_path: Path) -> None:
        actual = tmp_path / "source.mp4"
        actual.write_bytes(b"data")
        info = {
            "_filename": str(tmp_path / "source.webm"),  # stale, pre-merge
            "requested_downloads": [{"filepath": str(actual)}],
        }
        assert YtDlpSource()._locate_output(info, tmp_path) == actual

    def test_falls_back_to_scanning_the_directory(self, tmp_path: Path) -> None:
        actual = tmp_path / "source.mkv"
        actual.write_bytes(b"data" * 100)
        assert YtDlpSource()._locate_output({}, tmp_path) == actual

    def test_ignores_part_files(self, tmp_path: Path) -> None:
        (tmp_path / "source.mp4.part").write_bytes(b"x" * 10_000)
        complete = tmp_path / "source.mp4"
        complete.write_bytes(b"x" * 100)
        assert YtDlpSource()._locate_output({}, tmp_path) == complete

    def test_ignores_zero_byte_files(self, tmp_path: Path) -> None:
        (tmp_path / "source.webm").write_bytes(b"")
        real = tmp_path / "source.mp4"
        real.write_bytes(b"data")
        assert YtDlpSource()._locate_output({}, tmp_path) == real

    def test_raises_when_nothing_was_written(self, tmp_path: Path) -> None:
        with pytest.raises(DownloadError) as excinfo:
            YtDlpSource()._locate_output({}, tmp_path)
        assert excinfo.value.reason == "output_missing"


class TestOptions:
    def test_format_selector_caps_resolution(self, tmp_path: Path) -> None:
        """Normalization re-encodes anyway, so a 4K source is wasted download time."""
        options = YtDlpSource()._build_options(tmp_path)
        assert "height<=1080" in options["format"]

    def test_output_is_confined_to_dest_dir(self, tmp_path: Path) -> None:
        options = YtDlpSource()._build_options(tmp_path)
        assert options["paths"]["home"] == str(tmp_path)

    def test_outtmpl_is_a_bare_filename(self, tmp_path: Path) -> None:
        """yt-dlp joins outtmpl onto paths["home"]. A dir-qualified template
        here yields output/work/output/work/source.mp4."""
        options = YtDlpSource()._build_options(tmp_path)
        assert options["outtmpl"] == "source.%(ext)s"
        assert str(tmp_path) not in options["outtmpl"]

    def test_playlists_are_not_followed(self, tmp_path: Path) -> None:
        assert YtDlpSource()._build_options(tmp_path)["noplaylist"] is True

    def test_certificate_checking_is_on(self, tmp_path: Path) -> None:
        assert YtDlpSource()._build_options(tmp_path)["nocheckcertificate"] is False


class TestDnsFallback:
    """A hijacked lookup resets during the TLS handshake, which is
    indistinguishable from a flaky connection at the call site. The two
    resolvers must be compared before concluding anything."""

    def _failing_then_ok(self, tmp_path: Path):  # type: ignore[no-untyped-def]
        calls = {"n": 0}
        media = tmp_path / "source.mp4"

        def download(url: str, dest: Path) -> dict[str, Any]:
            calls["n"] += 1
            if calls["n"] <= 3:  # first attempt exhausts its 3 retries
                raise classify_download_failure("connection forcibly closed by the remote host")
            media.write_bytes(b"video")
            return {"title": "Recovered", "filepath": str(media)}

        return calls, download

    def test_retries_over_doh_when_interference_is_detected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls, download = self._failing_then_ok(tmp_path)
        source = YtDlpSource(resolve_dns=False, dns_mode="auto")
        monkeypatch.setattr(source, "_download_once", download)
        monkeypatch.setattr("dialogue_extractor.infra.retry.time.sleep", lambda _: None)
        monkeypatch.setattr(
            "dialogue_extractor.ingestion.ytdlp_source.detect_interference",
            lambda host: (True, ("202.83.21.14",), ("95.163.61.73",)),
        )

        raw = source.acquire("https://ok.ru/video/1", tmp_path)
        assert raw.title == "Recovered"
        assert calls["n"] == 4, "should retry after switching resolvers"

    def test_does_not_retry_when_dns_looks_fine(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without evidence of interference, the original error stands."""
        calls, download = self._failing_then_ok(tmp_path)
        source = YtDlpSource(resolve_dns=False, dns_mode="auto")
        monkeypatch.setattr(source, "_download_once", download)
        monkeypatch.setattr("dialogue_extractor.infra.retry.time.sleep", lambda _: None)
        monkeypatch.setattr(
            "dialogue_extractor.ingestion.ytdlp_source.detect_interference",
            lambda host: (False, ("95.163.61.73",), ("95.163.61.73",)),
        )

        with pytest.raises(DownloadError):
            source.acquire("https://ok.ru/video/1", tmp_path)
        assert calls["n"] == 3, "must not attempt a fourth time"

    def test_system_mode_never_falls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls, download = self._failing_then_ok(tmp_path)
        source = YtDlpSource(resolve_dns=False, dns_mode="system")
        monkeypatch.setattr(source, "_download_once", download)
        monkeypatch.setattr("dialogue_extractor.infra.retry.time.sleep", lambda _: None)

        def must_not_be_called(host: str):  # type: ignore[no-untyped-def]
            raise AssertionError("interference detection should not run in system mode")

        monkeypatch.setattr(
            "dialogue_extractor.ingestion.ytdlp_source.detect_interference", must_not_be_called
        )

        with pytest.raises(DownloadError):
            source.acquire("https://ok.ru/video/1", tmp_path)
        assert calls["n"] == 3

    def test_geo_block_is_not_mistaken_for_dns_interference(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A geo-block is a GeoRestrictedError, so the DoH path never engages."""
        source = YtDlpSource(resolve_dns=False, dns_mode="auto")
        monkeypatch.setattr(
            source,
            "_download_once",
            lambda url, dest: (_ for _ in ()).throw(
                classify_download_failure("geo restricted")
            ),
        )
        monkeypatch.setattr("dialogue_extractor.infra.retry.time.sleep", lambda _: None)

        with pytest.raises(GeoRestrictedError):
            source.acquire("https://ok.ru/video/1", tmp_path)


class TestFfmpegLocation:
    def test_points_ytdlp_at_the_validated_binary(self, tmp_path: Path) -> None:
        """yt-dlp needs ffmpeg for HLS and muxing, and finds it on PATH --
        which is not necessarily a usable build."""
        options = YtDlpSource()._build_options(tmp_path)
        assert "ffmpeg_location" in options
        assert Path(options["ffmpeg_location"]).exists()


class TestNoImpersonation:
    def test_impersonation_is_not_enabled(self, tmp_path: Path) -> None:
        """curl_cffi resolves inside libcurl and never calls socket.getaddrinfo,
        so enabling impersonation silently bypasses the DoH fallback and breaks
        downloads on any network that interferes with DNS."""
        assert "impersonate" not in YtDlpSource()._build_options(tmp_path)


class TestEmptyErrorMessages:
    """A bare assertion inside yt-dlp carries no message. Reporting
    "Download failed: " with nothing after it tells the user nothing at all."""

    def _run_with_ytdlp_raising(
        self, monkeypatch: pytest.MonkeyPatch, exc: BaseException, tmp_path: Path
    ) -> DownloadError:
        import yt_dlp

        class FakeYDL:
            def __init__(self, opts: dict[str, Any]) -> None: ...
            def __enter__(self) -> FakeYDL:
                return self
            def __exit__(self, *args: Any) -> None: ...
            def extract_info(self, url: str, download: bool = True) -> dict[str, Any]:
                raise exc

        monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
        source = YtDlpSource(resolve_dns=False)
        with pytest.raises(DownloadError) as excinfo:
            source._download_once("https://ok.ru/video/1", tmp_path)
        return excinfo.value

    def test_blank_message_names_the_exception_type(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        error = self._run_with_ytdlp_raising(monkeypatch, AssertionError(""), tmp_path)
        assert "AssertionError" in error.detail
        assert error.detail.strip() != "Download failed:"

    def test_whitespace_only_message_is_treated_as_blank(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        error = self._run_with_ytdlp_raising(monkeypatch, RuntimeError("   "), tmp_path)
        assert "RuntimeError" in error.detail

    def test_real_message_is_preserved(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        error = self._run_with_ytdlp_raising(
            monkeypatch, RuntimeError("connection timed out"), tmp_path
        )
        assert "connection timed out" in error.detail
