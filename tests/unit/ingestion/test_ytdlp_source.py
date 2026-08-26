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
    RateLimitedError,
)
from dialogue_extractor.ingestion.ytdlp_source import (
    YtDlpSource,
    _is_retryable,
    _source_tag,
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
        media = tmp_path / f"source_{_source_tag('https://ok.ru/video/1')}.mp4"

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
    """yt-dlp's reported filename can predate a merge or container change.

    Every lookup is scoped to the requesting URL's tag. See
    TestLocateOutputNeverReturnsAnotherVideo for why that scoping exists.
    """

    URL = "https://youtu.be/CoBS_v70xbE"

    def tagged(self, tmp_path: Path, suffix: str) -> Path:
        from dialogue_extractor.ingestion.ytdlp_source import _source_tag

        return tmp_path / f"source_{_source_tag(self.URL)}{suffix}"

    def locate(self, info: dict, tmp_path: Path) -> Path:  # type: ignore[type-arg]
        from dialogue_extractor.ingestion.ytdlp_source import _source_tag

        return YtDlpSource()._locate_output(info, tmp_path, _source_tag(self.URL))

    def test_prefers_reported_filepath(self, tmp_path: Path) -> None:
        actual = self.tagged(tmp_path, ".mp4")
        actual.write_bytes(b"data")
        assert self.locate({"filepath": str(actual)}, tmp_path) == actual

    def test_falls_back_to_requested_downloads(self, tmp_path: Path) -> None:
        actual = self.tagged(tmp_path, ".mp4")
        actual.write_bytes(b"data")
        info = {
            "_filename": str(self.tagged(tmp_path, ".webm")),  # stale, pre-merge
            "requested_downloads": [{"filepath": str(actual)}],
        }
        assert self.locate(info, tmp_path) == actual

    def test_falls_back_to_scanning_the_directory(self, tmp_path: Path) -> None:
        actual = self.tagged(tmp_path, ".mkv")
        actual.write_bytes(b"data" * 100)
        assert self.locate({}, tmp_path) == actual

    def test_ignores_part_files(self, tmp_path: Path) -> None:
        complete = self.tagged(tmp_path, ".mp4")
        complete.write_bytes(b"data" * 100)
        self.tagged(tmp_path, ".mp4.part").write_bytes(b"x" * 5000)
        assert self.locate({}, tmp_path) == complete

    def test_ignores_zero_byte_files(self, tmp_path: Path) -> None:
        real = self.tagged(tmp_path, ".mkv")
        real.write_bytes(b"data" * 100)
        self.tagged(tmp_path, ".mp4").write_bytes(b"")
        assert self.locate({}, tmp_path) == real

    def test_raises_when_nothing_was_written(self, tmp_path: Path) -> None:
        with pytest.raises(DownloadError):
            self.locate({}, tmp_path)


class TestOptions:
    def test_format_selector_caps_resolution(self, tmp_path: Path) -> None:
        """Normalization re-encodes anyway, so a 4K source is wasted download time."""
        options = YtDlpSource()._build_options(tmp_path, "https://youtu.be/CoBS_v70xbE")
        assert "height<=1080" in options["format"]

    def test_output_is_confined_to_dest_dir(self, tmp_path: Path) -> None:
        options = YtDlpSource()._build_options(tmp_path, "https://youtu.be/CoBS_v70xbE")
        assert options["paths"]["home"] == str(tmp_path)

    def test_outtmpl_is_a_bare_filename(self, tmp_path: Path) -> None:
        """yt-dlp joins outtmpl onto paths["home"]. A dir-qualified template
        here yields output/work/output/work/source.mp4.

        The name is no longer the fixed "source.%(ext)s" it once was: sharing
        one destination between videos made yt-dlp skip the second download and
        return the first video's bytes. See TestDownloadsDoNotCollide.
        """
        options = YtDlpSource()._build_options(tmp_path, "https://youtu.be/CoBS_v70xbE")
        template = str(options["outtmpl"])
        assert template.startswith("source_")
        assert template.endswith(".%(ext)s")
        assert str(tmp_path) not in template

    def test_playlists_are_not_followed(self, tmp_path: Path) -> None:
        options = YtDlpSource()._build_options(tmp_path, "https://youtu.be/X")
        assert options["noplaylist"] is True

    def test_certificate_checking_is_on(self, tmp_path: Path) -> None:
        options = YtDlpSource()._build_options(tmp_path, "https://youtu.be/X")
        assert options["nocheckcertificate"] is False


class TestDnsFallback:
    """A hijacked lookup resets during the TLS handshake, which is
    indistinguishable from a flaky connection at the call site. The two
    resolvers must be compared before concluding anything."""

    def _failing_then_ok(self, tmp_path: Path):  # type: ignore[no-untyped-def]
        calls = {"n": 0}
        media = tmp_path / f"source_{_source_tag('https://ok.ru/video/1')}.mp4"

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
        options = YtDlpSource()._build_options(tmp_path, "https://youtu.be/CoBS_v70xbE")
        assert "ffmpeg_location" in options
        assert Path(options["ffmpeg_location"]).exists()


class TestNoImpersonation:
    def test_impersonation_is_not_enabled(self, tmp_path: Path) -> None:
        """curl_cffi resolves inside libcurl and never calls socket.getaddrinfo,
        so enabling impersonation silently bypasses the DoH fallback and breaks
        downloads on any network that interferes with DNS."""
        options = YtDlpSource()._build_options(tmp_path, "https://youtu.be/X")
        assert "impersonate" not in options


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


class TestRateLimiting:
    """Bot detection is not a transient download failure, and retrying it hurts.

    It arrives looking ordinary -- YouTube phrases it as an error about signing
    in -- but every retry is another request from the address already under
    suspicion. Classified as a generic failure it was retried twice, which
    deepened the block instead of getting past it.
    """

    @pytest.mark.parametrize(
        "message",
        [
            "ERROR: [youtube] CoBS_v70xbE: Sign in to confirm you're not a bot. "
            "Use --cookies-from-browser or --cookies for the authentication.",
            "Sign in to confirm you are not a bot",
            "HTTP Error 429: Too Many Requests",
            "Requested format is not available; rate limit exceeded",
        ],
    )
    def test_it_is_recognised(self, message: str) -> None:
        error = classify_download_failure(message)
        assert isinstance(error, RateLimitedError)
        assert error.reason == "rate_limited"

    def test_it_is_never_retried(self) -> None:
        error = classify_download_failure("Sign in to confirm you're not a bot")
        assert _is_retryable(error) is False

    def test_a_genuine_blip_is_still_retried(self) -> None:
        """The distinction has to cut both ways or a flaky network stops working."""
        error = classify_download_failure("Unable to download webpage: timed out")
        assert _is_retryable(error) is True

    def test_the_remediation_names_a_way_out(self) -> None:
        """"This may be transient, check your network" was actively misleading:
        the network is fine, and the fixes are a local file, cookies, or time."""
        error = classify_download_failure("Sign in to confirm you're not a bot")
        assert error.remediation is not None
        remediation = error.remediation.lower()
        assert "file path" in remediation
        assert "cookies" in remediation

    def test_it_does_not_swallow_unrelated_failures(self) -> None:
        for message, reason in [
            ("Video unavailable", "unavailable"),
            ("This video is not available in your country", "geo_restricted"),
        ]:
            assert classify_download_failure(message).reason == reason


class TestDownloadsDoNotCollide:
    """Two videos must never share a download destination.

    yt-dlp skips a download whose target file already exists -- it reports
    "has already been downloaded" and hands back what is there. With one shared
    `source.mp4` in the working directory, requesting a second video returned
    the *first* one's bytes, which then flowed through normalization, caching
    and transcription as though it were the video that had been asked for. The
    result was a transcript of the wrong film, reported with full confidence.
    """

    def options_for(self, url: str, tmp_path: Path) -> dict[str, object]:
        return YtDlpSource()._build_options(tmp_path, url)

    def test_different_urls_get_different_destinations(self, tmp_path: Path) -> None:
        a = self.options_for("https://youtu.be/AAAAAAAAAAA", tmp_path)["outtmpl"]
        b = self.options_for("https://youtu.be/BBBBBBBBBBB", tmp_path)["outtmpl"]
        assert a != b

    def test_the_same_url_is_stable_across_runs(self, tmp_path: Path) -> None:
        """Stable, or a re-run could never reuse anything."""
        url = "https://youtu.be/CoBS_v70xbE"
        assert (
            self.options_for(url, tmp_path)["outtmpl"]
            == self.options_for(url, tmp_path)["outtmpl"]
        )

    def test_a_stale_file_is_replaced_not_reused(self, tmp_path: Path) -> None:
        """The second guard: a partial or leftover file under the same name is
        overwritten rather than resumed."""
        assert self.options_for("https://youtu.be/X", tmp_path)["overwrites"] is True

    def test_the_template_stays_a_bare_filename(self, tmp_path: Path) -> None:
        """yt-dlp joins outtmpl onto paths["home"]; a directory here doubles it."""
        template = str(self.options_for("https://youtu.be/X", tmp_path)["outtmpl"])
        assert "/" not in template and "\\" not in template
        assert template.endswith(".%(ext)s")

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("https://youtu.be/AAA", "https://youtu.be/BBB"),
            ("https://ok.ru/video/111", "https://ok.ru/video/222"),
            ("https://youtu.be/AAA", "https://ok.ru/video/111"),
        ],
    )
    def test_across_hosts_too(self, a: str, b: str, tmp_path: Path) -> None:
        assert self.options_for(a, tmp_path) != self.options_for(b, tmp_path)


class TestLocateOutputNeverReturnsAnotherVideo:
    """The directory search must not pick up a previous video's leftovers.

    This is the second half of the collision. Scoping the *download* filename
    was not enough: when yt-dlp reported no filename of its own, `_locate_output`
    fell back to globbing a bare `source.*` and returned whatever an earlier run
    had left in the directory. That file was then normalized, transcribed and
    reported as the requested video, with full confidence and a plausible frame.
    Returning nothing is recoverable; returning the wrong film is not.
    """

    def locate(self, info: dict, dest: Path, url: str) -> Path:  # type: ignore[type-arg]
        from dialogue_extractor.ingestion.ytdlp_source import _source_tag

        return YtDlpSource()._locate_output(info, dest, _source_tag(url))

    def test_a_previous_videos_file_is_not_returned(self, tmp_path: Path) -> None:
        (tmp_path / "source.mp4").write_bytes(b"PREVIOUS VIDEO")
        (tmp_path / "source_deadbeef.mp4").write_bytes(b"ANOTHER VIDEO")

        with pytest.raises(DownloadError) as excinfo:
            self.locate({}, tmp_path, "https://youtu.be/BRAND_NEW")
        assert excinfo.value.reason == "output_missing"

    def test_this_videos_file_is_found_by_search(self, tmp_path: Path) -> None:
        from dialogue_extractor.ingestion.ytdlp_source import _source_tag

        url = "https://youtu.be/CoBS_v70xbE"
        mine = tmp_path / f"source_{_source_tag(url)}.mp4"
        mine.write_bytes(b"MY VIDEO")
        (tmp_path / "source.mp4").write_bytes(b"SOMEONE ELSE")

        assert self.locate({}, tmp_path, url).read_bytes() == b"MY VIDEO"

    def test_a_reported_path_for_a_different_source_is_refused(
        self, tmp_path: Path
    ) -> None:
        """Even if yt-dlp names a real, non-empty file, a foreign one is wrong."""
        stale = tmp_path / "source.mp4"
        stale.write_bytes(b"PREVIOUS VIDEO")

        with pytest.raises(DownloadError):
            self.locate({"filepath": str(stale)}, tmp_path, "https://youtu.be/NEW")

    def test_a_correctly_reported_path_is_used(self, tmp_path: Path) -> None:
        from dialogue_extractor.ingestion.ytdlp_source import _source_tag

        url = "https://youtu.be/CoBS_v70xbE"
        mine = tmp_path / f"source_{_source_tag(url)}.mkv"
        mine.write_bytes(b"MERGED OUTPUT")

        got = self.locate({"filepath": str(mine)}, tmp_path, url)
        assert got == mine

    def test_requested_downloads_is_honoured_when_scoped(
        self, tmp_path: Path
    ) -> None:
        from dialogue_extractor.ingestion.ytdlp_source import _source_tag

        url = "https://youtu.be/X"
        mine = tmp_path / f"source_{_source_tag(url)}.mp4"
        mine.write_bytes(b"OK")

        info = {"requested_downloads": [{"filepath": str(mine)}]}
        assert self.locate(info, tmp_path, url) == mine

    def test_partial_fragments_are_ignored(self, tmp_path: Path) -> None:
        from dialogue_extractor.ingestion.ytdlp_source import _source_tag

        url = "https://youtu.be/X"
        tag = _source_tag(url)
        (tmp_path / f"source_{tag}.mp4.part").write_bytes(b"HALF")

        with pytest.raises(DownloadError):
            self.locate({}, tmp_path, url)
