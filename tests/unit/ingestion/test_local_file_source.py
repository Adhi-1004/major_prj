"""Local-file acquisition -- the fallback when a download cannot be made to work."""

from __future__ import annotations

from pathlib import Path

import pytest

from dialogue_extractor.domain.errors import InvalidURLError
from dialogue_extractor.ingestion.local_file_source import LocalFileSource


@pytest.fixture
def sample(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"video bytes")
    return path


class TestSupports:
    @pytest.mark.parametrize(
        "uri",
        ["D:/clip.mp4", "D:\\clip.mp4", "./clip.mp4", "/abs/clip.mp4", "clip.mp4",
         "file:///D:/clip.mp4"],
    )
    def test_accepts_paths_and_file_urls(self, uri: str) -> None:
        assert LocalFileSource().supports(uri) is True

    @pytest.mark.parametrize("uri", ["https://ok.ru/video/1", "http://example.com/v.mp4"])
    def test_rejects_remote_urls(self, uri: str) -> None:
        assert LocalFileSource().supports(uri) is False


class TestAcquire:
    def test_copies_into_the_work_directory(self, sample: Path, tmp_path: Path) -> None:
        """A run must never mutate or depend on the user's original file."""
        work = tmp_path / "work"
        raw = LocalFileSource().acquire(str(sample), work)

        assert raw.path.parent == work
        assert raw.path.read_bytes() == b"video bytes"
        assert sample.exists(), "the original must be left untouched"

    def test_preserves_the_extension(self, tmp_path: Path) -> None:
        source = tmp_path / "clip.mkv"
        source.write_bytes(b"data")
        raw = LocalFileSource().acquire(str(source), tmp_path / "work")
        assert raw.path.suffix == ".mkv"

    def test_accepts_a_file_url(self, sample: Path, tmp_path: Path) -> None:
        raw = LocalFileSource().acquire(sample.resolve().as_uri(), tmp_path / "work")
        assert raw.path.read_bytes() == b"video bytes"

    def test_records_the_source_and_title(self, sample: Path, tmp_path: Path) -> None:
        raw = LocalFileSource().acquire(str(sample), tmp_path / "work")
        assert raw.title == "clip"
        assert "clip.mp4" in raw.source_uri

    def test_creates_the_work_directory(self, sample: Path, tmp_path: Path) -> None:
        work = tmp_path / "deep" / "work"
        LocalFileSource().acquire(str(sample), work)
        assert work.is_dir()

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidURLError) as excinfo:
            LocalFileSource().acquire(str(tmp_path / "absent.mp4"), tmp_path / "work")
        assert excinfo.value.reason == "file_not_found"

    def test_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidURLError) as excinfo:
            LocalFileSource().acquire(str(tmp_path), tmp_path / "work")
        assert excinfo.value.reason == "not_a_file"

    def test_path_with_spaces(self, tmp_path: Path) -> None:
        """User profile paths routinely contain spaces."""
        directory = tmp_path / "My Videos"
        directory.mkdir()
        source = directory / "my clip.mp4"
        source.write_bytes(b"data")

        raw = LocalFileSource().acquire(str(source), tmp_path / "work")
        assert raw.path.read_bytes() == b"data"

    def test_file_url_with_spaces(self, tmp_path: Path) -> None:
        """file:// percent-encodes spaces; they must be decoded back."""
        directory = tmp_path / "My Videos"
        directory.mkdir()
        source = directory / "my clip.mp4"
        source.write_bytes(b"data")

        raw = LocalFileSource().acquire(source.resolve().as_uri(), tmp_path / "work")
        assert raw.path.read_bytes() == b"data"
