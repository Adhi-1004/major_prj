"""Clip extraction against real media.

Two properties are checked on the produced *file* rather than on the flags that
were passed, because both fail silently:

  * faststart -- the service fails quietly on non-streamable MP4s
  * the frame offset -- forgetting it yields a real frame at the wrong moment
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.conftest import requires_ffmpeg

from dialogue_extractor.domain.enums import Mode
from dialogue_extractor.domain.errors import ClipBuildError
from dialogue_extractor.domain.models import Candidate, MediaManifest, RawMedia
from dialogue_extractor.domain.timing import first_frame_at_or_after
from dialogue_extractor.ingestion.ffprobe_validator import validate_and_probe
from dialogue_extractor.ingestion.normalizer import is_faststart, normalize
from dialogue_extractor.vision.clip_builder import build_clip, halve

pytestmark = requires_ffmpeg


@pytest.fixture(scope="module")
def manifest(counter_clip_25fps: Path, tmp_path_factory: pytest.TempPathFactory) -> MediaManifest:
    """Normalize the frame-indexed fixture, giving a real manifest to trim from."""
    work = tmp_path_factory.mktemp("clipwork")
    raw = RawMedia(path=counter_clip_25fps, source_uri=str(counter_clip_25fps), title="fixture")
    return normalize(raw, work)


def candidate(start: float, end: float) -> Candidate:
    return Candidate(start=start, end=end, speaker_id="SPEAKER_00", text="x", mode=Mode.TARGET)


class TestClipOutput:
    def test_produces_a_clip(self, manifest: MediaManifest, tmp_path: Path) -> None:
        clip = build_clip(candidate(1.0, 2.0), manifest, tmp_path)
        assert clip.path.exists()
        assert clip.size_bytes > 0

    def test_clip_is_faststart(self, manifest: MediaManifest, tmp_path: Path) -> None:
        """Required by the service, which fails quietly without it."""
        clip = build_clip(candidate(1.0, 2.0), manifest, tmp_path)
        assert is_faststart(clip.path) is True

    def test_clip_is_h264(self, manifest: MediaManifest, tmp_path: Path) -> None:
        clip = build_clip(candidate(1.0, 2.0), manifest, tmp_path)
        assert validate_and_probe(clip.path, require_audio=False).video_codec == "h264"

    def test_clip_duration_matches_the_window(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        clip = build_clip(candidate(1.0, 2.5), manifest, tmp_path)
        probed = validate_and_probe(clip.path, require_audio=False)
        assert probed.duration == pytest.approx(1.5, abs=0.15)

    def test_clip_keeps_constant_frame_rate(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        """The clip's own frame indexing must stay linear, or the returned
        frame id cannot be converted back to a timestamp."""
        clip = build_clip(candidate(1.0, 2.0), manifest, tmp_path)
        probed = validate_and_probe(clip.path, require_audio=False)
        assert probed.is_vfr is False

    def test_audio_is_extracted_alongside(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        clip = build_clip(candidate(1.0, 2.0), manifest, tmp_path)
        assert clip.audio_path is not None
        assert clip.audio_path.exists()

    def test_audio_can_be_skipped(self, manifest: MediaManifest, tmp_path: Path) -> None:
        clip = build_clip(candidate(1.0, 2.0), manifest, tmp_path, extract_audio=False)
        assert clip.audio_path is None


class TestFrameOffset:
    """Invariant 1, established at the point the offset is computed."""

    @pytest.mark.parametrize("start", [0.0, 0.4, 1.0, 2.36])
    def test_clip_start_frame_matches_the_window(
        self, manifest: MediaManifest, tmp_path: Path, start: float
    ) -> None:
        clip = build_clip(candidate(start, start + 1.0), manifest, tmp_path)
        assert clip.clip_start_frame == first_frame_at_or_after(start, manifest.fps)

    def test_window_at_zero_has_no_offset(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        clip = build_clip(candidate(0.0, 1.0), manifest, tmp_path)
        assert clip.clip_start_frame == 0

    def test_offset_recovers_the_original_frame(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        from dialogue_extractor.domain.timing import clip_frame_to_global

        clip = build_clip(candidate(2.0, 3.0), manifest, tmp_path)
        # Frame 0 of the clip is whatever frame the source had at t=2.0.
        assert clip_frame_to_global(0, clip.clip_start_frame) == first_frame_at_or_after(
            2.0, manifest.fps
        )


class TestExtractionIsFrameExact:
    def test_clip_starts_at_the_requested_frame(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        """The fixture encodes each frame's index in its pixels, so this reads
        the answer out of the clip rather than trusting ffmpeg's accounting.

        Re-encoding rather than stream-copying is what makes this hold: a copy
        can only cut on a keyframe, so the clip would silently start elsewhere
        and every derived frame index would be off by an unknown amount.
        """
        import subprocess

        from PIL import Image
        from tests.conftest import FFMPEG, decode_frame_index

        start = 2.0
        clip = build_clip(candidate(start, start + 0.5), manifest, tmp_path)

        first_png = tmp_path / "first.png"
        subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(clip.path), "-frames:v", "1", str(first_png)],
            check=True, capture_output=True, timeout=60,
        )

        decoded = decode_frame_index(Image.open(first_png).convert("RGB").getpixel((160, 120)))
        assert decoded == pytest.approx(clip.clip_start_frame, abs=1), (
            f"clip starts at source frame {decoded}, expected {clip.clip_start_frame}"
        )


class TestDownscaling:
    def test_tall_video_is_downscaled(self, manifest: MediaManifest, tmp_path: Path) -> None:
        clip = build_clip(candidate(1.0, 2.0), manifest, tmp_path, max_height=120)
        probed = validate_and_probe(clip.path, require_audio=False)
        assert probed.height <= 120

    def test_width_stays_even_for_yuv420p(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        clip = build_clip(candidate(1.0, 2.0), manifest, tmp_path, max_height=118)
        assert validate_and_probe(clip.path, require_audio=False).width % 2 == 0

    def test_short_video_is_not_upscaled(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        clip = build_clip(candidate(1.0, 2.0), manifest, tmp_path, max_height=2000)
        assert validate_and_probe(clip.path, require_audio=False).height == manifest.height


class TestFailureModes:
    def test_empty_window_is_rejected(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        with pytest.raises(ClipBuildError) as excinfo:
            build_clip(candidate(2.0, 2.0), manifest, tmp_path)
        assert excinfo.value.reason == "empty_window"

    def test_inverted_window_is_rejected(
        self, manifest: MediaManifest, tmp_path: Path
    ) -> None:
        with pytest.raises(ClipBuildError):
            build_clip(candidate(3.0, 1.0), manifest, tmp_path)


class TestHalve:
    def test_keeps_the_start(self) -> None:
        halved = halve(candidate(10.0, 30.0))
        assert halved.start == 10.0
        assert halved.end == pytest.approx(20.0)

    def test_preserves_other_fields(self) -> None:
        original = candidate(10.0, 30.0)
        halved = halve(original)
        assert halved.speaker_id == original.speaker_id
        assert halved.mode is original.mode

    def test_refuses_to_halve_nothing(self) -> None:
        with pytest.raises(ClipBuildError) as excinfo:
            halve(candidate(10.0, 10.0))
        assert excinfo.value.reason == "cannot_shrink"
