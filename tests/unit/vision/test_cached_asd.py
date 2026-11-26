"""The detection cache: identical requests are answered from disk, others never are."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dialogue_extractor.domain.models import ASDFrame, ASDResult, BoundingBox, Face
from dialogue_extractor.infra.disk_cache import DiskCache
from dialogue_extractor.interfaces.ports import ASDProvider
from dialogue_extractor.vision.providers.cached_asd import CachedASDProvider


class CountingProvider(ASDProvider):
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def health_check(self) -> None:
        return None

    def detect(
        self,
        clip_path: Path,
        audio_path: Path | None,
        diarization: dict[str, Any],
        clip_start_frame: int,
        fps: float,
    ) -> ASDResult:
        self.calls += 1
        face = Face(1, BoundingBox(0.1, 0.2, 0.3, 0.4), True, 0.87, diarized_speaker_id=0)
        unassociated = Face(2, BoundingBox(0.5, 0.5, 0.1, 0.1), False, 0.6, None)
        return ASDResult(
            frames=(ASDFrame(0, (face, unassociated)), ASDFrame(1, ())),
            clip_start_frame=clip_start_frame,
            fps=fps,
            provider=self.name,
        )


def make(tmp_path: Path) -> tuple[CachedASDProvider, CountingProvider, Path]:
    inner = CountingProvider()
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip-bytes")
    return CachedASDProvider(inner, DiskCache(tmp_path / "cache")), inner, clip


DIAR = {"words": [{"text": "hi", "start": 0.0, "end": 0.2, "speaker_id": "speaker_1"}]}


def test_second_identical_request_is_served_from_cache(tmp_path: Path) -> None:
    provider, inner, clip = make(tmp_path)
    first = provider.detect(clip, None, DIAR, 100, 25.0)
    second = provider.detect(clip, None, DIAR, 100, 25.0)
    assert inner.calls == 1
    assert (provider.misses, provider.hits) == (1, 1)
    assert first.frames == second.frames


def test_cached_result_keeps_speaker_zero_and_none_distinct(tmp_path: Path) -> None:
    """0 is a real speaker; None is unassociated. Round-tripping must not merge them."""
    provider, _, clip = make(tmp_path)
    provider.detect(clip, None, DIAR, 0, 25.0)
    cached = provider.detect(clip, None, DIAR, 0, 25.0)
    faces = cached.frames[0].faces
    assert faces[0].diarized_speaker_id == 0
    assert faces[1].diarized_speaker_id is None


def test_offset_and_fps_come_from_the_request_not_the_cache(tmp_path: Path) -> None:
    provider, _, clip = make(tmp_path)
    provider.detect(clip, None, DIAR, 0, 25.0)
    again = provider.detect(clip, None, DIAR, 500, 25.0)
    assert again.clip_start_frame == 500


def test_different_diarization_is_a_different_request(tmp_path: Path) -> None:
    provider, inner, clip = make(tmp_path)
    provider.detect(clip, None, DIAR, 0, 25.0)
    other = {"words": [{"text": "hi", "start": 0.0, "end": 0.2, "speaker_id": "speaker_2"}]}
    provider.detect(clip, None, other, 0, 25.0)
    assert inner.calls == 2


def test_different_clip_bytes_are_a_different_request(tmp_path: Path) -> None:
    provider, inner, clip = make(tmp_path)
    provider.detect(clip, None, DIAR, 0, 25.0)
    clip.write_bytes(b"other-bytes")
    provider.detect(clip, None, DIAR, 0, 25.0)
    assert inner.calls == 2
