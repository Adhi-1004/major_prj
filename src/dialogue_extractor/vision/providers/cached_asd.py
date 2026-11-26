"""Caching active-speaker detection results, as a decorator over any provider.

The detector is the one stage that costs a network round-trip -- and on a
metered or rate-limited service, money and quota -- every time it runs. Its
answer is a function of exactly two things: the clip it was sent and the
diarization sent with it. So both are hashed into the key, and an identical
request is answered from disk.

This is what makes the ablation study affordable and *controlled*. Four of the
five ablations (speaker check, persistence threshold, shot cuts, onset
constraint) change only how the per-frame output is interpreted, not what is
sent. With this cache they are evaluated on byte-identical detector output, so
any difference between configurations is caused by the rule being ablated and
not by run-to-run variation in a remote service.

The provider name is part of the key, so switching backends never serves one
model's answer as another's.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dialogue_extractor.domain.models import ASDFrame, ASDResult, BoundingBox, Face
from dialogue_extractor.infra.disk_cache import make_key
from dialogue_extractor.infra.logging import get_logger
from dialogue_extractor.interfaces.ports import ASDProvider, Cache

log = get_logger(__name__)

_CHUNK = 1024 * 1024


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def frames_to_payload(frames: tuple[ASDFrame, ...]) -> list[dict[str, Any]]:
    return [
        {
            "frame_id": frame.frame_id,
            "faces": [
                {
                    "face_id": face.face_id,
                    "bbox": [face.bbox.x, face.bbox.y, face.bbox.width, face.bbox.height],
                    "is_speaking": face.is_speaking,
                    "confidence": face.face_detection_confidence,
                    "speaker": face.diarized_speaker_id,
                }
                for face in frame.faces
            ],
        }
        for frame in frames
    ]


def frames_from_payload(payload: list[dict[str, Any]]) -> tuple[ASDFrame, ...]:
    return tuple(
        ASDFrame(
            frame_id=int(item["frame_id"]),
            faces=tuple(
                Face(
                    face_id=int(face["face_id"]),
                    bbox=BoundingBox(*(float(v) for v in face["bbox"])),
                    is_speaking=bool(face["is_speaking"]),
                    face_detection_confidence=float(face["confidence"]),
                    diarized_speaker_id=(None if face["speaker"] is None else int(face["speaker"])),
                )
                for face in item["faces"]
            ),
        )
        for item in payload
    )


class CachedASDProvider(ASDProvider):
    """Wraps an ASDProvider with a content-addressed cache.

    Counts hits and misses so an evaluation can report how many requests were
    actually sent to the service.
    """

    def __init__(self, inner: ASDProvider, cache: Cache) -> None:
        self._inner = inner
        self._cache = cache
        self.name = inner.name
        self.hits = 0
        self.misses = 0

    def health_check(self) -> None:
        self._inner.health_check()

    def cache_key(self, clip_path: Path, diarization: dict[str, Any]) -> str:
        diarization_hash = hashlib.sha256(
            json.dumps(diarization, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return make_key("asd", self.name, _file_sha256(clip_path), diarization_hash)

    def detect(
        self,
        clip_path: Path,
        audio_path: Path | None,
        diarization: dict[str, Any],
        clip_start_frame: int,
        fps: float,
    ) -> ASDResult:
        key = self.cache_key(clip_path, diarization)
        cached = self._cache.get(key)
        if isinstance(cached, list):
            try:
                frames = frames_from_payload(cached)
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("asd_cache.unreadable", key=key, error=type(exc).__name__)
            else:
                self.hits += 1
                log.info("asd_cache.hit", key=key, frames=len(frames))
                return ASDResult(
                    frames=frames,
                    clip_start_frame=clip_start_frame,
                    fps=fps,
                    provider=self.name,
                )

        result = self._inner.detect(clip_path, audio_path, diarization, clip_start_frame, fps)
        self.misses += 1
        self._cache.put(key, frames_to_payload(result.frames))
        return result
