"""Diarization caching, mirroring CachedTranscriber.

Diarizing a feature-length file is minutes of GPU time. Without this, every
change to a later stage pays that cost again before reaching the code being
worked on.
"""

from __future__ import annotations

from pathlib import Path

from dialogue_extractor.audio.pyannote_diarizer import turns_from_dict, turns_to_dict
from dialogue_extractor.config.constants import MIN_SPEAKER_TURN_SECONDS
from dialogue_extractor.domain.models import SpeakerTurn
from dialogue_extractor.infra.disk_cache import make_key
from dialogue_extractor.infra.logging import get_logger
from dialogue_extractor.ingestion.normalizer import sha256_of
from dialogue_extractor.interfaces.ports import Cache, Diarizer

log = get_logger(__name__)


class CachedDiarizer(Diarizer):
    """Wraps a Diarizer with a content-addressed disk cache."""

    def __init__(self, inner: Diarizer, cache: Cache, *, fingerprint: str = "") -> None:
        self._inner = inner
        self._cache = cache
        self._fingerprint = fingerprint

    def cache_key(self, audio_path: Path) -> str:
        return make_key("diarization", sha256_of(audio_path), self._fingerprint)

    def diarize(self, audio_path: Path) -> tuple[SpeakerTurn, ...]:
        key = self.cache_key(audio_path)

        cached = self._cache.get(key)
        if cached is not None:
            try:
                turns = turns_from_dict(cached)
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("cache.diarization_unreadable", key=key, error=type(exc).__name__)
            else:
                if turns:
                    log.info("diarization.cache_hit", key=key, turns=len(turns))
                    return turns

        turns = self._inner.diarize(audio_path)
        self._cache.put(key, turns_to_dict(turns))
        return turns


def pyannote_fingerprint(
    pipeline_name: str,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    min_turn_seconds: float = MIN_SPEAKER_TURN_SECONDS,
) -> str:
    """Config fingerprint for the cache key.

    Every parameter that changes the stored turns belongs here, or a config
    change is silently ignored on any machine that ran the stage before it.

    `min_turn_seconds` is included because the short-turn filter runs during
    extraction, so the cache stores *filtered* turns. Omitting it meant tuning
    the threshold had no effect on an existing cache -- the very fix it was
    added for kept reporting the old, unfiltered attribution.
    """
    return (
        f"pyannote|{pipeline_name}|n={num_speakers}|min={min_speakers}"
        f"|max={max_speakers}|min_turn={min_turn_seconds}"
    )
