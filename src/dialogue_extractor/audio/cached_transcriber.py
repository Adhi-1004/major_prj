"""Transcript caching, as a decorator over any Transcriber.

Kept separate from WhisperTranscriber rather than built into it: caching is
orthogonal to transcribing, and a decorator means the cache can wrap any future
backend without either one knowing about the other. The pipeline sees only the
Transcriber port and cannot tell the difference.

Keyed by the audio file's content hash plus every decoding parameter that
affects the result. A transcript produced by `large-v3` must never be served to
a run configured for `medium` -- same audio, different meaning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dialogue_extractor.config import constants
from dialogue_extractor.domain.models import Transcript, Word
from dialogue_extractor.infra.disk_cache import make_key
from dialogue_extractor.infra.logging import get_logger
from dialogue_extractor.ingestion.normalizer import sha256_of
from dialogue_extractor.interfaces.ports import Cache, Transcriber

log = get_logger(__name__)


def transcript_to_dict(transcript: Transcript) -> dict[str, Any]:
    return {
        "language": transcript.language,
        "duration": transcript.duration,
        "words": [
            {
                "text": w.text,
                "start": w.start,
                "end": w.end,
                "probability": w.probability,
                "speaker_id": w.speaker_id,
            }
            for w in transcript.words
        ],
    }


def transcript_from_dict(payload: dict[str, Any]) -> Transcript:
    """Rebuild a Transcript from cached JSON.

    Raises (Key/Type/Value)Error on a malformed payload; the caller treats that
    as a cache miss rather than a failure.
    """
    words = tuple(
        Word(
            text=str(w["text"]),
            start=float(w["start"]),
            end=float(w["end"]),
            probability=float(w.get("probability", 1.0)),
            speaker_id=w.get("speaker_id"),
        )
        for w in payload["words"]
    )
    return Transcript(
        words=words,
        language=str(payload["language"]),
        duration=float(payload["duration"]),
    )


class CachedTranscriber(Transcriber):
    """Wraps a Transcriber with a content-addressed disk cache."""

    def __init__(self, inner: Transcriber, cache: Cache, *, fingerprint: str = "") -> None:
        """
        Args:
            fingerprint: Everything about the decoding configuration that
                changes the output -- model size, language, beam size. Part of
                the key, so a config change cannot silently reuse a stale
                transcript.
        """
        self._inner = inner
        self._cache = cache
        self._fingerprint = fingerprint

    def cache_key(self, audio_path: Path) -> str:
        return make_key("transcript", sha256_of(audio_path), self._fingerprint)

    def transcribe(self, audio_path: Path) -> Transcript:
        key = self.cache_key(audio_path)

        cached = self._cache.get(key)
        if cached is not None:
            try:
                transcript = transcript_from_dict(cached)
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("cache.transcript_unreadable", key=key, error=type(exc).__name__)
            else:
                log.info(
                    "transcript.cache_hit", key=key, words=len(transcript.words)
                )
                return transcript

        transcript = self._inner.transcribe(audio_path)
        self._cache.put(key, transcript_to_dict(transcript))
        return transcript


def whisper_fingerprint(
    model_size: str,
    language: str | None,
    beam_size: int,
    compute_type: str = "",
) -> str:
    """Config fingerprint for the cache key.

    `compute_type` is included because quantization changes the output: the same
    audio through the same model gave 325 words at int8 on CPU and 337 at
    int8_float16 on GPU. Serving one run's transcript to the other would quietly
    break the reproducibility the pinned decoding parameters exist to provide.

    The decoding constants are included for the same reason: they are pinned
    today, but if one is ever changed, an existing cache must not keep serving
    transcripts produced under the old setting.
    """
    parts = (
        f"whisper|{model_size}|{language or 'auto'}|beam={beam_size}"
        f"|temp={constants.WHISPER_TEMPERATURE}"
        f"|cond={constants.WHISPER_CONDITION_ON_PREVIOUS_TEXT}"
    )
    return f"{parts}|{compute_type}" if compute_type else parts
