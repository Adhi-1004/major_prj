"""Serializing aligned words into the diarization format the ASD service reads.

This is the contract between our audio stages and NVIDIA's Active Speaker
Detection NIM. It is small, and getting it wrong is expensive, because the
service does not reject a malformed payload -- it proceeds as though no
diarization were supplied and returns confidently wrong results.

The service accepts two schemas, and they are *not* interchangeable:

  sample format (what we emit)          RIVA ASR format (what we must not)
  ----------------------------          ---------------------------------
  {"words": [{                          {"results": [{"alternatives": [{
     "text":       str,                     "words": [{
     "start":      float SECONDS,              "word":      str,
     "end":        float SECONDS,              "startTime": int MILLISECONDS,
     "speaker_id": str                         "endTime":   int MILLISECONDS,
  }]}                                          "speakerTag": int
                                            }]}]}]}

Seconds versus milliseconds is the trap: a payload in the wrong unit is still
valid JSON and still parses, it just describes a video a thousand times longer.
Hence a dedicated module with its own tests, rather than an inline dict literal.
"""

from __future__ import annotations

from typing import Any

from dialogue_extractor.config.constants import UNASSIGNED_SPEAKER_ID
from dialogue_extractor.domain.models import Word

SAMPLE_FORMAT = "sample"
RIVA_FORMAT = "riva"


def to_nvidia_diarization(
    words: tuple[Word, ...],
    *,
    language_code: str = "en-US",
    clip_start: float = 0.0,
    unassigned_speaker: str = UNASSIGNED_SPEAKER_ID,
) -> dict[str, Any]:
    """Build the sample-format diarization payload.

    Args:
        clip_start: Seconds subtracted from every timestamp. The service sees
            only the trimmed candidate clip, so its timeline starts at zero --
            passing whole-video timestamps alongside a 30-second clip would put
            every word past the end of the media it was given.
        unassigned_speaker: Words in a diarization gap still need *some* id;
            omitting them would leave the audio looking silent to the service.

    Returns:
        A JSON-serializable dict. Timestamps are floats in **seconds**.
    """
    payload_words: list[dict[str, Any]] = []

    for word in words:
        start = round(word.start - clip_start, 3)
        end = round(word.end - clip_start, 3)
        # Trimming can clip a word that straddles the clip boundary.
        if end <= 0:
            continue
        payload_words.append(
            {
                "text": word.text,
                "start": max(0.0, start),
                "end": max(0.0, end),
                "speaker_id": word.speaker_id or unassigned_speaker,
            }
        )

    return {
        "language_code": language_code,
        "text": " ".join(w["text"] for w in payload_words),
        "words": payload_words,
    }


def validate_diarization_payload(payload: dict[str, Any]) -> None:
    """Assert the payload matches the sample-format contract.

    Called before every request. The service will not tell us if this is wrong,
    so the check has to happen on our side.

    Raises:
        ValueError: describing the first violation found.
    """
    if "words" not in payload:
        raise ValueError("diarization payload has no 'words' key")

    words = payload["words"]
    if not isinstance(words, list):
        raise ValueError(f"'words' must be a list, got {type(words).__name__}")
    if not words:
        raise ValueError("diarization payload contains no words")

    for index, word in enumerate(words):
        for field in ("text", "start", "end", "speaker_id"):
            if field not in word:
                raise ValueError(f"word {index} is missing {field!r}")

        if not isinstance(word["speaker_id"], str):
            raise ValueError(
                f"word {index}: speaker_id must be a string "
                f"(the RIVA schema's integer speakerTag is a different format), "
                f"got {type(word['speaker_id']).__name__}"
            )

        for field in ("start", "end"):
            value = word[field]
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"word {index}: {field} must be a number, got {value!r}")
            if value < 0:
                raise ValueError(f"word {index}: {field} is negative ({value})")

        if word["end"] < word["start"]:
            raise ValueError(f"word {index}: end {word['end']} precedes start {word['start']}")


def looks_like_milliseconds(payload: dict[str, Any], clip_duration: float) -> bool:
    """Heuristic guard against the seconds/milliseconds mix-up.

    A payload whose last timestamp overshoots the clip by orders of magnitude is
    almost certainly in the wrong unit. Cheap, and it catches the one mistake
    that would otherwise pass silently all the way to a wrong answer.
    """
    words = payload.get("words") or []
    if not words or clip_duration <= 0:
        return False
    last_end = max(float(w["end"]) for w in words)
    return last_end > clip_duration * 10
