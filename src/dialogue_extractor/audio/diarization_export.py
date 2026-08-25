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

from dataclasses import dataclass
from typing import Any

from dialogue_extractor.config.constants import UNASSIGNED_SPEAKER_ID
from dialogue_extractor.domain.models import Word

SAMPLE_FORMAT = "sample"
RIVA_FORMAT = "riva"

UNASSIGNED_CODE = 0
"""Integer this codec reserves for words diarization could not attribute.

Note this is *our* convention on the way out, not the service's on the way back.
The service signals "no association" with -1; zero is a perfectly ordinary
speaker id to it. The two do not clash because this codec numbers real speakers
from 1, so a returned 0 can only be our unattributed bucket -- and `decode`
maps it to None because it was never registered as a label.
"""


@dataclass(frozen=True, slots=True)
class SpeakerCodec:
    r"""Bidirectional map between our speaker labels and the service's integers.

    Needed because the wire protocol is not what the JSON example suggests. The
    documented payload shows `"speaker_id": "speaker_0"`, but the proto field is
    an **int32**, and the service's parser recovers it by running `\d+` over the
    string. That has three consequences:

      * A label with no digits raises. Our own unassigned placeholder,
        "UNKNOWN", is exactly that -- and real output routinely contains
        unattributed words, so every such request would fail.
      * A label like "HOLMES" fails the same way, so any diarizer that names
        speakers rather than numbering them is unusable.
      * Responses come back carrying the *integer*, so comparing one directly
        against "SPEAKER_03" can never match -- and that comparison is the
        on-screen/off-screen check the whole system rests on.

    So labels are assigned integers explicitly here, emitted in a form the
    parser recovers exactly, and mapped back when the response arrives.
    """

    to_code: dict[str, int]
    to_label: dict[int, str]

    @classmethod
    def from_words(cls, words: tuple[Word, ...]) -> SpeakerCodec:
        """Build a codec covering every speaker present, in first-seen order.

        First-seen rather than sorted, so the numbering is stable for a given
        input regardless of how the labels happen to sort.
        """
        to_code: dict[str, int] = {}
        to_label: dict[int, str] = {}
        next_code = UNASSIGNED_CODE + 1

        for word in words:
            label = word.speaker_id
            if label is None or label in to_code:
                continue
            to_code[label] = next_code
            to_label[next_code] = label
            next_code += 1

        return cls(to_code=to_code, to_label=to_label)

    def encode(self, label: str | None) -> int:
        """Label to wire integer. Unknown labels fall back to the reserved code."""
        if label is None:
            return UNASSIGNED_CODE
        return self.to_code.get(label, UNASSIGNED_CODE)

    def decode(self, code: int) -> str | None:
        """Wire integer back to our label, or None if it was unattributed."""
        return self.to_label.get(code)

    def wire_label(self, label: str | None) -> str:
        r"""The string form emitted in the JSON payload.

        `speaker_<n>` keeps the documented string shape while guaranteeing the
        service's `\d+` extraction recovers exactly the integer we assigned.
        """
        return f"speaker_{self.encode(label)}"


def to_nvidia_diarization(
    words: tuple[Word, ...],
    *,
    language_code: str = "en-US",
    clip_start: float = 0.0,
    unassigned_speaker: str = UNASSIGNED_SPEAKER_ID,
    codec: SpeakerCodec | None = None,
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
                "speaker_id": (
                    codec.wire_label(word.speaker_id)
                    if codec is not None
                    else (word.speaker_id or unassigned_speaker)
                ),
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
