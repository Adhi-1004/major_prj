"""The diarization contract with NVIDIA's Active Speaker Detection NIM.

The design document's edge-case table commits to exactly this test:

    "Diarization JSON unit mismatch (seconds vs. milliseconds) | Confirmed
     against the exact parser in use before relying on it; unit tested directly"

It earns its place because the service does not reject a malformed payload. It
treats the session as having no diarization at all and returns confidently
wrong results -- so every violation this catches would otherwise surface as a
plausible-looking wrong frame, not as an error.
"""

from __future__ import annotations

import json

import pytest

from dialogue_extractor.audio.diarization_export import (
    looks_like_milliseconds,
    to_nvidia_diarization,
    validate_diarization_payload,
)
from dialogue_extractor.domain.models import Word


def w(text: str, start: float, end: float, speaker: str | None = "SPEAKER_00") -> Word:
    return Word(text=text, start=start, end=end, speaker_id=speaker)


SAMPLE_WORDS = (
    w("My", 10.0, 10.2),
    w("mind", 10.2, 10.5),
    w("rebels", 10.5, 10.9),
    w("at", 10.9, 11.0),
    w("stagnation", 11.0, 11.6),
)


class TestSchemaShape:
    def test_top_level_keys(self) -> None:
        payload = to_nvidia_diarization(SAMPLE_WORDS)
        assert "words" in payload
        assert "language_code" in payload
        assert "text" in payload

    def test_word_field_names_match_the_contract_exactly(self) -> None:
        """text/start/end/speaker_id -- not word/startTime/endTime/speakerTag,
        which is the RIVA schema and a different format entirely."""
        word = to_nvidia_diarization(SAMPLE_WORDS)["words"][0]
        assert set(word) == {"text", "start", "end", "speaker_id"}

    def test_does_not_emit_riva_field_names(self) -> None:
        word = to_nvidia_diarization(SAMPLE_WORDS)["words"][0]
        for riva_field in ("word", "startTime", "endTime", "speakerTag", "languageCode"):
            assert riva_field not in word

    def test_is_json_serializable(self) -> None:
        payload = to_nvidia_diarization(SAMPLE_WORDS)
        assert json.loads(json.dumps(payload))["words"][0]["text"] == "My"

    def test_word_order_survives_a_json_round_trip(self) -> None:
        payload = json.loads(json.dumps(to_nvidia_diarization(SAMPLE_WORDS)))
        assert [x["text"] for x in payload["words"]] == [
            "My", "mind", "rebels", "at", "stagnation"
        ]


class TestUnits:
    """Seconds, not milliseconds. The single most consequential detail here."""

    def test_timestamps_are_seconds(self) -> None:
        payload = to_nvidia_diarization(SAMPLE_WORDS)
        assert payload["words"][0]["start"] == pytest.approx(10.0)
        assert payload["words"][0]["end"] == pytest.approx(10.2)

    def test_timestamps_are_not_scaled_to_milliseconds(self) -> None:
        payload = to_nvidia_diarization(SAMPLE_WORDS)
        assert payload["words"][0]["start"] != pytest.approx(10_000)

    def test_timestamps_are_floats_not_ints(self) -> None:
        """The RIVA schema uses integer milliseconds; ints here would be a
        symptom of having built the wrong payload."""
        payload = to_nvidia_diarization(SAMPLE_WORDS)
        assert isinstance(payload["words"][0]["start"], float)

    def test_sub_second_precision_is_preserved(self) -> None:
        payload = to_nvidia_diarization((w("x", 1.234, 1.567),))
        assert payload["words"][0]["start"] == pytest.approx(1.234)
        assert payload["words"][0]["end"] == pytest.approx(1.567)


class TestSpeakerIds:
    def test_speaker_id_is_a_string(self) -> None:
        """RIVA's speakerTag is an integer; a number here is the wrong schema."""
        payload = to_nvidia_diarization(SAMPLE_WORDS)
        assert isinstance(payload["words"][0]["speaker_id"], str)

    def test_unassigned_words_get_a_placeholder_not_null(self) -> None:
        """Omitting them would make that audio look silent to the service."""
        payload = to_nvidia_diarization((w("noise", 1.0, 1.2, None),))
        assert payload["words"][0]["speaker_id"] == "UNKNOWN"

    def test_placeholder_is_configurable(self) -> None:
        payload = to_nvidia_diarization(
            (w("noise", 1.0, 1.2, None),), unassigned_speaker="SPK_NA"
        )
        assert payload["words"][0]["speaker_id"] == "SPK_NA"

    def test_multiple_speakers_are_preserved(self) -> None:
        words = (w("a", 0.0, 0.5, "SPEAKER_00"), w("b", 1.0, 1.5, "SPEAKER_01"))
        ids = [x["speaker_id"] for x in to_nvidia_diarization(words)["words"]]
        assert ids == ["SPEAKER_00", "SPEAKER_01"]


class TestClipOffset:
    """The service sees only the trimmed clip, whose timeline starts at zero."""

    def test_offsets_are_subtracted(self) -> None:
        payload = to_nvidia_diarization(SAMPLE_WORDS, clip_start=9.0)
        assert payload["words"][0]["start"] == pytest.approx(1.0)
        assert payload["words"][0]["end"] == pytest.approx(1.2)

    def test_whole_video_timestamps_would_fall_outside_a_short_clip(self) -> None:
        """Without the offset every word lands past the end of the media sent."""
        without = to_nvidia_diarization(SAMPLE_WORDS, clip_start=0.0)
        with_offset = to_nvidia_diarization(SAMPLE_WORDS, clip_start=9.0)
        assert without["words"][0]["start"] > 5.0
        assert with_offset["words"][0]["start"] < 5.0

    def test_never_emits_negative_timestamps(self) -> None:
        payload = to_nvidia_diarization((w("x", 10.0, 10.5),), clip_start=10.2)
        assert payload["words"][0]["start"] >= 0.0

    def test_words_entirely_before_the_clip_are_dropped(self) -> None:
        words = (w("before", 1.0, 2.0), w("inside", 12.0, 12.5))
        payload = to_nvidia_diarization(words, clip_start=10.0)
        assert [x["text"] for x in payload["words"]] == ["inside"]

    def test_zero_offset_is_identity(self) -> None:
        assert (
            to_nvidia_diarization(SAMPLE_WORDS, clip_start=0.0)
            == to_nvidia_diarization(SAMPLE_WORDS)
        )


class TestValidation:
    def test_accepts_a_well_formed_payload(self) -> None:
        validate_diarization_payload(to_nvidia_diarization(SAMPLE_WORDS))

    def test_rejects_missing_words_key(self) -> None:
        with pytest.raises(ValueError, match="no 'words' key"):
            validate_diarization_payload({"language_code": "en-US"})

    def test_rejects_empty_word_list(self) -> None:
        with pytest.raises(ValueError, match="contains no words"):
            validate_diarization_payload({"words": []})

    def test_rejects_non_list_words(self) -> None:
        with pytest.raises(ValueError, match="must be a list"):
            validate_diarization_payload({"words": "nope"})

    @pytest.mark.parametrize("missing", ["text", "start", "end", "speaker_id"])
    def test_rejects_missing_fields(self, missing: str) -> None:
        word = {"text": "x", "start": 0.0, "end": 1.0, "speaker_id": "S0"}
        del word[missing]
        with pytest.raises(ValueError, match=missing):
            validate_diarization_payload({"words": [word]})

    def test_rejects_integer_speaker_tag(self) -> None:
        """Catches a RIVA-format payload passed where sample format is expected."""
        payload = {"words": [{"text": "x", "start": 0.0, "end": 1.0, "speaker_id": 0}]}
        with pytest.raises(ValueError, match="must be a string"):
            validate_diarization_payload(payload)

    def test_rejects_negative_timestamps(self) -> None:
        payload = {"words": [{"text": "x", "start": -1.0, "end": 1.0, "speaker_id": "S0"}]}
        with pytest.raises(ValueError, match="negative"):
            validate_diarization_payload(payload)

    def test_rejects_inverted_timestamps(self) -> None:
        payload = {"words": [{"text": "x", "start": 5.0, "end": 1.0, "speaker_id": "S0"}]}
        with pytest.raises(ValueError, match="precedes start"):
            validate_diarization_payload(payload)

    def test_rejects_non_numeric_timestamps(self) -> None:
        payload = {"words": [{"text": "x", "start": "0.0", "end": 1.0, "speaker_id": "S0"}]}
        with pytest.raises(ValueError, match="must be a number"):
            validate_diarization_payload(payload)

    def test_rejects_boolean_timestamps(self) -> None:
        """bool is a subclass of int; True must not pass as a timestamp."""
        payload = {"words": [{"text": "x", "start": True, "end": 1.0, "speaker_id": "S0"}]}
        with pytest.raises(ValueError, match="must be a number"):
            validate_diarization_payload(payload)

    def test_error_names_the_offending_word(self) -> None:
        words = [
            {"text": "ok", "start": 0.0, "end": 1.0, "speaker_id": "S0"},
            {"text": "bad", "start": 5.0, "end": 1.0, "speaker_id": "S0"},
        ]
        with pytest.raises(ValueError, match="word 1"):
            validate_diarization_payload({"words": words})


class TestMillisecondGuard:
    def test_detects_a_millisecond_payload(self) -> None:
        """The mistake that stays valid JSON and describes a video 1000x longer."""
        payload = {
            "words": [{"text": "x", "start": 10000.0, "end": 10200.0, "speaker_id": "S0"}]
        }
        assert looks_like_milliseconds(payload, clip_duration=30.0) is True

    def test_accepts_a_correct_seconds_payload(self) -> None:
        payload = to_nvidia_diarization(SAMPLE_WORDS, clip_start=9.0)
        assert looks_like_milliseconds(payload, clip_duration=30.0) is False

    def test_tolerates_slight_overshoot(self) -> None:
        """A word running just past the clip edge is normal, not a unit error."""
        payload = {"words": [{"text": "x", "start": 29.0, "end": 31.0, "speaker_id": "S0"}]}
        assert looks_like_milliseconds(payload, clip_duration=30.0) is False

    def test_no_conclusion_without_a_duration(self) -> None:
        payload = to_nvidia_diarization(SAMPLE_WORDS)
        assert looks_like_milliseconds(payload, clip_duration=0.0) is False

    def test_empty_payload(self) -> None:
        assert looks_like_milliseconds({"words": []}, clip_duration=30.0) is False


class TestEndToEndShape:
    def test_realistic_payload_validates_and_passes_the_unit_guard(self) -> None:
        payload = to_nvidia_diarization(SAMPLE_WORDS, clip_start=8.5)
        validate_diarization_payload(payload)
        assert looks_like_milliseconds(payload, clip_duration=30.0) is False
        assert payload["text"] == "My mind rebels at stagnation"
