r"""Our diarization payload, fed to NVIDIA's own parser.

Every other test of this payload asserts against a contract *we* wrote down from
the documentation. This one asserts against the code that will actually consume
it, which is the only way to catch the places the documentation is wrong -- and
it is wrong in two consequential places:

  * The documented JSON shows `"speaker_id": "speaker_0"`, but the proto field
    is an int32 that the parser recovers by running `\\d+` over the string. A
    label with no digits raises, and our unassigned placeholder "UNKNOWN" is
    exactly that.
  * The documented JSON is in seconds, but the proto carries uint32
    milliseconds. The parser multiplies by 1000, so emitting milliseconds would
    describe a clip a thousand times too long -- silently, since both are valid.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from dialogue_extractor.audio.diarization_export import (
    SpeakerCodec,
    to_nvidia_diarization,
    validate_diarization_payload,
)
from dialogue_extractor.domain.models import Word

VENDOR_ROOT = Path("src/dialogue_extractor/vision/providers/nim")

pytestmark = pytest.mark.skipif(
    not (VENDOR_ROOT / "diarization.py").exists(),
    reason="vendored NVIDIA client not present",
)


@pytest.fixture(scope="module")
def nvidia_parser():  # type: ignore[no-untyped-def]
    """NVIDIA's real SampleDiarizationParser, loaded from the vendored client."""
    for path in (VENDOR_ROOT / "interfaces", VENDOR_ROOT):
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)

    from diarization import SampleDiarizationParser  # type: ignore[import-not-found]

    return SampleDiarizationParser()


def words_with_speakers() -> tuple[Word, ...]:
    """Realistic output: several speakers plus unattributed words.

    The unattributed ones are the point -- a real run of this pipeline produced
    15 of them out of 313.
    """
    return (
        Word("You", 38.91, 39.09, speaker_id="SPEAKER_04"),
        Word("know", 39.09, 39.30, speaker_id="SPEAKER_04"),
        Word("Tell", 45.53, 45.93, speaker_id=None),  # unattributed
        Word("me", 45.93, 46.10, speaker_id="SPEAKER_04"),
        Word("Yes", 47.00, 47.30, speaker_id="SPEAKER_03"),
    )


class TestRealParserAcceptsOurPayload:
    def test_parses_without_error(self, nvidia_parser) -> None:  # type: ignore[no-untyped-def]
        words = words_with_speakers()
        codec = SpeakerCodec.from_words(words)
        payload = to_nvidia_diarization(words, clip_start=38.0, codec=codec)

        segments, transcript = nvidia_parser.parse(json.dumps(payload))
        assert len(segments) == len(words)

    def test_can_parse_recognizes_the_shape(self, nvidia_parser) -> None:  # type: ignore[no-untyped-def]
        words = words_with_speakers()
        payload = to_nvidia_diarization(words, codec=SpeakerCodec.from_words(words))
        assert nvidia_parser.can_parse(json.dumps(payload)) is True

    def test_unattributed_words_do_not_crash_the_parser(
        self, nvidia_parser
    ) -> None:  # type: ignore[no-untyped-def]
        """Without the codec these become "UNKNOWN", which has no digits, and
        the parser raises. Real output always contains such words."""
        words = (Word("Tell", 45.53, 45.93, speaker_id=None),)
        payload = to_nvidia_diarization(words, codec=SpeakerCodec.from_words(words))

        segments, _ = nvidia_parser.parse(json.dumps(payload))
        assert len(segments) == 1

    def test_the_old_placeholder_really_would_have_crashed(
        self, nvidia_parser
    ) -> None:  # type: ignore[no-untyped-def]
        """Pins the failure this codec exists to prevent."""
        words = (Word("Tell", 45.53, 45.93, speaker_id=None),)
        payload = to_nvidia_diarization(words)  # no codec -> "UNKNOWN"

        assert payload["words"][0]["speaker_id"] == "UNKNOWN"
        with pytest.raises(ValueError, match="Cannot extract speaker id"):
            nvidia_parser.parse(json.dumps(payload))

    def test_named_speakers_survive(self, nvidia_parser) -> None:  # type: ignore[no-untyped-def]
        """A diarizer that names speakers rather than numbering them must work."""
        words = (
            Word("Elementary", 1.0, 1.8, speaker_id="HOLMES"),
            Word("Indeed", 2.0, 2.5, speaker_id="WATSON"),
        )
        payload = to_nvidia_diarization(words, codec=SpeakerCodec.from_words(words))

        segments, _ = nvidia_parser.parse(json.dumps(payload))
        assert len(segments) == 2
        assert segments[0].speaker_id != segments[1].speaker_id


class TestUnitsThroughTheRealParser:
    def test_seconds_become_milliseconds(self, nvidia_parser) -> None:  # type: ignore[no-untyped-def]
        """The parser multiplies by 1000. Emitting milliseconds would describe a
        clip a thousand times too long, and both are valid JSON."""
        words = (Word("hello", 10.0, 10.5, speaker_id="SPEAKER_00"),)
        payload = to_nvidia_diarization(words, codec=SpeakerCodec.from_words(words))

        segments, _ = nvidia_parser.parse(json.dumps(payload))
        assert segments[0].start_time == 10_000
        assert segments[0].end_time == 10_500

    def test_clip_offset_lands_near_zero(self, nvidia_parser) -> None:  # type: ignore[no-untyped-def]
        words = (Word("hello", 100.0, 100.5, speaker_id="SPEAKER_00"),)
        payload = to_nvidia_diarization(
            words, clip_start=99.5, codec=SpeakerCodec.from_words(words)
        )

        segments, _ = nvidia_parser.parse(json.dumps(payload))
        assert segments[0].start_time == 500  # 0.5s into the clip

    def test_sub_millisecond_precision_rounds_sanely(
        self, nvidia_parser
    ) -> None:  # type: ignore[no-untyped-def]
        words = (Word("hi", 1.2345, 1.6789, speaker_id="SPEAKER_00"),)
        payload = to_nvidia_diarization(words, codec=SpeakerCodec.from_words(words))

        segments, _ = nvidia_parser.parse(json.dumps(payload))
        assert segments[0].start_time == pytest.approx(1234, abs=1)
        assert segments[0].end_time == pytest.approx(1679, abs=1)


class TestSpeakerCodecRoundTrip:
    def test_labels_survive_the_wire_format(self, nvidia_parser) -> None:  # type: ignore[no-untyped-def]
        """The response returns integers, so decoding them back to our labels is
        what makes the on-screen speaker cross-check possible at all."""
        words = words_with_speakers()
        codec = SpeakerCodec.from_words(words)
        payload = to_nvidia_diarization(words, codec=codec)

        segments, _ = nvidia_parser.parse(json.dumps(payload))

        for word, segment in zip(words, segments, strict=True):
            assert codec.decode(segment.speaker_id) == word.speaker_id

    def test_distinct_speakers_get_distinct_codes(self) -> None:
        words = words_with_speakers()
        codec = SpeakerCodec.from_words(words)
        assert codec.encode("SPEAKER_04") != codec.encode("SPEAKER_03")

    def test_unattributed_decodes_to_none(self) -> None:
        codec = SpeakerCodec.from_words(words_with_speakers())
        assert codec.decode(codec.encode(None)) is None

    def test_numbering_is_stable_for_the_same_input(self) -> None:
        words = words_with_speakers()
        assert SpeakerCodec.from_words(words).to_code == (
            SpeakerCodec.from_words(words).to_code
        )

    def test_unassigned_code_is_never_reused_for_a_real_speaker(self) -> None:
        codec = SpeakerCodec.from_words(words_with_speakers())
        assert codec.encode(None) not in codec.to_label


class TestPayloadStillValidatesLocally:
    def test_our_own_validator_agrees(self) -> None:
        words = words_with_speakers()
        payload = to_nvidia_diarization(words, codec=SpeakerCodec.from_words(words))
        validate_diarization_payload(payload)
