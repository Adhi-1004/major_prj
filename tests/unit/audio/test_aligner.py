"""Word-to-speaker alignment and segment grouping.

Transcription and diarization produce two independent timelines. Every way they
can disagree needs a defined, deterministic answer -- this is where that is
decided, and it is pure, so it can be pinned exhaustively.
"""

from __future__ import annotations

import pytest

from dialogue_extractor.audio.aligner import (
    assign_speakers,
    build_segments,
    speaker_statistics,
)
from dialogue_extractor.domain.models import SpeakerTurn, Word


def w(text: str, start: float, end: float, speaker: str | None = None) -> Word:
    return Word(text=text, start=start, end=end, speaker_id=speaker)


def turn(speaker: str, start: float, end: float) -> SpeakerTurn:
    return SpeakerTurn(speaker_id=speaker, start=start, end=end)


class TestAssignSpeakers:
    def test_word_inside_a_turn(self) -> None:
        words = (w("hello", 1.0, 1.5),)
        turns = (turn("SPEAKER_00", 0.0, 5.0),)
        assert assign_speakers(words, turns)[0].speaker_id == "SPEAKER_00"

    def test_word_straddling_two_turns_goes_to_the_greater_overlap(self) -> None:
        """0.8s inside SPEAKER_01 vs 0.2s inside SPEAKER_00."""
        words = (w("hello", 4.8, 5.8),)
        turns = (turn("SPEAKER_00", 0.0, 5.0), turn("SPEAKER_01", 5.0, 10.0))
        assert assign_speakers(words, turns)[0].speaker_id == "SPEAKER_01"

    def test_straddling_the_other_way(self) -> None:
        words = (w("hello", 4.2, 5.2),)
        turns = (turn("SPEAKER_00", 0.0, 5.0), turn("SPEAKER_01", 5.0, 10.0))
        assert assign_speakers(words, turns)[0].speaker_id == "SPEAKER_00"

    def test_word_in_a_gap_is_kept_with_no_speaker(self) -> None:
        """Dropping it would punch a hole in the stream Phase 4 matches over,
        making a phrase that spans the gap unfindable."""
        words = (w("hello", 6.0, 6.5),)
        turns = (turn("SPEAKER_00", 0.0, 5.0), turn("SPEAKER_01", 8.0, 10.0))

        result = assign_speakers(words, turns)
        assert len(result) == 1
        assert result[0].speaker_id is None
        assert result[0].text == "hello"

    def test_no_words_are_ever_lost(self) -> None:
        words = tuple(w(f"w{i}", float(i), float(i) + 0.5) for i in range(20))
        turns = (turn("SPEAKER_00", 0.0, 3.0), turn("SPEAKER_01", 15.0, 18.0))
        assert len(assign_speakers(words, turns)) == 20

    def test_exact_tie_is_deterministic(self) -> None:
        """Equal overlap on both sides must not depend on turn ordering."""
        words = (w("hello", 4.5, 5.5),)
        forward = (turn("SPEAKER_00", 0.0, 5.0), turn("SPEAKER_01", 5.0, 10.0))
        reversed_order = (turn("SPEAKER_01", 5.0, 10.0), turn("SPEAKER_00", 0.0, 5.0))

        assert (
            assign_speakers(words, forward)[0].speaker_id
            == assign_speakers(words, reversed_order)[0].speaker_id
        )

    def test_tie_breaks_toward_the_earlier_turn(self) -> None:
        words = (w("hello", 4.5, 5.5),)
        turns = (turn("SPEAKER_00", 0.0, 5.0), turn("SPEAKER_01", 5.0, 10.0))
        assert assign_speakers(words, turns)[0].speaker_id == "SPEAKER_00"

    def test_zero_length_word_uses_containment(self) -> None:
        """A zero-length word overlaps nothing, but still sits inside a turn."""
        words = (w("hmm", 3.0, 3.0),)
        turns = (turn("SPEAKER_00", 0.0, 5.0),)
        assert assign_speakers(words, turns)[0].speaker_id == "SPEAKER_00"

    def test_zero_length_word_in_a_gap_is_unassigned(self) -> None:
        words = (w("hmm", 6.0, 6.0),)
        turns = (turn("SPEAKER_00", 0.0, 5.0),)
        assert assign_speakers(words, turns)[0].speaker_id is None

    def test_no_turns_leaves_words_untouched(self) -> None:
        """Diarization failing must not destroy the transcript."""
        words = (w("hello", 1.0, 1.5), w("world", 1.5, 2.0))
        result = assign_speakers(words, ())
        assert result == words

    def test_no_words_returns_empty(self) -> None:
        assert assign_speakers((), (turn("SPEAKER_00", 0.0, 5.0),)) == ()

    def test_overlapping_turns_pick_the_dominant_speaker(self) -> None:
        """Diarization can emit overlapping turns for crosstalk."""
        words = (w("hello", 4.0, 5.0),)
        turns = (turn("SPEAKER_00", 0.0, 4.2), turn("SPEAKER_01", 3.9, 8.0))
        assert assign_speakers(words, turns)[0].speaker_id == "SPEAKER_01"

    def test_preserves_all_other_word_fields(self) -> None:
        words = (Word("hello", 1.0, 1.5, probability=0.87),)
        result = assign_speakers(words, (turn("SPEAKER_00", 0.0, 5.0),))
        assert result[0].text == "hello"
        assert result[0].probability == pytest.approx(0.87)
        assert result[0].start == 1.0
        assert result[0].end == 1.5

    def test_unsorted_turns_are_handled(self) -> None:
        words = (w("hello", 7.0, 7.5),)
        turns = (turn("SPEAKER_01", 5.0, 10.0), turn("SPEAKER_00", 0.0, 5.0))
        assert assign_speakers(words, turns)[0].speaker_id == "SPEAKER_01"

    def test_realistic_two_speaker_conversation(self) -> None:
        words = (
            w("Good", 0.5, 0.8), w("morning", 0.8, 1.3),
            w("Good", 2.5, 2.8), w("morning", 2.8, 3.3), w("Holmes", 3.3, 3.9),
            w("Watson", 5.0, 5.6),
        )
        turns = (
            turn("SPEAKER_00", 0.0, 2.0),
            turn("SPEAKER_01", 2.2, 4.2),
            turn("SPEAKER_00", 4.8, 6.0),
        )
        result = assign_speakers(words, turns)
        assert [x.speaker_id for x in result] == [
            "SPEAKER_00", "SPEAKER_00",
            "SPEAKER_01", "SPEAKER_01", "SPEAKER_01",
            "SPEAKER_00",
        ]


class TestBuildSegments:
    def test_speaker_change_splits(self) -> None:
        words = (
            w("hello", 0.0, 0.5, "SPEAKER_00"),
            w("there", 0.5, 1.0, "SPEAKER_00"),
            w("hi", 1.1, 1.5, "SPEAKER_01"),
        )
        segments = build_segments(words)
        assert len(segments) == 2
        assert segments[0].text == "hello there"
        assert segments[1].text == "hi"

    def test_short_gap_does_not_split(self) -> None:
        words = (
            w("hello", 0.0, 0.5, "SPEAKER_00"),
            w("there", 0.9, 1.4, "SPEAKER_00"),  # 0.4s gap, under the threshold
        )
        assert len(build_segments(words, max_gap=0.75)) == 1

    def test_long_gap_splits_even_for_one_speaker(self) -> None:
        """One speaker's turn can span a long pause; two lines minutes apart
        would otherwise merge into a segment covering footage with neither."""
        words = (
            w("hello", 0.0, 0.5, "SPEAKER_00"),
            w("there", 5.0, 5.5, "SPEAKER_00"),
        )
        segments = build_segments(words, max_gap=0.75)
        assert len(segments) == 2

    def test_gap_exactly_at_threshold_does_not_split(self) -> None:
        words = (
            w("a", 0.0, 0.5, "SPEAKER_00"),
            w("b", 1.25, 1.5, "SPEAKER_00"),  # exactly 0.75
        )
        assert len(build_segments(words, max_gap=0.75)) == 1

    def test_unassigned_words_form_their_own_segment(self) -> None:
        words = (
            w("hello", 0.0, 0.5, "SPEAKER_00"),
            w("noise", 0.6, 0.9, None),
            w("there", 1.0, 1.4, "SPEAKER_00"),
        )
        segments = build_segments(words)
        assert [s.speaker_id for s in segments] == ["SPEAKER_00", None, "SPEAKER_00"]

    def test_empty_input(self) -> None:
        assert build_segments(()) == ()

    def test_single_word(self) -> None:
        segments = build_segments((w("hello", 0.0, 0.5, "SPEAKER_00"),))
        assert len(segments) == 1
        assert segments[0].text == "hello"

    def test_segment_bounds_come_from_its_words(self) -> None:
        words = (
            w("hello", 1.0, 1.5, "SPEAKER_00"),
            w("there", 1.5, 2.25, "SPEAKER_00"),
        )
        segment = build_segments(words)[0]
        assert segment.start == 1.0
        assert segment.end == 2.25

    def test_every_word_lands_in_exactly_one_segment(self) -> None:
        words = tuple(
            w(f"w{i}", i * 0.5, i * 0.5 + 0.4, f"SPEAKER_0{i % 3}") for i in range(30)
        )
        segments = build_segments(words)
        total = sum(len(s.words) for s in segments)
        assert total == len(words)

    def test_segments_are_in_chronological_order(self) -> None:
        words = tuple(
            w(f"w{i}", i * 2.0, i * 2.0 + 0.4, f"SPEAKER_0{i % 2}") for i in range(10)
        )
        segments = build_segments(words)
        assert all(
            segments[i].start <= segments[i + 1].start for i in range(len(segments) - 1)
        )


class TestSpeakerStatistics:
    def test_counts_per_speaker(self) -> None:
        words = (
            w("a", 0.0, 0.1, "SPEAKER_00"),
            w("b", 0.1, 0.2, "SPEAKER_00"),
            w("c", 0.2, 0.3, "SPEAKER_01"),
        )
        assert speaker_statistics(words) == {"SPEAKER_00": 2, "SPEAKER_01": 1}

    def test_unassigned_words_are_counted_separately(self) -> None:
        words = (w("a", 0.0, 0.1, "SPEAKER_00"), w("b", 0.1, 0.2, None))
        assert speaker_statistics(words) == {"SPEAKER_00": 1, "": 1}

    def test_empty(self) -> None:
        assert speaker_statistics(()) == {}
