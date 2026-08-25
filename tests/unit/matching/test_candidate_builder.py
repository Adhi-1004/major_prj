"""Turning matches and segments into windows worth verifying.

Every candidate costs a re-encode and a network round-trip, so both too many and
too few are real failures: too many is slow and expensive, too few means the
answer was never looked for.
"""

from __future__ import annotations

import pytest

from dialogue_extractor.domain.enums import Mode
from dialogue_extractor.domain.models import Candidate, SpeechSegment, Word
from dialogue_extractor.matching.candidate_builder import (
    apply_budget,
    candidates_from_matches,
    candidates_from_segments,
)
from dialogue_extractor.matching.fuzzy_matcher import find_matches

DURATION = 600.0
TARGET = "My mind rebels at stagnation"


def phrase(text: str, start: float = 10.0, gap: float = 0.3, speaker: str | None = None):
    words = []
    t = start
    for token in text.split(" "):
        words.append(Word(text=token, start=t, end=t + gap * 0.8, speaker_id=speaker))
        t += gap
    return tuple(words)


def segment(text: str, start: float, gap: float = 0.3, speaker: str | None = "SPEAKER_00"):
    return SpeechSegment(words=phrase(text, start, gap, speaker), speaker_id=speaker)


class TestModeAPadding:
    def test_window_is_padded_on_both_sides(self) -> None:
        words = phrase("my mind rebels at stagnation", start=100.0)
        matches = find_matches(words, TARGET)
        candidate = candidates_from_matches(matches, DURATION, pad=1.5)[0]

        assert candidate.start == pytest.approx(matches[0].start - 1.5)
        assert candidate.end == pytest.approx(matches[0].end + 1.5)

    def test_clamps_at_the_start_of_the_video(self) -> None:
        """A line delivered in the opening seconds must not produce t < 0."""
        words = phrase("my mind rebels at stagnation", start=0.2)
        candidates = candidates_from_matches(find_matches(words, TARGET), DURATION, pad=1.5)
        assert candidates[0].start == 0.0

    def test_clamps_at_the_end_of_the_video(self) -> None:
        words = phrase("my mind rebels at stagnation", start=DURATION - 1.0)
        candidates = candidates_from_matches(find_matches(words, TARGET), DURATION, pad=1.5)
        assert candidates[0].end <= DURATION

    def test_zero_padding_is_the_bare_match(self) -> None:
        words = phrase("my mind rebels at stagnation", start=100.0)
        matches = find_matches(words, TARGET)
        candidate = candidates_from_matches(matches, DURATION, pad=0.0)[0]
        assert candidate.start == pytest.approx(matches[0].start)


class TestModeAMetadata:
    def test_mode_is_target(self) -> None:
        words = phrase("my mind rebels at stagnation")
        assert candidates_from_matches(find_matches(words, TARGET), DURATION)[0].mode is (
            Mode.TARGET
        )

    def test_carries_the_match_score(self) -> None:
        words = phrase("my mind rebels at stagnation")
        candidate = candidates_from_matches(find_matches(words, TARGET), DURATION)[0]
        assert candidate.match_score == pytest.approx(100.0)

    def test_carries_the_dominant_speaker(self) -> None:
        words = phrase("my mind rebels at stagnation", speaker="SPEAKER_03")
        candidate = candidates_from_matches(find_matches(words, TARGET), DURATION)[0]
        assert candidate.speaker_id == "SPEAKER_03"

    def test_carries_the_matched_words(self) -> None:
        """Needed downstream to build the diarization payload for the clip."""
        words = phrase("my mind rebels at stagnation")
        candidate = candidates_from_matches(find_matches(words, TARGET), DURATION)[0]
        assert len(candidate.words) == 5


class TestModeAOrdering:
    def test_candidates_are_chronological(self) -> None:
        """Mode A tries them in time order and stops at the first that verifies,
        so ordering decides which occurrence is reported as first on screen."""
        words = (
            phrase("my mind rebels at stagnation", start=200.0)
            + phrase("filler words in between here", start=100.0)
            + phrase("my mind rebels at stagnation", start=10.0)
        )
        candidates = candidates_from_matches(find_matches(words, TARGET), DURATION)
        assert [c.start for c in candidates] == sorted(c.start for c in candidates)

    def test_no_matches_gives_no_candidates(self) -> None:
        assert candidates_from_matches((), DURATION) == ()


class TestModeADeduplication:
    def test_contained_windows_are_collapsed(self) -> None:
        """A window wholly inside another covers no new footage, so verifying it
        would pay twice for one answer."""
        from dialogue_extractor.matching.candidate_builder import _deduplicate

        outer = Candidate(10.0, 40.0, "S0", "outer", Mode.DISCOVERY)
        inner = Candidate(15.0, 25.0, "S0", "inner", Mode.DISCOVERY)

        kept = _deduplicate([outer, inner])
        assert len(kept) == 1
        assert kept[0].text == "outer"

    def test_partial_overlap_is_not_collapsed(self) -> None:
        """Two distinct lines seconds apart produce overlapping padded windows
        and still need checking separately -- merging them would let one line's
        visual verification be credited to the other."""
        from dialogue_extractor.matching.candidate_builder import _deduplicate

        first = Candidate(10.0, 25.0, "S0", "first", Mode.DISCOVERY)
        second = Candidate(20.0, 35.0, "S0", "second", Mode.DISCOVERY)

        assert len(_deduplicate([first, second])) == 2

    def test_distinct_lines_are_not_merged(self) -> None:
        """Merging would let one line's visual verification be credited to the
        other, which is exactly the mistake the brief is about."""
        words = phrase("my mind rebels at stagnation", start=10.0) + phrase(
            "my mind rebels at stagnation", start=300.0
        )
        candidates = candidates_from_matches(find_matches(words, TARGET), DURATION, pad=1.5)
        assert len(candidates) == 2


class TestModeALengthCap:
    def test_over_long_window_is_trimmed(self) -> None:
        words = phrase("my mind rebels at stagnation", start=100.0)
        candidates = candidates_from_matches(
            find_matches(words, TARGET), DURATION, pad=60.0, max_seconds=30.0
        )
        assert candidates[0].duration <= 30.0

    def test_capped_window_still_contains_the_utterance(self) -> None:
        """The regression this caught: anchoring the cap on the *padded* start
        produced [40, 70] for a line spoken at t=100 -- a clip containing none
        of the dialogue, which the detector would correctly report as nobody
        speaking, for a line that is plainly on screen."""
        words = phrase("my mind rebels at stagnation", start=100.0)
        matches = find_matches(words, TARGET)
        candidate = candidates_from_matches(
            matches, DURATION, pad=60.0, max_seconds=30.0
        )[0]

        assert candidate.start <= matches[0].start, "utterance starts before the clip"
        assert candidate.end >= matches[0].end, "utterance ends after the clip"
        assert candidate.duration <= 30.0

    def test_capped_window_keeps_some_lead_in(self) -> None:
        """A speaker is often visible before they speak, and the answer is the
        first frame where they are both visible and speaking."""
        words = phrase("my mind rebels at stagnation", start=100.0)
        matches = find_matches(words, TARGET)
        candidate = candidates_from_matches(
            matches, DURATION, pad=60.0, max_seconds=30.0
        )[0]
        assert candidate.start < matches[0].start

    def test_utterance_longer_than_the_cap_keeps_its_onset(self) -> None:
        words = phrase("word " * 40, start=100.0, gap=1.0)
        matches = find_matches(words, "word word word")
        if matches:
            candidate = candidates_from_matches(
                matches, DURATION, pad=5.0, max_seconds=10.0
            )[0]
            assert candidate.duration <= 10.0 + 1e-6


class TestModeBMerging:
    def test_adjacent_same_speaker_segments_merge(self) -> None:
        segments = (
            segment("first thing said here", 10.0, speaker="SPEAKER_00"),
            segment("second thing said here", 12.0, speaker="SPEAKER_00"),
        )
        candidates = candidates_from_segments(segments, DURATION)
        assert len(candidates) == 1

    def test_merging_never_crosses_a_speaker_change(self) -> None:
        """A candidate must have exactly one diarized speaker to cross-check
        against the face reported as speaking."""
        segments = (
            segment("first thing said here", 10.0, speaker="SPEAKER_00"),
            segment("second thing said here", 12.0, speaker="SPEAKER_01"),
        )
        candidates = candidates_from_segments(segments, DURATION)
        assert len(candidates) == 2
        assert [c.speaker_id for c in candidates] == ["SPEAKER_00", "SPEAKER_01"]

    def test_merge_respects_the_length_cap(self) -> None:
        segments = tuple(
            segment("some words spoken here", 10.0 + i * 10.0, speaker="SPEAKER_00")
            for i in range(6)
        )
        candidates = candidates_from_segments(segments, DURATION, max_seconds=30.0)
        assert len(candidates) > 1
        assert all(c.duration <= 30.0 + 1e-6 for c in candidates)

    def test_every_segment_ends_up_in_some_candidate(self) -> None:
        segments = tuple(
            segment("words here", 10.0 + i * 5.0, speaker=f"SPEAKER_0{i % 3}")
            for i in range(12)
        )
        candidates = candidates_from_segments(segments, DURATION)
        total_words = sum(len(c.words) for c in candidates)
        assert total_words == sum(len(s.words) for s in segments)

    def test_unassigned_speaker_segments_are_kept(self) -> None:
        segments = (segment("mystery words here", 10.0, speaker=None),)
        candidates = candidates_from_segments(segments, DURATION)
        assert len(candidates) == 1
        assert candidates[0].speaker_id is None


class TestModeBMetadata:
    def test_mode_is_discovery(self) -> None:
        candidates = candidates_from_segments((segment("words here", 10.0),), DURATION)
        assert candidates[0].mode is Mode.DISCOVERY

    def test_match_score_is_none_not_one(self) -> None:
        """There is no target phrase to score against. Substituting 1.0 would
        grant every discovery result credit for something never verified."""
        candidates = candidates_from_segments((segment("words here", 10.0),), DURATION)
        assert candidates[0].match_score is None

    def test_text_is_the_merged_dialogue(self) -> None:
        segments = (
            segment("first thing", 10.0, speaker="SPEAKER_00"),
            segment("second thing", 12.0, speaker="SPEAKER_00"),
        )
        candidates = candidates_from_segments(segments, DURATION)
        assert candidates[0].text == "first thing second thing"

    def test_empty_segments_give_no_candidates(self) -> None:
        assert candidates_from_segments((), DURATION) == ()

    def test_segment_with_no_words_is_skipped(self) -> None:
        empty = SpeechSegment(words=(), speaker_id="SPEAKER_00")
        assert candidates_from_segments((empty,), DURATION) == ()


class TestModeBOrdering:
    def test_candidates_are_chronological(self) -> None:
        segments = tuple(
            segment("words here", 10.0 + i * 20.0, speaker=f"SPEAKER_0{i % 2}")
            for i in range(8)
        )
        candidates = candidates_from_segments(segments, DURATION)
        assert [c.start for c in candidates] == sorted(c.start for c in candidates)


class TestBudget:
    def test_unlimited_by_default(self) -> None:
        candidates = candidates_from_segments(
            tuple(segment("words", 10.0 + i * 20.0, speaker=f"S{i}") for i in range(50)),
            DURATION,
        )
        kept, truncated = apply_budget(candidates)
        assert len(kept) == len(candidates)
        assert truncated is False

    def test_trims_to_the_budget(self) -> None:
        candidates = candidates_from_segments(
            tuple(segment("words", 10.0 + i * 20.0, speaker=f"S{i}") for i in range(20)),
            DURATION,
        )
        kept, truncated = apply_budget(candidates, max_candidates=5)
        assert len(kept) == 5
        assert truncated is True

    def test_keeps_the_earliest(self) -> None:
        candidates = candidates_from_segments(
            tuple(segment("words", 10.0 + i * 20.0, speaker=f"S{i}") for i in range(20)),
            DURATION,
        )
        kept, _ = apply_budget(candidates, max_candidates=3)
        assert kept == candidates[:3]

    def test_budget_larger_than_the_list_is_not_truncation(self) -> None:
        candidates = candidates_from_segments((segment("words", 10.0),), DURATION)
        kept, truncated = apply_budget(candidates, max_candidates=100)
        assert truncated is False
        assert len(kept) == 1

    def test_empty_input(self) -> None:
        kept, truncated = apply_budget((), max_candidates=5)
        assert kept == ()
        assert truncated is False


class TestDeterminism:
    def test_repeated_calls_agree(self) -> None:
        segments = tuple(
            segment("words here", 10.0 + i * 5.0, speaker=f"SPEAKER_0{i % 3}")
            for i in range(10)
        )
        first = candidates_from_segments(segments, DURATION)
        second = candidates_from_segments(segments, DURATION)
        assert [(c.start, c.end, c.speaker_id) for c in first] == [
            (c.start, c.end, c.speaker_id) for c in second
        ]
