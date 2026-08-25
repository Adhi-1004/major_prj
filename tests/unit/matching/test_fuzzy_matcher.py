"""Phrase localization in the transcript word stream.

The two properties that matter most: word order must count (so a scrambled
sentence is not a match), and a phrase said twice must be reported twice (a
flashback is a real second occurrence, and which one is on screen is decided by
visual verification later, not here).
"""

from __future__ import annotations

import pytest

from dialogue_extractor.domain.models import Word
from dialogue_extractor.matching.fuzzy_matcher import (
    best_match,
    find_matches,
    score_phrase,
)

TARGET = "My mind rebels at stagnation"


def stream(*specs: tuple[str, float, float] | tuple[str, float, float, str]) -> tuple[Word, ...]:
    words = []
    for spec in specs:
        text, start, end = spec[0], spec[1], spec[2]
        speaker = spec[3] if len(spec) > 3 else None
        words.append(Word(text=text, start=start, end=end, speaker_id=speaker))
    return tuple(words)


def phrase(text: str, start: float = 10.0, gap: float = 0.3, speaker: str | None = None):
    """Turn a sentence into a word stream with plausible timings."""
    words = []
    t = start
    for token in text.split(" "):
        words.append(Word(text=token, start=t, end=t + gap * 0.8, speaker_id=speaker))
        t += gap
    return tuple(words)


class TestScorePhrase:
    def test_identical_is_perfect(self) -> None:
        assert score_phrase("my mind rebels", "my mind rebels") == 100.0

    def test_empty_scores_zero(self) -> None:
        assert score_phrase("", "anything") == 0.0
        assert score_phrase("anything", "") == 0.0

    def test_is_order_sensitive(self) -> None:
        """The whole reason token_set_ratio was rejected: it would score these
        two identically, and a scrambled sentence is not the line anyone said."""
        correct = score_phrase("my mind rebels at stagnation", "my mind rebels at stagnation")
        scrambled = score_phrase("my mind rebels at stagnation", "stagnation at rebels mind my")
        assert correct == 100.0
        assert scrambled < 70.0, f"scrambled scored {scrambled}, order is not counting"


class TestExactMatch:
    def test_finds_the_phrase(self) -> None:
        words = phrase("well my mind rebels at stagnation indeed")
        matches = find_matches(words, TARGET)

        assert len(matches) == 1
        assert matches[0].text == "my mind rebels at stagnation"
        assert matches[0].score == pytest.approx(100.0)

    def test_reports_the_right_word_indices(self) -> None:
        words = phrase("well my mind rebels at stagnation indeed")
        match = find_matches(words, TARGET)[0]
        assert match.start_index == 1
        assert match.end_index == 6

    def test_timings_come_from_the_matched_words(self) -> None:
        words = phrase("well my mind rebels at stagnation indeed", start=100.0, gap=0.5)
        match = find_matches(words, TARGET)[0]
        assert match.start == pytest.approx(words[1].start)
        assert match.end == pytest.approx(words[5].end)

    def test_case_and_punctuation_do_not_matter(self) -> None:
        words = phrase("My mind rebels at stagnation.")
        assert find_matches(words, "my mind rebels at stagnation")

    def test_phrase_at_the_very_start(self) -> None:
        words = phrase("my mind rebels at stagnation and so on", start=0.0)
        assert find_matches(words, TARGET)[0].start_index == 0

    def test_phrase_at_the_very_end(self) -> None:
        words = phrase("and so my mind rebels at stagnation")
        matches = find_matches(words, TARGET)
        assert matches[0].end_index == len(words)


class TestTolerance:
    def test_one_word_misheard_still_matches(self) -> None:
        """ASR reliably mishears a word; requiring exactness would make the
        whole fuzzy approach pointless."""
        words = phrase("my mind rebels at stagnant")
        matches = find_matches(words, TARGET)
        assert matches, "a single word error should not lose the match"
        assert matches[0].score < 100.0

    def test_missing_small_word_still_matches(self) -> None:
        words = phrase("my mind rebels stagnation")
        assert find_matches(words, TARGET)

    def test_inserted_word_still_matches(self) -> None:
        words = phrase("my own mind rebels at stagnation")
        assert find_matches(words, TARGET)

    def test_unrelated_text_does_not_match(self) -> None:
        words = phrase("the weather today is remarkably pleasant indeed")
        assert find_matches(words, TARGET) == ()

    def test_scrambled_words_do_not_match(self) -> None:
        words = phrase("stagnation at rebels mind my")
        assert find_matches(words, TARGET) == ()

    def test_partial_phrase_does_not_match(self) -> None:
        words = phrase("my mind")
        assert find_matches(words, TARGET) == ()

    def test_floor_is_respected(self) -> None:
        words = phrase("my mind rebels at stagnant")
        assert find_matches(words, TARGET, score_floor=99.0) == ()
        assert find_matches(words, TARGET, score_floor=60.0)


class TestRepeatedOccurrences:
    """A flashback or callback is a real second occurrence. Which one is on
    screen is decided by visual verification, not by discarding one here."""

    def test_both_occurrences_are_returned(self) -> None:
        first = phrase("my mind rebels at stagnation", start=10.0)
        filler = phrase("and much later in the film", start=60.0)
        second = phrase("my mind rebels at stagnation", start=120.0)

        matches = find_matches(first + filler + second, TARGET)
        assert len(matches) == 2

    def test_they_are_in_chronological_order(self) -> None:
        first = phrase("my mind rebels at stagnation", start=10.0)
        filler = phrase("and much later in the film", start=60.0)
        second = phrase("my mind rebels at stagnation", start=120.0)

        matches = find_matches(first + filler + second, TARGET)
        assert matches[0].start < matches[1].start
        assert matches[0].start == pytest.approx(10.0)

    def test_three_occurrences(self) -> None:
        words: tuple[Word, ...] = ()
        for offset in (10.0, 100.0, 200.0):
            words += phrase("my mind rebels at stagnation", start=offset)
            words += phrase("some other words entirely here", start=offset + 30)
        assert len(find_matches(words, TARGET)) == 3


class TestOverlapCollapsing:
    def test_one_utterance_is_reported_once(self) -> None:
        """Every plausible alignment scores above the floor; without collapsing,
        a single line comes back several times with different boundaries."""
        words = phrase("well my mind rebels at stagnation indeed")
        assert len(find_matches(words, TARGET)) == 1

    def test_the_best_alignment_wins(self) -> None:
        words = phrase("well my mind rebels at stagnation indeed")
        match = find_matches(words, TARGET)[0]
        assert match.text == "my mind rebels at stagnation"

    def test_adjacent_distinct_phrases_are_kept_separate(self) -> None:
        words = phrase("my mind rebels at stagnation", start=10.0) + phrase(
            "my mind rebels at stagnation", start=15.0
        )
        assert len(find_matches(words, TARGET)) == 2


class TestCrossSegmentPhrases:
    def test_phrase_spanning_a_segment_boundary_matches(self) -> None:
        """Segment boundaries are discarded upstream, so matching slides over
        one continuous stream and never sees an edge."""
        first_half = stream(("My", 10.0, 10.2), ("mind", 10.2, 10.5))
        second_half = stream(
            ("rebels", 10.5, 10.9), ("at", 10.9, 11.0), ("stagnation", 11.0, 11.6)
        )
        matches = find_matches(first_half + second_half, TARGET)
        assert len(matches) == 1
        assert matches[0].score == pytest.approx(100.0)


class TestNumberExpansion:
    def test_digits_in_the_transcript_match_words_in_the_target(self) -> None:
        """"21" normalizes to two tokens, so a word-sized window would miss it."""
        words = stream(("there", 0.0, 0.3), ("were", 0.3, 0.6), ("21", 0.6, 1.0),
                       ("steps", 1.0, 1.4))
        assert find_matches(words, "there were twenty one steps")

    def test_words_in_the_transcript_match_digits_in_the_target(self) -> None:
        words = phrase("there were twenty one steps")
        assert find_matches(words, "there were 21 steps")


class TestDominantSpeaker:
    def test_single_speaker(self) -> None:
        words = phrase("my mind rebels at stagnation", speaker="SPEAKER_03")
        assert find_matches(words, TARGET)[0].dominant_speaker() == "SPEAKER_03"

    def test_majority_wins_over_a_stray_word(self) -> None:
        """Diarization misattributes words at utterance boundaries; one stray
        word must not decide the speaker for the whole phrase, because that
        identity is what Phase 5 cross-checks against the speaking face."""
        words = stream(
            ("my", 10.0, 10.2, "SPEAKER_03"),
            ("mind", 10.2, 10.5, "SPEAKER_03"),
            ("rebels", 10.5, 10.9, "SPEAKER_05"),  # stray
            ("at", 10.9, 11.0, "SPEAKER_03"),
            ("stagnation", 11.0, 11.6, "SPEAKER_03"),
        )
        assert find_matches(words, TARGET)[0].dominant_speaker() == "SPEAKER_03"

    def test_unassigned_words_do_not_vote(self) -> None:
        words = stream(
            ("my", 10.0, 10.2, None),
            ("mind", 10.2, 10.5, None),
            ("rebels", 10.5, 10.9, "SPEAKER_03"),
            ("at", 10.9, 11.0, None),
            ("stagnation", 11.0, 11.6, None),
        )
        assert find_matches(words, TARGET)[0].dominant_speaker() == "SPEAKER_03"

    def test_all_unassigned_returns_none(self) -> None:
        words = phrase("my mind rebels at stagnation", speaker=None)
        assert find_matches(words, TARGET)[0].dominant_speaker() is None

    def test_tie_is_deterministic(self) -> None:
        words = stream(
            ("my", 10.0, 10.2, "SPEAKER_05"),
            ("mind", 10.2, 10.5, "SPEAKER_05"),
            ("rebels", 10.5, 10.9, None),
            ("at", 10.9, 11.0, "SPEAKER_03"),
            ("stagnation", 11.0, 11.6, "SPEAKER_03"),
        )
        match = find_matches(words, TARGET)[0]
        assert match.dominant_speaker() == "SPEAKER_05"  # earliest of the tied


class TestMatchProperties:
    def test_mean_probability(self) -> None:
        words = (
            Word("my", 10.0, 10.2, probability=0.9),
            Word("mind", 10.2, 10.5, probability=0.8),
            Word("rebels", 10.5, 10.9, probability=1.0),
            Word("at", 10.9, 11.0, probability=0.6),
            Word("stagnation", 11.0, 11.6, probability=0.7),
        )
        match = find_matches(words, TARGET)[0]
        assert match.mean_probability == pytest.approx(0.8)

    def test_original_words_are_preserved(self) -> None:
        """Timings and probabilities must survive matching intact."""
        words = phrase("my mind rebels at stagnation", speaker="SPEAKER_03")
        match = find_matches(words, TARGET)[0]
        assert match.words == words
        assert all(w.speaker_id == "SPEAKER_03" for w in match.words)


class TestEdgeCases:
    def test_empty_word_stream(self) -> None:
        assert find_matches((), TARGET) == ()

    def test_empty_target(self) -> None:
        assert find_matches(phrase("some words here"), "") == ()

    def test_punctuation_only_target(self) -> None:
        assert find_matches(phrase("some words here"), "...!") == ()

    def test_single_word_target(self) -> None:
        words = phrase("the word stagnation appears here")
        matches = find_matches(words, "stagnation")
        assert matches
        assert "stagnation" in matches[0].text.lower()

    def test_target_longer_than_the_transcript(self) -> None:
        assert find_matches(phrase("short"), TARGET) == ()

    def test_punctuation_only_words_are_skipped(self) -> None:
        words = stream(
            ("my", 10.0, 10.2), ("--", 10.2, 10.3), ("mind", 10.3, 10.5),
            ("rebels", 10.5, 10.9), ("at", 10.9, 11.0), ("stagnation", 11.0, 11.6),
        )
        assert find_matches(words, TARGET)


class TestDeterminism:
    def test_repeated_calls_agree(self) -> None:
        words = phrase("well my mind rebels at stagnation indeed")
        first = find_matches(words, TARGET)
        second = find_matches(words, TARGET)
        assert [(m.start_index, m.end_index, m.score) for m in first] == [
            (m.start_index, m.end_index, m.score) for m in second
        ]


class TestBestMatch:
    def test_returns_the_highest_scoring(self) -> None:
        exact = phrase("my mind rebels at stagnation", start=10.0)
        filler = phrase("nothing to see here at all", start=60.0)
        approximate = phrase("my mind rebels at stagnant", start=120.0)

        match = best_match(exact + filler + approximate, TARGET)
        assert match is not None
        assert match.start == pytest.approx(10.0)

    def test_none_when_nothing_matches(self) -> None:
        assert best_match(phrase("entirely unrelated words here"), TARGET) is None

    def test_tie_breaks_toward_the_earlier(self) -> None:
        first = phrase("my mind rebels at stagnation", start=10.0)
        filler = phrase("some filler words in between", start=60.0)
        second = phrase("my mind rebels at stagnation", start=120.0)

        match = best_match(first + filler + second, TARGET)
        assert match is not None
        assert match.start == pytest.approx(10.0)
