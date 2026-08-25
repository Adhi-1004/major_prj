"""Text canonicalization.

Everything normalization removes must be meaningless, and everything it keeps
must matter. The failure mode in one direction is a missed match; in the other,
a false one.
"""

from __future__ import annotations

import pytest

from dialogue_extractor.matching.text_normalizer import (
    normalize,
    normalize_word,
    number_to_words,
    tokenize,
)

TARGET = "My mind rebels at stagnation"


class TestCaseAndPunctuation:
    @pytest.mark.parametrize(
        "text",
        [
            "My mind rebels at stagnation",
            "my mind rebels at stagnation",
            "MY MIND REBELS AT STAGNATION",
            "My mind rebels at stagnation.",
            "My mind rebels at stagnation!",
            '"My mind rebels at stagnation."',
            "  My   mind  rebels   at stagnation  ",
            "My mind rebels at stagnation...",
            "-- My mind rebels at stagnation --",
        ],
    )
    def test_all_forms_of_the_target_agree(self, text: str) -> None:
        assert normalize(text) == "my mind rebels at stagnation"

    def test_collapses_internal_whitespace(self) -> None:
        assert normalize("a\t\tb\n c") == "a b c"

    def test_empty_and_punctuation_only(self) -> None:
        assert normalize("") == ""
        assert normalize("   ") == ""
        assert normalize("...!?") == ""


class TestApostrophes:
    """Dropped without leaving a gap. Splitting would create tokens that never
    match anything."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("don't", "dont"),
            ("dont", "dont"),
            ("don’t", "dont"),  # curly
            ("o'clock", "oclock"),
            ("Watson's", "watsons"),
            ("I'd almost run out", "id almost run out"),
            ("y'all're", "yallre"),
        ],
    )
    def test_contractions_and_possessives(self, text: str, expected: str) -> None:
        assert normalize(text) == expected

    def test_curly_and_straight_agree(self) -> None:
        assert normalize("don’t") == normalize("don't")

    def test_apostrophe_does_not_split_the_word(self) -> None:
        assert normalize("don't") != "don t"


class TestUnicode:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("“quoted”", "quoted"),
            ("em—dash", "em dash"),
            ("en–dash", "en dash"),
            ("non breaking", "non breaking"),
            ("ellipsis…here", "ellipsis here"),
            ("ﬁre", "fire"),  # ligature, folded by NFKC
        ],
    )
    def test_folds_typographic_characters(self, text: str, expected: str) -> None:
        assert normalize(text) == expected


class TestHyphens:
    def test_hyphen_separates_rather_than_joins(self) -> None:
        assert normalize("twenty-one") == "twenty one"

    def test_slash_separates(self) -> None:
        assert normalize("and/or") == "and or"

    def test_hyphenated_matches_spaced(self) -> None:
        assert normalize("well-known man") == normalize("well known man")


class TestNumberToWords:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, "zero"),
            (3, "three"),
            (13, "thirteen"),
            (20, "twenty"),
            (21, "twenty one"),
            (99, "ninety nine"),
            (100, "one hundred"),
            (105, "one hundred five"),
            (221, "two hundred twenty one"),
            (1000, "one thousand"),
            (1985, "one thousand nine hundred eighty five"),
            (999_999, "nine hundred ninety nine thousand nine hundred ninety nine"),
        ],
    )
    def test_spells_numbers(self, value: int, expected: str) -> None:
        assert number_to_words(value) == expected

    def test_omits_the_dialect_dependent_and(self) -> None:
        """"one hundred and five" vs "one hundred five" varies by dialect;
        fuzzy matching tolerates a missing word better than a wrong one."""
        assert "and" not in number_to_words(105).split(" ")

    def test_very_large_numbers_are_left_alone(self) -> None:
        assert number_to_words(1_000_000) == "1000000"

    def test_negative_is_left_alone(self) -> None:
        assert number_to_words(-5) == "-5"


class TestNumbersInText:
    @pytest.mark.parametrize(
        ("digits", "words"),
        [
            ("3 men", "three men"),
            ("21 steps", "twenty one steps"),
            ("chapter 7", "chapter seven"),
            ("in 1985", "in one thousand nine hundred eighty five"),
        ],
    )
    def test_digits_and_words_converge(self, digits: str, words: str) -> None:
        assert normalize(digits) == normalize(words)

    def test_digits_inside_a_word_are_untouched(self) -> None:
        """Only standalone integer tokens are expanded."""
        assert normalize("room4") == "room4"

    def test_number_expansion_can_add_tokens(self) -> None:
        """The reason the matcher counts tokens rather than words."""
        assert len(tokenize("21")) == 2


class TestNormalizeWord:
    def test_single_word(self) -> None:
        assert normalize_word("Stagnation.") == "stagnation"

    def test_punctuation_only_word_becomes_empty(self) -> None:
        assert normalize_word("--") == ""

    def test_may_return_several_tokens(self) -> None:
        assert normalize_word("21") == "twenty one"


class TestTokenize:
    def test_splits_normalized_text(self) -> None:
        assert tokenize(TARGET) == ("my", "mind", "rebels", "at", "stagnation")

    def test_empty_input(self) -> None:
        assert tokenize("") == ()
        assert tokenize("!!!") == ()


class TestOrderIsPreserved:
    def test_normalization_does_not_reorder(self) -> None:
        """Word order is the signal being localized. Anything that sorted or
        deduplicated tokens here would make matching meaningless."""
        assert normalize("my mind rebels") != normalize("rebels mind my")

    def test_normalization_does_not_deduplicate(self) -> None:
        assert normalize("very very good") == "very very good"


class TestIdempotence:
    @pytest.mark.parametrize(
        "text", [TARGET, "don't", "21 steps", "“quoted”", "twenty-one", ""]
    )
    def test_normalizing_twice_changes_nothing(self, text: str) -> None:
        once = normalize(text)
        assert normalize(once) == once
