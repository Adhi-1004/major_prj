"""Canonicalizing text so a typed phrase and an ASR transcript can be compared.

PURE: no I/O, no third-party imports.

The two sides of the comparison come from different worlds. The target phrase is
typed by a person, with punctuation, capitals, and whatever quote characters
their keyboard produced. The transcript comes from Whisper, which makes its own
choices about capitalisation, punctuation, and whether to write a number as
digits or words. Comparing them raw fails on differences that carry no meaning:

    "My mind rebels at stagnation."   vs   "my mind rebels at stagnation"
    "don't"                            vs   "dont"
    "3 men"                            vs   "three men"
    "don’t" (curly)               vs   "don't" (straight)

Normalization removes exactly those, and nothing else. In particular it does
**not** reorder or deduplicate tokens: word order is the signal being localized,
so anything that discards it would make the match meaningless.

One important property: `normalize_word` maps one input word to a normalized
*string*, which may contain spaces ("21" -> "twenty one"). Callers must treat
the result as text to be joined, not as a single token, or the token accounting
in the matcher goes wrong.
"""

from __future__ import annotations

import re
import unicodedata

# Curly quotes, prime marks, and the various dashes a word processor produces.
_APOSTROPHES = "‘’ʼ′`´"
_QUOTES = "“”„«»"
_DASHES = "‐‑‒–—―"

_TRANSLATION = str.maketrans(
    {
        **{c: "'" for c in _APOSTROPHES},
        **{c: '"' for c in _QUOTES},
        **{c: "-" for c in _DASHES},
        "…": "...",  # ellipsis
        " ": " ",  # non-breaking space
    }
)

# Everything that is not a letter, digit, or space, once quotes are unified.
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_WHITESPACE = re.compile(r"\s+")
_INTEGER = re.compile(r"^\d+$")

_UNITS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_TENS = (
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
)

MAX_SPELLED_NUMBER = 999_999
"""Above this, digits are left as-is.

Larger numbers are vanishingly rare in dialogue, and spelling them out invents
word-order choices ("one hundred thousand" vs "a hundred thousand") that would
be as likely to break a match as to fix one.
"""


def _under_thousand(value: int) -> str:
    if value < 20:
        return _UNITS[value]
    if value < 100:
        tens, unit = divmod(value, 10)
        return _TENS[tens] + (f" {_UNITS[unit]}" if unit else "")
    hundreds, remainder = divmod(value, 100)
    result = f"{_UNITS[hundreds]} hundred"
    return f"{result} {_under_thousand(remainder)}" if remainder else result


def number_to_words(value: int) -> str:
    """Spell a non-negative integer, e.g. 21 -> "twenty one".

    Deliberately without "and" ("one hundred five", not "one hundred and five"):
    the joining word varies by dialect, and fuzzy matching tolerates its absence
    far better than it tolerates the wrong one being inserted.
    """
    if value < 0 or value > MAX_SPELLED_NUMBER:
        return str(value)
    if value < 1000:
        return _under_thousand(value)
    thousands, remainder = divmod(value, 1000)
    result = f"{_under_thousand(thousands)} thousand"
    return f"{result} {_under_thousand(remainder)}" if remainder else result


def normalize(text: str) -> str:
    """Canonical form of `text`, for comparison only.

    Order of operations matters: unicode is folded first so curly quotes become
    straight ones, apostrophes are dropped *before* punctuation stripping so
    "don't" collapses to "dont" rather than splitting into "don t", and numbers
    are expanded last, on tokens that are already clean digits.
    """
    if not text:
        return ""

    # Compatibility-fold so ligatures and full-width forms reduce to ASCII-ish.
    folded = unicodedata.normalize("NFKC", text).translate(_TRANSLATION)
    lowered = folded.casefold()

    # Drop apostrophes without leaving a gap: "don't" -> "dont", "o'clock" ->
    # "oclock". Splitting instead would create tokens that never match.
    lowered = lowered.replace("'", "")

    # Hyphens and slashes separate words rather than joining them:
    # "twenty-one" -> "twenty one".
    lowered = re.sub(r"[-/]+", " ", lowered)

    cleaned = _NON_ALNUM.sub(" ", lowered)
    collapsed = _WHITESPACE.sub(" ", cleaned).strip()
    if not collapsed:
        return ""

    return " ".join(
        number_to_words(int(token)) if _INTEGER.match(token) else token
        for token in collapsed.split(" ")
    )


def normalize_word(text: str) -> str:
    """Normalize one transcript word.

    May return an empty string (the word was pure punctuation) or several
    space-separated tokens ("21" -> "twenty one"). Both cases are why the
    matcher counts tokens rather than assuming one word equals one token.
    """
    return normalize(text)


def tokenize(text: str) -> tuple[str, ...]:
    """Normalize and split into tokens."""
    normalized = normalize(text)
    return tuple(normalized.split(" ")) if normalized else ()
