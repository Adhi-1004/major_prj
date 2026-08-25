"""Locating a target phrase in the transcript's word stream.

PURE apart from importing rapidfuzz's scorer.

**Why `fuzz.ratio` and not `token_set_ratio`.** `token_set_ratio` is built to
score reordered and duplicate-token strings as equivalent -- it would rate
"stagnation at rebels mind my" identically to "my mind rebels at stagnation".
That is the right behaviour for matching names against a database. It is exactly
wrong here, where word order *is* the thing being localized and a scrambled
sentence is not the line anyone spoke. The sliding window already solves the
alignment problem that `token_set_ratio` would otherwise be compensating for, so
an order-sensitive scorer costs nothing and rules out a whole class of false
positives.

**Why windows are sized in tokens, not words.** Normalization can turn one
transcript word into several tokens ("21" -> "twenty one"). Sizing the window by
word count would then compare a two-token target against a one-word window and
score it near zero. Windows are grown by word but measured in tokens.

Matches are returned in chronological order, and the same phrase occurring twice
returns twice -- a flashback or a callback is a real second occurrence, and
which one is "first on screen" is decided later by visual verification, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from dialogue_extractor.config.constants import (
    MATCH_SCORE_FLOOR,
    MATCH_WINDOW_SLACK,
)
from dialogue_extractor.domain.models import Word
from dialogue_extractor.matching.text_normalizer import normalize, normalize_word


@dataclass(frozen=True, slots=True)
class PhraseMatch:
    """One place in the transcript where the target phrase may have been said."""

    start_index: int
    """Index of the first matched word in the original word tuple."""

    end_index: int
    """Index one past the last matched word."""

    score: float
    """rapidfuzz ratio, 0-100."""

    words: tuple[Word, ...]
    """The original Word objects, timings and probabilities intact."""

    @property
    def start(self) -> float:
        return self.words[0].start if self.words else 0.0

    @property
    def end(self) -> float:
        return self.words[-1].end if self.words else 0.0

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def mean_probability(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.probability for w in self.words) / len(self.words)

    def dominant_speaker(self) -> str | None:
        """The speaker most of the matched words belong to.

        A majority vote rather than "the first word's speaker": diarization can
        misattribute a word at an utterance boundary, and a single stray word
        must not decide the speaker for the whole phrase. That identity is what
        Phase 5 cross-checks against the face reported as speaking, so getting
        it wrong rejects a correct match.

        Ties break toward the earliest-occurring speaker, for determinism.
        """
        counts: dict[str, int] = {}
        order: dict[str, int] = {}
        for index, word in enumerate(self.words):
            if word.speaker_id is None:
                continue
            counts[word.speaker_id] = counts.get(word.speaker_id, 0) + 1
            order.setdefault(word.speaker_id, index)

        if not counts:
            return None
        return min(counts, key=lambda s: (-counts[s], order[s]))


def score_phrase(target: str, candidate: str) -> float:
    """Order-sensitive similarity of two already-normalized strings, 0-100."""
    if not target or not candidate:
        return 0.0
    return float(fuzz.ratio(target, candidate))


def _normalized_words(words: tuple[Word, ...]) -> list[tuple[str, int]]:
    """Normalized text and token count for each word, positionally aligned."""
    prepared: list[tuple[str, int]] = []
    for word in words:
        text = normalize_word(word.text)
        prepared.append((text, len(text.split(" ")) if text else 0))
    return prepared


def find_matches(
    words: tuple[Word, ...],
    target: str,
    *,
    score_floor: float = MATCH_SCORE_FLOOR,
    slack: int = MATCH_WINDOW_SLACK,
) -> tuple[PhraseMatch, ...]:
    """Find every occurrence of `target` in `words`, best-first within a region.

    Returns matches in chronological order, with overlapping windows collapsed
    to the single best-scoring one so a phrase is reported once rather than
    once per plausible alignment.
    """
    normalized_target = normalize(target)
    if not normalized_target or not words:
        return ()

    target_tokens = len(normalized_target.split(" "))
    minimum = max(1, target_tokens - slack)
    maximum = target_tokens + slack

    prepared = _normalized_words(words)
    raw: list[PhraseMatch] = []

    for start in range(len(words)):
        token_count = 0
        pieces: list[str] = []

        for end in range(start, len(words)):
            text, count = prepared[end]
            if count:
                pieces.append(text)
                token_count += count

            if token_count > maximum:
                break
            if token_count < minimum:
                continue

            score = score_phrase(normalized_target, " ".join(pieces))
            if score >= score_floor:
                raw.append(
                    PhraseMatch(
                        start_index=start,
                        end_index=end + 1,
                        score=score,
                        words=words[start : end + 1],
                    )
                )

    return _collapse_overlaps(raw)


def _collapse_overlaps(matches: list[PhraseMatch]) -> tuple[PhraseMatch, ...]:
    """Keep the best-scoring match per overlapping region.

    Every plausible alignment of a phrase scores above the floor, so without
    this a single utterance is reported several times with slightly different
    boundaries. Sorting by score first, then walking in that order, means the
    tightest alignment wins and the rest are absorbed.

    Ties break toward the earlier and shorter match, so the result does not
    depend on the order windows happened to be generated in.
    """
    if not matches:
        return ()

    ordered = sorted(
        matches,
        key=lambda m: (-m.score, m.start_index, m.end_index - m.start_index),
    )

    kept: list[PhraseMatch] = []
    for match in ordered:
        if any(
            match.start_index < existing.end_index and existing.start_index < match.end_index
            for existing in kept
        ):
            continue
        kept.append(match)

    return tuple(sorted(kept, key=lambda m: (m.start_index, m.end_index)))


def best_match(
    words: tuple[Word, ...],
    target: str,
    *,
    score_floor: float = MATCH_SCORE_FLOOR,
    slack: int = MATCH_WINDOW_SLACK,
) -> PhraseMatch | None:
    """Highest-scoring match, or None. Ties break toward the earlier occurrence."""
    matches = find_matches(words, target, score_floor=score_floor, slack=slack)
    if not matches:
        return None
    return min(matches, key=lambda m: (-m.score, m.start_index))
