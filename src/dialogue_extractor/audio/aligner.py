"""Attaching speakers to words, and grouping words into speech segments.

PURE: no I/O, no models, no third-party imports. All the subtle behaviour of
this stage is decided here, which is what makes it cheap to test exhaustively.

Transcription and diarization run independently and produce two unrelated
timelines: one of words, one of speaker turns. Neither knows about the other, so
a word can sit wholly inside one turn, straddle two, or fall in a gap where
diarization heard nothing. Each case needs a defined answer:

  * inside a turn        -> that speaker
  * straddling turns     -> the speaker whose turn it overlaps *most*
  * in a gap             -> None, and the word is kept

Keeping unassigned words matters. Dropping them would silently punch holes in
the word stream that Phase 4 slides a match window over, so a target phrase
spanning a diarization gap would become unfindable -- a failure that would look
like "the line isn't in this video".
"""

from __future__ import annotations

from itertools import pairwise

from dialogue_extractor.config.constants import MAX_INTRA_SEGMENT_GAP
from dialogue_extractor.domain.models import SpeakerTurn, SpeechSegment, Word
from dialogue_extractor.domain.timing import overlap


def assign_speakers(
    words: tuple[Word, ...],
    turns: tuple[SpeakerTurn, ...],
) -> tuple[Word, ...]:
    """Tag each word with the speaker whose turn it overlaps most.

    Ties break toward the earlier-starting turn, so the result is deterministic
    rather than dependent on the order diarization happened to emit turns in.
    """
    if not turns:
        return words

    ordered = sorted(turns, key=lambda t: (t.start, t.end))

    return tuple(
        Word(
            text=word.text,
            start=word.start,
            end=word.end,
            probability=word.probability,
            speaker_id=_best_speaker(word, ordered),
        )
        for word in words
    )


def _best_speaker(word: Word, ordered_turns: list[SpeakerTurn]) -> str | None:
    """Speaker of the turn overlapping `word` most, or None if none does."""
    best_id: str | None = None
    best_overlap = 0.0

    for turn in ordered_turns:
        # Turns are sorted, so once one starts after the word ends, so do the rest.
        if turn.start >= word.end:
            break
        if turn.end <= word.start:
            continue

        amount = overlap(word.start, word.end, turn.start, turn.end)
        if amount > best_overlap:
            best_overlap, best_id = amount, turn.speaker_id

    if best_id is not None:
        return best_id

    # A zero-length word (start == end) has no overlap with anything, but still
    # sits inside a turn. Fall back to containment so it is not orphaned.
    for turn in ordered_turns:
        if turn.start <= word.start <= turn.end:
            return turn.speaker_id
    return None


def build_segments(
    words: tuple[Word, ...],
    *,
    max_gap: float = MAX_INTRA_SEGMENT_GAP,
) -> tuple[SpeechSegment, ...]:
    """Group consecutive words into segments.

    A segment breaks on a speaker change, or on a silence longer than `max_gap`.
    The gap rule matters because one speaker's turn can span a long pause -- two
    separate lines minutes apart would otherwise merge into a single segment
    whose time span covers footage containing neither of them.
    """
    if not words:
        return ()

    segments: list[SpeechSegment] = []
    current: list[Word] = [words[0]]

    for previous, word in pairwise(words):
        speaker_changed = word.speaker_id != previous.speaker_id
        gap_too_long = (word.start - previous.end) > max_gap

        if speaker_changed or gap_too_long:
            segments.append(
                SpeechSegment(words=tuple(current), speaker_id=current[0].speaker_id)
            )
            current = [word]
        else:
            current.append(word)

    segments.append(SpeechSegment(words=tuple(current), speaker_id=current[0].speaker_id))
    return tuple(segments)


def speaker_statistics(words: tuple[Word, ...]) -> dict[str, int]:
    """Word count per speaker, with unassigned words under the empty key.

    Used for reporting and for sanity-checking diarization: a video where every
    word landed on one speaker, or where most words are unassigned, is worth
    surfacing rather than silently proceeding.
    """
    counts: dict[str, int] = {}
    for word in words:
        key = word.speaker_id or ""
        counts[key] = counts.get(key, 0) + 1
    return counts
