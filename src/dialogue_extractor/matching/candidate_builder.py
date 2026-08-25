"""Turning matches (Mode A) or segments (Mode B) into windows worth verifying.

PURE: no I/O, no third-party imports.

A candidate is a span of video that will be trimmed into a clip and sent for
active-speaker detection. Every one costs a network round-trip and a re-encode,
so this stage decides both *where* to look and *how much*.

The two modes differ only in where candidates come from:

  Mode A  one target phrase -> a few padded windows, tried in time order,
          stopping at the first that verifies on screen.
  Mode B  no target -> every speech segment, merged into chunks so a
          feature-length film does not become ~1200 separate requests.
"""

from __future__ import annotations

from dialogue_extractor.config.constants import (
    ASD_MIN_CLIP_SECONDS,
    CHUNK_MAX_SECONDS,
    FRAGMENT_MERGE_GAP,
    WINDOW_PAD_SECONDS,
)
from dialogue_extractor.domain.enums import Mode
from dialogue_extractor.domain.models import Candidate, SpeechSegment
from dialogue_extractor.domain.timing import pad_window
from dialogue_extractor.matching.fuzzy_matcher import PhraseMatch


def candidates_from_matches(
    matches: tuple[PhraseMatch, ...],
    duration: float,
    *,
    pad: float = WINDOW_PAD_SECONDS,
    max_seconds: float = CHUNK_MAX_SECONDS,
    min_seconds: float = ASD_MIN_CLIP_SECONDS,
) -> tuple[Candidate, ...]:
    """Mode A: one padded window per match, in chronological order.

    Padding absorbs word-timestamp jitter so the clip reliably contains the
    whole utterance. It does not affect final precision -- the exact frame comes
    from the active-speaker model at frame granularity, not from the ASR
    timestamp -- it only affects whether the utterance is inside the clip at all.
    """
    candidates: list[Candidate] = []

    for match in matches:
        start, end = pad_window(match.start, match.end, pad, duration)
        start, end = ensure_min_duration(start, end, duration, minimum=min_seconds)
        start, end = _clamp_length(
            start, end, max_seconds, duration,
            core_start=match.start, core_end=match.end,
        )

        candidates.append(
            Candidate(
                start=start,
                end=end,
                speaker_id=match.dominant_speaker(),
                text=match.text,
                mode=Mode.TARGET,
                match_score=match.score,
                words=match.words,
            )
        )

    return _deduplicate(candidates)


def candidates_from_segments(
    segments: tuple[SpeechSegment, ...],
    duration: float,
    *,
    pad: float = 0.0,
    max_seconds: float = CHUNK_MAX_SECONDS,
    min_seconds: float = ASD_MIN_CLIP_SECONDS,
    absorb_fragments: bool = True,
) -> tuple[Candidate, ...]:
    """Mode B: every segment, with adjacent same-speaker segments merged.

    Merging is what keeps discovery affordable: consecutive segments from one
    speaker are usually one continuous piece of footage, so verifying them as a
    single clip costs one request instead of several and gives the detector more
    temporal context to work with.

    Merging never crosses a speaker change, because the whole point of a
    candidate is that it has *one* diarized speaker to cross-check against the
    face reported as speaking.
    """
    if not segments:
        return ()

    if absorb_fragments:
        segments = absorb_unattributed_fragments(segments)
        if not segments:
            return ()

    merged: list[list[SpeechSegment]] = [[segments[0]]]

    for segment in segments[1:]:
        group = merged[-1]
        same_speaker = segment.speaker_id == group[0].speaker_id
        would_span = segment.end - group[0].start

        if same_speaker and would_span <= max_seconds:
            group.append(segment)
        else:
            merged.append([segment])

    candidates: list[Candidate] = []
    for group in merged:
        words = tuple(w for segment in group for w in segment.words)
        if not words:
            continue

        start, end = pad_window(group[0].start, group[-1].end, pad, duration)
        start, end = ensure_min_duration(start, end, duration, minimum=min_seconds)
        start, end = _clamp_length(
            start, end, max_seconds, duration,
            core_start=group[0].start, core_end=group[-1].end,
        )

        candidates.append(
            Candidate(
                start=start,
                end=end,
                speaker_id=group[0].speaker_id,
                text=" ".join(segment.text for segment in group).strip(),
                mode=Mode.DISCOVERY,
                match_score=None,  # No target phrase, so the term is undefined.
                words=words,
            )
        )

    return _deduplicate(candidates)


def ensure_min_duration(
    start: float,
    end: float,
    duration: float,
    *,
    minimum: float = ASD_MIN_CLIP_SECONDS,
) -> tuple[float, float]:
    """Widen a window to at least `minimum` seconds, centred on its content.

    The detection service returns *zero frames* for a clip below roughly three
    seconds -- see ASD_MIN_CLIP_SECONDS. It does not complain; it simply gives
    nothing back, which is indistinguishable from "nobody was on screen".

    Widening rather than discarding, because a short line is still dialogue
    someone might search for. Growth is symmetric so the utterance keeps its
    place in the middle, and falls back to one-sided growth at the boundaries
    of the video rather than giving up.
    """
    span = end - start
    if span >= minimum or duration <= 0:
        return start, end

    if duration <= minimum:
        # The whole video is shorter than the floor; send all of it.
        return 0.0, duration

    needed = minimum - span
    new_start = max(0.0, start - needed / 2.0)
    new_end = min(duration, new_start + minimum)
    # Running into the end pushes the start back instead of returning short.
    new_start = max(0.0, new_end - minimum)
    return new_start, new_end


def absorb_unattributed_fragments(
    segments: tuple[SpeechSegment, ...],
    *,
    max_gap: float = FRAGMENT_MERGE_GAP,
) -> tuple[SpeechSegment, ...]:
    """Fold speaker-less single-word fragments into the neighbour they belong to.

    Diarization strands words at utterance boundaries: "Tell" sits immediately
    before "me, Doctor, as a medical man...", with no gap and no speaker. Left
    alone it becomes its own candidate -- one that costs a request and has no
    speaker to cross-check against, so its result can never be more than
    unverified.

    Only a *zero-gap* neighbour absorbs it. A fragment genuinely isolated in
    time keeps its own candidate: assigning it a speaker on proximity alone
    would be a guess, and this stage exists to avoid guesses.
    """
    if not segments:
        return ()

    result: list[SpeechSegment] = []

    for segment in segments:
        is_fragment = segment.speaker_id is None and len(segment.words) <= 1

        if is_fragment and result:
            previous = result[-1]
            if previous.speaker_id is not None and (segment.start - previous.end) <= max_gap:
                result[-1] = SpeechSegment(
                    words=previous.words + segment.words,
                    speaker_id=previous.speaker_id,
                )
                continue

        result.append(segment)

    # A fragment can also lead into the utterance it belongs to, which the
    # backward pass above cannot see.
    merged: list[SpeechSegment] = []
    index = 0
    while index < len(result):
        segment = result[index]
        is_fragment = segment.speaker_id is None and len(segment.words) <= 1
        following = result[index + 1] if index + 1 < len(result) else None

        if (
            is_fragment
            and following is not None
            and following.speaker_id is not None
            and (following.start - segment.end) <= max_gap
        ):
            merged.append(
                SpeechSegment(
                    words=segment.words + following.words,
                    speaker_id=following.speaker_id,
                )
            )
            index += 2
            continue

        merged.append(segment)
        index += 1

    return tuple(merged)


def _clamp_length(
    start: float,
    end: float,
    max_seconds: float,
    duration: float,
    *,
    core_start: float,
    core_end: float,
) -> tuple[float, float]:
    """Cap a window's length while keeping the utterance inside it.

    The cap is anchored on the *utterance*, not on the padded window's start.
    Anchoring on the padded start is subtly catastrophic: with generous padding
    the window can begin long before the line is spoken, so trimming forward
    from there produces a clip containing none of the dialogue -- and the
    detector would correctly report no one speaking, for a line that is on
    screen.

    Remaining room is spent on lead-in first, because a speaker is often already
    visible before they start talking, and the answer is the *first* frame where
    they are both visible and speaking.
    """
    if max_seconds <= 0 or (end - start) <= max_seconds:
        return start, end

    core_length = core_end - core_start
    if core_length >= max_seconds:
        # The utterance alone exceeds the cap: keep its onset and take what fits.
        capped_start = max(0.0, core_start)
        return capped_start, min(duration, capped_start + max_seconds)

    slack = max_seconds - core_length
    lead = min(core_start - start, slack / 2.0)
    capped_start = max(0.0, core_start - lead)
    return capped_start, min(duration, capped_start + max_seconds)


def _deduplicate(candidates: list[Candidate]) -> tuple[Candidate, ...]:
    """Drop windows wholly contained in an earlier one, and sort by time.

    Padding can make two nearby matches produce overlapping windows covering
    the same footage. Verifying both would pay twice for one answer.

    Only *containment* is collapsed, not mere overlap: two distinct lines a
    second apart genuinely need checking separately, and merging them would let
    one line's visual verification be credited to the other.
    """
    if not candidates:
        return ()

    ordered = sorted(candidates, key=lambda c: (c.start, -(c.end - c.start)))

    kept: list[Candidate] = []
    for candidate in ordered:
        if any(
            existing.start <= candidate.start and candidate.end <= existing.end
            for existing in kept
        ):
            continue
        kept.append(candidate)

    return tuple(kept)


def apply_budget(
    candidates: tuple[Candidate, ...],
    *,
    max_candidates: int = 0,
) -> tuple[tuple[Candidate, ...], bool]:
    """Trim to a budget, reporting whether anything was dropped.

    Returns (kept, truncated). The flag exists so a partial sweep is never
    presented as an exhaustive one.
    """
    if max_candidates <= 0 or len(candidates) <= max_candidates:
        return candidates, False
    return candidates[:max_candidates], True
