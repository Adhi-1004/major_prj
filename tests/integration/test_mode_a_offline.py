"""Mode A end-to-end, offline: transcript JSON + target phrase -> candidates.

No network, no models, no GPU. This is the regression gate for the whole
audio-side pipeline: it pins the exact candidate list produced from a fixed
transcript, so any change to normalization, matching, alignment, or windowing
that alters the answer shows up here rather than in a live run.

The fixture is built in code rather than committed, so the timings and speaker
attribution it depends on are visible next to the assertions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dialogue_extractor.audio.aligner import assign_speakers, build_segments
from dialogue_extractor.audio.cached_transcriber import (
    transcript_from_dict,
    transcript_to_dict,
)
from dialogue_extractor.audio.diarization_export import (
    to_nvidia_diarization,
    validate_diarization_payload,
)
from dialogue_extractor.domain.enums import Mode
from dialogue_extractor.domain.models import SpeakerTurn, Transcript, Word
from dialogue_extractor.matching.candidate_builder import (
    apply_budget,
    candidates_from_matches,
    candidates_from_segments,
)
from dialogue_extractor.matching.fuzzy_matcher import find_matches

TARGET = "My mind rebels at stagnation"
DURATION = 300.0


def spoken(text: str, start: float, speaker: str, gap: float = 0.32) -> list[Word]:
    """A line of dialogue with plausible word timings."""
    words = []
    t = start
    for token in text.split(" "):
        words.append(Word(text=token, start=round(t, 3), end=round(t + gap * 0.75, 3),
                          probability=0.93))
        t += gap
    return words


@pytest.fixture
def scripted() -> tuple[Transcript, tuple[SpeakerTurn, ...]]:
    """A short two-speaker scene containing the target line twice.

    The second occurrence stands in for a flashback or callback: both must be
    found, because which one is on screen is decided by visual verification, not
    by discarding one here.
    """
    words: list[Word] = []
    words += spoken("Good morning Watson", 5.0, "HOLMES")
    words += spoken("You are up early Holmes", 12.0, "WATSON")
    words += spoken("My mind rebels at stagnation", 20.0, "HOLMES")
    words += spoken("Give me problems give me work", 30.0, "HOLMES")
    words += spoken("You said it yourself once", 120.0, "WATSON")
    words += spoken("My mind rebels at stagnation", 128.0, "HOLMES")
    words += spoken("Indeed I did", 140.0, "HOLMES")

    transcript = Transcript(words=tuple(words), language="en", duration=DURATION)

    turns = (
        SpeakerTurn("HOLMES", 4.5, 8.0),
        SpeakerTurn("WATSON", 11.5, 16.5),
        SpeakerTurn("HOLMES", 19.5, 25.0),
        SpeakerTurn("HOLMES", 29.5, 35.5),
        SpeakerTurn("WATSON", 119.5, 124.0),
        SpeakerTurn("HOLMES", 127.5, 133.0),
        SpeakerTurn("HOLMES", 139.5, 143.0),
    )
    return transcript, turns


class TestModeAPipeline:
    def test_finds_both_occurrences(
        self, scripted: tuple[Transcript, tuple[SpeakerTurn, ...]]
    ) -> None:
        transcript, turns = scripted
        aligned = assign_speakers(transcript.words, turns)

        matches = find_matches(aligned, TARGET)
        assert len(matches) == 2
        assert matches[0].score == pytest.approx(100.0)
        assert matches[1].score == pytest.approx(100.0)

    def test_produces_the_exact_expected_candidates(
        self, scripted: tuple[Transcript, tuple[SpeakerTurn, ...]]
    ) -> None:
        """The regression gate. If this changes, something upstream changed."""
        transcript, turns = scripted
        aligned = assign_speakers(transcript.words, turns)
        candidates = candidates_from_matches(
            find_matches(aligned, TARGET), DURATION, pad=1.5
        )

        assert len(candidates) == 2

        first, second = candidates
        assert first.start == pytest.approx(18.5, abs=0.01)
        assert first.end == pytest.approx(23.02, abs=0.01)
        assert first.speaker_id == "HOLMES"
        assert first.mode is Mode.TARGET
        assert first.match_score == pytest.approx(100.0)

        assert second.start == pytest.approx(126.5, abs=0.01)
        assert second.speaker_id == "HOLMES"

    def test_candidates_are_chronological(
        self, scripted: tuple[Transcript, tuple[SpeakerTurn, ...]]
    ) -> None:
        """Mode A stops at the first that verifies on screen, so order decides
        which occurrence is reported."""
        transcript, turns = scripted
        aligned = assign_speakers(transcript.words, turns)
        candidates = candidates_from_matches(find_matches(aligned, TARGET), DURATION)
        assert candidates[0].start < candidates[1].start

    def test_each_candidate_contains_its_utterance(
        self, scripted: tuple[Transcript, tuple[SpeakerTurn, ...]]
    ) -> None:
        transcript, turns = scripted
        aligned = assign_speakers(transcript.words, turns)
        matches = find_matches(aligned, TARGET)
        candidates = candidates_from_matches(matches, DURATION)

        for candidate, match in zip(candidates, matches, strict=True):
            assert candidate.start <= match.start
            assert candidate.end >= match.end

    def test_wrong_phrase_finds_nothing(
        self, scripted: tuple[Transcript, tuple[SpeakerTurn, ...]]
    ) -> None:
        transcript, turns = scripted
        aligned = assign_speakers(transcript.words, turns)
        assert candidates_from_matches(
            find_matches(aligned, "the curious incident of the dog"), DURATION
        ) == ()

    def test_misheard_target_still_matches(
        self, scripted: tuple[Transcript, tuple[SpeakerTurn, ...]]
    ) -> None:
        """The user may type the line slightly differently from how it was said."""
        transcript, turns = scripted
        aligned = assign_speakers(transcript.words, turns)
        assert find_matches(aligned, "my mind rebels against stagnation")


class TestModeBPipeline:
    def test_enumerates_every_segment(
        self, scripted: tuple[Transcript, tuple[SpeakerTurn, ...]]
    ) -> None:
        transcript, turns = scripted
        aligned = assign_speakers(transcript.words, turns)
        segments = build_segments(aligned)
        candidates = candidates_from_segments(segments, DURATION)

        assert candidates
        assert all(c.mode is Mode.DISCOVERY for c in candidates)

    def test_no_candidate_spans_two_speakers(
        self, scripted: tuple[Transcript, tuple[SpeakerTurn, ...]]
    ) -> None:
        """A candidate must have one diarized speaker to cross-check against
        the face the detector reports as speaking."""
        transcript, turns = scripted
        aligned = assign_speakers(transcript.words, turns)
        candidates = candidates_from_segments(build_segments(aligned), DURATION)

        for candidate in candidates:
            speakers = {w.speaker_id for w in candidate.words if w.speaker_id}
            assert len(speakers) <= 1, f"candidate spans {speakers}"

    def test_match_score_is_absent_in_discovery(
        self, scripted: tuple[Transcript, tuple[SpeakerTurn, ...]]
    ) -> None:
        transcript, turns = scripted
        aligned = assign_speakers(transcript.words, turns)
        candidates = candidates_from_segments(build_segments(aligned), DURATION)
        assert all(c.match_score is None for c in candidates)

    def test_no_words_are_lost_across_candidates(
        self, scripted: tuple[Transcript, tuple[SpeakerTurn, ...]]
    ) -> None:
        transcript, turns = scripted
        aligned = assign_speakers(transcript.words, turns)
        candidates = candidates_from_segments(build_segments(aligned), DURATION)
        assert sum(len(c.words) for c in candidates) == len(aligned)

    def test_budget_truncates_honestly(
        self, scripted: tuple[Transcript, tuple[SpeakerTurn, ...]]
    ) -> None:
        transcript, turns = scripted
        aligned = assign_speakers(transcript.words, turns)
        candidates = candidates_from_segments(build_segments(aligned), DURATION)

        kept, truncated = apply_budget(candidates, max_candidates=2)
        assert len(kept) == 2
        assert truncated is True


class TestCandidateToAsdPayload:
    """The handoff to Phase 5: a candidate must produce a valid payload."""

    def test_payload_validates_and_is_clip_relative(
        self, scripted: tuple[Transcript, tuple[SpeakerTurn, ...]]
    ) -> None:
        transcript, turns = scripted
        aligned = assign_speakers(transcript.words, turns)
        candidate = candidates_from_matches(find_matches(aligned, TARGET), DURATION)[0]

        window = tuple(
            w for w in aligned if candidate.start <= w.start <= candidate.end
        )
        payload = to_nvidia_diarization(window, clip_start=candidate.start)

        validate_diarization_payload(payload)
        assert payload["words"][0]["start"] >= 0.0
        assert payload["words"][-1]["end"] <= candidate.duration + 1.0
        assert all(isinstance(w["speaker_id"], str) for w in payload["words"])


class TestTranscriptRoundTrip:
    def test_pipeline_works_from_serialized_transcript(
        self, scripted: tuple[Transcript, tuple[SpeakerTurn, ...]], tmp_path: Path
    ) -> None:
        """The real flow loads a cached transcript from disk, so matching must
        survive the JSON round-trip unchanged."""
        transcript, turns = scripted

        path = tmp_path / "transcript.json"
        path.write_text(json.dumps(transcript_to_dict(transcript)), encoding="utf-8")
        reloaded = transcript_from_dict(json.loads(path.read_text(encoding="utf-8")))

        direct = find_matches(assign_speakers(transcript.words, turns), TARGET)
        via_disk = find_matches(assign_speakers(reloaded.words, turns), TARGET)

        assert [(m.start, m.end, m.score) for m in direct] == [
            (m.start, m.end, m.score) for m in via_disk
        ]


class TestDeterminism:
    def test_whole_pipeline_is_reproducible(
        self, scripted: tuple[Transcript, tuple[SpeakerTurn, ...]]
    ) -> None:
        transcript, turns = scripted

        def run() -> list[tuple[float, float, str | None]]:
            aligned = assign_speakers(transcript.words, turns)
            candidates = candidates_from_matches(find_matches(aligned, TARGET), DURATION)
            return [(c.start, c.end, c.speaker_id) for c in candidates]

        assert run() == run()
