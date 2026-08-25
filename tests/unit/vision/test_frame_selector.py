"""Frame selection -- where "on-screen at this exact frame" is decided.

Every rule tested here exists because of a specific way the answer can be wrong,
and each wrong answer is *plausible*: a real frame, from the right video, of a
real person, at the wrong moment or of the wrong speaker. None of them look like
failures, which is why they need pinning.
"""

from __future__ import annotations

import pytest

from dialogue_extractor.domain.enums import AmbiguityReason
from dialogue_extractor.domain.models import ASDFrame, ASDResult, BoundingBox, Face
from dialogue_extractor.vision.frame_selector import select_frame

FPS = 25.0  # k_frames_for(25, 120ms) == 3


def face(
    *,
    speaking: bool = True,
    speaker: int | None = 1,
    face_id: int = 1,
    confidence: float = 0.9,
) -> Face:
    return Face(
        face_id=face_id,
        bbox=BoundingBox(0.1, 0.1, 0.2, 0.3),
        is_speaking=speaking,
        face_detection_confidence=confidence,
        diarized_speaker_id=speaker,
    )


def result(
    frames: list[ASDFrame], *, clip_start_frame: int = 0, fps: float = FPS
) -> ASDResult:
    return ASDResult(
        frames=tuple(frames), clip_start_frame=clip_start_frame, fps=fps, provider="test"
    )


def pattern(spec: str, *, speaker: int | None = 1, start_id: int = 0) -> list[ASDFrame]:
    """Build frames from a compact string.

    ``S`` speaking, ``.`` visible but silent, ``_`` no face at all.
    """
    frames = []
    for offset, symbol in enumerate(spec):
        frame_id = start_id + offset
        if symbol == "_":
            frames.append(ASDFrame(frame_id=frame_id))
        elif symbol == ".":
            frames.append(ASDFrame(frame_id, (face(speaking=False, speaker=speaker),)))
        elif symbol == "S":
            frames.append(ASDFrame(frame_id, (face(speaking=True, speaker=speaker),)))
        else:
            raise ValueError(f"unknown symbol {symbol!r}")
    return frames


class TestCleanDetection:
    def test_finds_a_clear_run(self) -> None:
        outcome = select_frame(result(pattern("...SSSSSS...")), speaker_code=1)
        assert outcome.found
        assert outcome.hit is not None
        assert outcome.hit.global_frame == 3

    def test_returns_the_first_frame_of_the_run(self) -> None:
        """The question is where the dialogue begins to appear, not where the
        detector is most confident."""
        frames = pattern("..SSSSSS..")
        outcome = select_frame(result(frames), speaker_code=1)
        assert outcome.hit is not None
        assert outcome.hit.global_frame == 2

    def test_timestamp_follows_from_the_frame(self) -> None:
        outcome = select_frame(result(pattern("SSSS")), speaker_code=1)
        assert outcome.hit is not None
        assert outcome.hit.timestamp == pytest.approx(0.0)

    def test_run_length_is_recorded(self) -> None:
        outcome = select_frame(result(pattern("..SSSSS..")), speaker_code=1)
        assert outcome.hit is not None
        assert outcome.hit.run_length == 5

    def test_no_ambiguity_flag_on_a_clean_result(self) -> None:
        outcome = select_frame(result(pattern("..SSSSS..")), speaker_code=1)
        assert outcome.reason is None


class TestFlickerRejection:
    """A single frame reading is_speaking is not evidence of speech. Taking it
    produces a confident, precise, wrong answer."""

    def test_single_frame_flicker_is_rejected(self) -> None:
        outcome = select_frame(result(pattern("...S...")), speaker_code=1)
        assert not outcome.found
        assert outcome.reason is AmbiguityReason.SHORT_SPEAKING_RUN

    def test_two_frames_is_still_too_short_at_25fps(self) -> None:
        outcome = select_frame(result(pattern("...SS...")), speaker_code=1)
        assert not outcome.found

    def test_exactly_the_required_length_qualifies(self) -> None:
        outcome = select_frame(result(pattern("...SSS...")), speaker_code=1)
        assert outcome.found
        assert outcome.required_run == 3

    def test_flicker_before_a_real_run_is_skipped(self) -> None:
        """The lone S at index 1 must not be mistaken for the answer."""
        outcome = select_frame(result(pattern(".S..SSSSS")), speaker_code=1)
        assert outcome.hit is not None
        assert outcome.hit.global_frame == 4

    def test_the_reason_explains_the_threshold(self) -> None:
        outcome = select_frame(result(pattern("...SS...")), speaker_code=1)
        assert "2 frame" in outcome.detail
        assert "3 are required" in outcome.detail


class TestSpeakerCrossCheck:
    """The detector reports whether *a* face is speaking. That is not the
    question. Observed on real footage: a padded window around one line held
    three speakers' dialogue."""

    def test_the_wrong_speaker_is_not_accepted(self) -> None:
        outcome = select_frame(result(pattern("SSSSSS", speaker=2)), speaker_code=1)
        assert not outcome.found
        assert outcome.reason is AmbiguityReason.SPEAKER_MISMATCH

    def test_the_right_speaker_in_a_crowd_is_found(self) -> None:
        """Six faces, one of them ours, one of them talking over us."""
        frames = [
            ASDFrame(
                i,
                (
                    face(speaking=(i >= 3), speaker=1, face_id=1),
                    face(speaking=True, speaker=2, face_id=2),
                    face(speaking=False, speaker=None, face_id=3),
                ),
            )
            for i in range(10)
        ]
        outcome = select_frame(result(frames), speaker_code=1)
        assert outcome.hit is not None
        assert outcome.hit.global_frame == 3
        assert outcome.hit.face.diarized_speaker_id == 1

    def test_another_speaker_talking_first_does_not_win(self) -> None:
        """Exactly the real case: the previous line's speaker occupies the
        lead-in padding."""
        frames = [
            ASDFrame(i, (face(speaking=True, speaker=2, face_id=2),)) for i in range(5)
        ] + [
            ASDFrame(i, (face(speaking=True, speaker=1, face_id=1),)) for i in range(5, 12)
        ]
        outcome = select_frame(result(frames), speaker_code=1)
        assert outcome.hit is not None
        assert outcome.hit.global_frame == 5

    def test_mismatch_reason_names_the_codes_present(self) -> None:
        outcome = select_frame(result(pattern("SSSSSS", speaker=7)), speaker_code=1)
        assert "7" in outcome.detail

    def test_unattributed_line_accepts_any_speaker_but_flags_it(self) -> None:
        """Better than refusing outright, but the identity is unverified and the
        result must say so."""
        outcome = select_frame(result(pattern("SSSSSS", speaker=2)), speaker_code=None)
        assert outcome.found
        assert outcome.reason is AmbiguityReason.SPEAKER_MISMATCH
        assert "unverified" in outcome.detail


class TestNoOneOnScreen:
    def test_no_faces_at_all(self) -> None:
        outcome = select_frame(result(pattern("_____")), speaker_code=1)
        assert not outcome.found
        assert outcome.reason is AmbiguityReason.NO_FACE_DETECTED

    def test_faces_but_nobody_speaking_is_voice_over(self) -> None:
        """The distinction the whole brief is built around."""
        outcome = select_frame(result(pattern("........")), speaker_code=1)
        assert not outcome.found
        assert outcome.reason is AmbiguityReason.NO_FACE_DETECTED
        assert "voice-over" in outcome.detail

    def test_empty_response(self) -> None:
        outcome = select_frame(result([]), speaker_code=1)
        assert not outcome.found
        assert outcome.reason is AmbiguityReason.NO_FACE_DETECTED


class TestSpeakerEntersFrameLate:
    """The design document's edge case: return the first frame where "dialogue
    in progress" AND "speaker visible and speaking" are both true."""

    def test_speaker_absent_then_present(self) -> None:
        frames = [ASDFrame(i) for i in range(5)] + [
            ASDFrame(i, (face(speaking=True, speaker=1),)) for i in range(5, 12)
        ]
        outcome = select_frame(result(frames), speaker_code=1)
        assert outcome.hit is not None
        assert outcome.hit.global_frame == 5

    def test_speaker_visible_but_silent_then_speaks(self) -> None:
        outcome = select_frame(result(pattern("....SSSSS")), speaker_code=1)
        assert outcome.hit is not None
        assert outcome.hit.global_frame == 4


class TestShotCuts:
    """Face tracking has no concept of a cut. A run credited across one can
    attribute speech to whoever the tracker happened to follow."""

    def test_a_run_is_not_credited_across_a_cut(self) -> None:
        outcome = select_frame(
            result(pattern("SSSSSS")), speaker_code=1, cut_frames=frozenset({2})
        )
        assert outcome.hit is not None
        # Frames 0-1 are too short; the run resumes at the cut.
        assert outcome.hit.global_frame == 2

    def test_a_cut_can_disqualify_a_short_run_entirely(self) -> None:
        outcome = select_frame(
            result(pattern("SSSS")), speaker_code=1, cut_frames=frozenset({2})
        )
        assert not outcome.found
        assert outcome.reason is AmbiguityReason.SHORT_SPEAKING_RUN

    def test_no_cuts_leaves_the_run_intact(self) -> None:
        outcome = select_frame(result(pattern("SSSSSS")), speaker_code=1)
        assert outcome.hit is not None
        assert outcome.hit.run_length == 6


class TestFrameGaps:
    def test_a_gap_in_frame_ids_breaks_the_run(self) -> None:
        """Frames the detector never reported cannot be assumed continuous."""
        frames = [
            ASDFrame(0, (face(),)),
            ASDFrame(1, (face(),)),
            ASDFrame(9, (face(),)),  # gap
            ASDFrame(10, (face(),)),
        ]
        outcome = select_frame(result(frames), speaker_code=1)
        assert not outcome.found
        assert outcome.longest_run == 2

    def test_contiguous_after_a_gap_can_still_qualify(self) -> None:
        frames = [
            ASDFrame(0, (face(),)),
            ASDFrame(9, (face(),)),
            ASDFrame(10, (face(),)),
            ASDFrame(11, (face(),)),
            ASDFrame(12, (face(),)),
        ]
        outcome = select_frame(result(frames), speaker_code=1)
        assert outcome.hit is not None
        assert outcome.hit.global_frame == 9


class TestClipOffset:
    """Invariant 1, at the point the answer is produced."""

    def test_offset_is_applied_to_the_result(self) -> None:
        outcome = select_frame(
            result(pattern("..SSSSS"), clip_start_frame=897), speaker_code=1
        )
        assert outcome.hit is not None
        assert outcome.hit.global_frame == 897 + 2

    def test_timestamp_uses_the_global_frame(self) -> None:
        outcome = select_frame(
            result(pattern("SSSS"), clip_start_frame=250, fps=25.0), speaker_code=1
        )
        assert outcome.hit is not None
        assert outcome.hit.timestamp == pytest.approx(10.0)


class TestFrameRateScaling:
    """The smoothing rule is a duration, so it must behave the same at any fps."""

    @pytest.mark.parametrize(
        ("fps", "required"), [(23.976, 3), (25.0, 3), (30.0, 4), (60.0, 7)]
    )
    def test_required_run_scales(self, fps: float, required: int) -> None:
        outcome = select_frame(result(pattern("S" * 20), fps=fps), speaker_code=1)
        assert outcome.required_run == required

    @pytest.mark.parametrize("fps", [23.976, 25.0, 30.0, 60.0])
    def test_a_fixed_duration_of_speech_qualifies_at_every_rate(self, fps: float) -> None:
        """200 ms of speech is a real utterance whatever the frame rate."""
        frame_count = max(1, round(0.2 * fps))
        spec = "." * 3 + "S" * frame_count + "." * 3
        assert select_frame(result(pattern(spec), fps=fps), speaker_code=1).found

    @pytest.mark.parametrize("fps", [23.976, 25.0, 30.0, 60.0])
    def test_a_fixed_duration_of_flicker_is_rejected_at_every_rate(
        self, fps: float
    ) -> None:
        """30 ms is flicker whatever the frame rate. A fixed K=3 would accept
        this at 60 fps and reject it at 24."""
        frame_count = max(1, round(0.03 * fps))
        spec = "." * 3 + "S" * frame_count + "." * 3
        assert not select_frame(result(pattern(spec), fps=fps), speaker_code=1).found


class TestSimultaneousSpeakers:
    def test_two_speaking_faces_is_flagged(self) -> None:
        frames = [
            ASDFrame(
                i,
                (
                    face(speaking=True, speaker=1, face_id=1),
                    face(speaking=True, speaker=2, face_id=2),
                ),
            )
            for i in range(6)
        ]
        outcome = select_frame(result(frames), speaker_code=1)
        assert outcome.found
        assert outcome.reason is AmbiguityReason.MULTIPLE_SIMULTANEOUS_SPEAKERS

    def test_a_crowded_frame_is_flagged_as_possibly_truncated(self) -> None:
        """At the service's per-frame cap, faces beyond it may be missing."""
        frames = [
            ASDFrame(
                i,
                tuple(
                    face(speaking=(n == 0), speaker=1 if n == 0 else None, face_id=n)
                    for n in range(5)
                ),
            )
            for i in range(6)
        ]
        outcome = select_frame(result(frames), speaker_code=1, max_speakers=5)
        assert outcome.found
        assert outcome.reason is AmbiguityReason.MAX_SPEAKERS_REACHED


class TestFaceSelection:
    def test_the_most_confident_matching_face_wins(self) -> None:
        frames = [
            ASDFrame(
                i,
                (
                    face(speaking=True, speaker=1, face_id=1, confidence=0.55),
                    face(speaking=True, speaker=1, face_id=2, confidence=0.95),
                ),
            )
            for i in range(6)
        ]
        outcome = select_frame(result(frames), speaker_code=1)
        assert outcome.hit is not None
        assert outcome.hit.face.face_id == 2

    def test_mean_confidence_is_over_the_run(self) -> None:
        frames = [
            ASDFrame(i, (face(speaking=True, speaker=1, confidence=c),))
            for i, c in enumerate([0.8, 0.9, 1.0, 0.9])
        ]
        outcome = select_frame(result(frames), speaker_code=1)
        assert outcome.hit is not None
        assert outcome.hit.mean_face_confidence == pytest.approx(0.9)


class TestDiagnostics:
    def test_counts_are_reported(self) -> None:
        outcome = select_frame(result(pattern("_..SSSSS.")), speaker_code=1)
        assert outcome.frames_examined == 9
        assert outcome.frames_with_face == 8
        assert outcome.frames_with_any_speaker == 5
        assert outcome.frames_with_target_speaking == 5

    def test_speaker_codes_seen_are_reported(self) -> None:
        frames = [
            ASDFrame(i, (face(speaker=1, face_id=1), face(speaker=3, face_id=2)))
            for i in range(4)
        ]
        outcome = select_frame(result(frames), speaker_code=1)
        assert outcome.speaker_codes_seen == (1, 3)

    def test_every_failure_carries_an_explanation(self) -> None:
        for spec in ("_____", ".....", "...S...", "SSSSSS"):
            outcome = select_frame(
                result(pattern(spec, speaker=9)), speaker_code=1
            )
            if not outcome.found:
                assert outcome.reason is not None
                assert outcome.detail, f"no explanation for {spec!r}"


class TestDeterminism:
    def test_repeated_calls_agree(self) -> None:
        frames = pattern(".S..SSSSS.SSS")
        first = select_frame(result(frames), speaker_code=1)
        second = select_frame(result(frames), speaker_code=1)
        assert first.hit is not None and second.hit is not None
        assert first.hit.global_frame == second.hit.global_frame
