"""Where the onset constraint places the line's first frame inside a clip.

The constraint is only as good as this number: too late and a correct on-screen
line is rejected, too early and the previous sentence wins again.
"""

from __future__ import annotations

import math
from pathlib import Path

from dialogue_extractor.config.constants import ONSET_TOLERANCE_SECONDS
from dialogue_extractor.domain.enums import Mode
from dialogue_extractor.domain.models import Candidate, Word
from dialogue_extractor.pipeline.mode_a import onset_frame_for
from dialogue_extractor.vision.clip_builder import Clip

FPS = 25.0


def clip(start: float, end: float) -> Clip:
    return Clip(
        path=Path("clip.mp4"),
        audio_path=None,
        start=start,
        end=end,
        clip_start_frame=round(start * FPS),
        fps=FPS,
        size_bytes=1,
    )


def candidate(first_word_start: float, *, start: float, end: float) -> Candidate:
    words = (
        Word("my", first_word_start, first_word_start + 0.2, 0.9, "S1"),
        Word("mind", first_word_start + 0.2, first_word_start + 0.5, 0.9, "S1"),
    )
    return Candidate(
        start=start,
        end=end,
        speaker_id="S1",
        text="my mind",
        mode=Mode.TARGET,
        match_score=100.0,
        words=words,
    )


def test_onset_is_first_word_minus_tolerance_relative_to_clip() -> None:
    # Clip starts at 8.0 s = frame 200. The word starts at 10.0 s, so the onset
    # is 10.0 - tolerance, mapped with the same ceil rule ffmpeg trimming uses.
    c = candidate(10.0, start=8.0, end=12.0)
    expected = math.ceil((10.0 - ONSET_TOLERANCE_SECONDS) * FPS - 1e-9) - 200
    assert onset_frame_for(c, clip(8.0, 12.0)) == expected


def test_onset_never_precedes_the_clip() -> None:
    c = candidate(0.1, start=0.0, end=3.5)
    assert onset_frame_for(c, clip(0.0, 3.5)) == 0


def test_zero_tolerance_lands_on_the_word() -> None:
    c = candidate(10.0, start=8.0, end=12.0)
    assert onset_frame_for(c, clip(8.0, 12.0), tolerance=0.0) == round(2.0 * FPS)


def test_no_words_means_no_constraint() -> None:
    c = Candidate(start=1.0, end=4.5, speaker_id=None, text="", mode=Mode.DISCOVERY)
    assert onset_frame_for(c, clip(1.0, 4.5)) is None
