"""Choosing the frame that answers the question.

PURE: no I/O, no third-party imports. This is where "the dialogue is spoken
on-screen at this exact frame" is finally decided, so every rule here is one the
result depends on.

Four conditions must hold together, and each exists because of a specific way
the answer can be wrong:

1. **The face must be the diarized speaker's.**
   The detector reports whether *a* visible face is producing the audio. That is
   not the question. In a six-person shot several faces are tracked and one is
   speaking -- if that one is not the speaker of the matched line, the frame
   shows someone else talking over it. Verified on real footage: a padded window
   around one line contained three speakers' dialogue, and only the cross-check
   distinguishes them.

2. **The run must last a minimum duration.**
   Detectors flicker. A single frame reading `is_speaking` is not evidence of
   speech, and taking it produces a confident, precise, wrong answer. The
   threshold is a duration converted to frames per video, never a fixed count --
   a fixed K would mean 125 ms at 24 fps and 50 ms at 60 fps.

3. **The run must not span a shot cut.**
   Face tracking has no concept of a cut. The person in frame after one may not
   be the person before it, so a run credited across a cut can attribute speech
   to whoever the tracker happened to follow.

4. **Frames must be consecutive.**
   A gap in frame ids means the detector returned nothing for those frames, and
   treating the two sides as continuous assumes something unobserved.

5. **The run must start at or after the line's onset** (when an onset is given).
   The candidate window is padded on both sides. If the same speaker is already
   talking in the lead-in, an unconstrained search returns a frame from their
   *previous* sentence -- the right person, the wrong line.

When no run qualifies, the reason is reported rather than a bare failure -- the
brief asks specifically that uncertain results be explained.
"""

from __future__ import annotations

from dataclasses import dataclass

from dialogue_extractor.config.constants import ASD_MAX_SPEAKERS, MIN_SPEAKING_MS
from dialogue_extractor.domain.enums import AmbiguityReason
from dialogue_extractor.domain.models import ASDFrame, ASDResult, Face, FrameHit
from dialogue_extractor.domain.timing import clip_frame_to_global, frame_to_seconds, k_frames_for


@dataclass(frozen=True, slots=True)
class SelectionOutcome:
    """The chosen frame, or why there wasn't one.

    Diagnostics travel with the answer so a null result can say *which* of the
    four conditions failed, instead of "not on screen" for every cause.
    """

    hit: FrameHit | None
    reason: AmbiguityReason | None = None
    detail: str = ""
    frames_examined: int = 0
    frames_with_face: int = 0
    frames_with_any_speaker: int = 0
    frames_with_target_speaking: int = 0
    longest_run: int = 0
    required_run: int = 1
    speaker_codes_seen: tuple[int, ...] = ()
    max_faces_in_a_frame: int = 0

    @property
    def found(self) -> bool:
        return self.hit is not None


@dataclass(frozen=True, slots=True)
class _Run:
    start_index: int
    frames: tuple[ASDFrame, ...]
    faces: tuple[Face, ...]

    @property
    def length(self) -> int:
        return len(self.frames)


def _qualifying_face(frame: ASDFrame, speaker_code: int | None) -> Face | None:
    """The speaking face belonging to `speaker_code`, if any.

    With no expected speaker (the candidate's words were never attributed), any
    speaking face qualifies -- but the caller is told, so the result can be
    marked less certain rather than presented as a verified match.

    When several faces match, the most confidently detected one wins.
    """
    speaking = [f for f in frame.faces if f.is_speaking]
    if not speaking:
        return None

    if speaker_code is None:
        return max(speaking, key=lambda f: f.face_detection_confidence)

    matching = [f for f in speaking if f.diarized_speaker_id == speaker_code]
    if not matching:
        return None
    return max(matching, key=lambda f: f.face_detection_confidence)


def _find_runs(
    frames: tuple[ASDFrame, ...],
    speaker_code: int | None,
    cut_frames: frozenset[int],
    onset_frame: int | None = None,
) -> list[_Run]:
    """Maximal runs of consecutive frames where the target speaker is speaking.

    Frames before `onset_frame` never join a run: speech there belongs to
    whatever was said before the queried line.
    """
    runs: list[_Run] = []
    current_frames: list[ASDFrame] = []
    current_faces: list[Face] = []
    start_index = 0

    def flush() -> None:
        if current_frames:
            runs.append(
                _Run(
                    start_index=start_index,
                    frames=tuple(current_frames),
                    faces=tuple(current_faces),
                )
            )

    for index, frame in enumerate(frames):
        before_onset = onset_frame is not None and frame.frame_id < onset_frame
        face = None if before_onset else _qualifying_face(frame, speaker_code)

        if face is None:
            flush()
            current_frames, current_faces = [], []
            continue

        breaks_run = current_frames and (
            # A gap in frame ids: the frames between were never observed.
            frame.frame_id != current_frames[-1].frame_id + 1
            # A cut: whoever is in shot afterwards may be someone else entirely.
            or frame.frame_id in cut_frames
        )
        if breaks_run:
            flush()
            current_frames, current_faces = [], []

        if not current_frames:
            start_index = index
        current_frames.append(frame)
        current_faces.append(face)

    flush()
    return runs


def select_frame(
    result: ASDResult,
    speaker_code: int | None,
    *,
    min_speaking_ms: float = MIN_SPEAKING_MS,
    cut_frames: frozenset[int] = frozenset(),
    max_speakers: int = ASD_MAX_SPEAKERS,
    onset_frame: int | None = None,
    match_speaker: bool = True,
) -> SelectionOutcome:
    """Pick the first frame where the target speaker is visibly speaking.

    Args:
        speaker_code: The wire code for the diarized speaker of the matched
            line. None accepts any speaking face, and says so in the outcome.
        cut_frames: Clip-relative frames on which a shot changes.
        onset_frame: Clip-relative frame at which the line begins (already
            widened by the onset tolerance). Runs may not start before it.
            None disables the constraint, which is ablation A5.
        match_speaker: False accepts any speaking face without flagging the
            result -- the speaker-agnostic configuration of ablation A1. The
            production pipeline never sets it.

    Returns:
        The earliest qualifying run's *first* frame -- the question is where the
        dialogue begins to appear, not where it is clearest.
    """
    frames = result.frames
    required = k_frames_for(result.fps, min_speaking_ms)
    expected_speaker = speaker_code
    if not match_speaker:
        speaker_code = None

    codes_seen = tuple(
        sorted(
            {
                face.diarized_speaker_id
                for frame in frames
                for face in frame.faces
                if face.diarized_speaker_id is not None
            }
        )
    )
    frames_with_face = sum(1 for f in frames if f.faces)
    frames_with_any_speaker = sum(1 for f in frames if f.speaking_faces)
    max_faces = max((len(f.faces) for f in frames), default=0)

    def outcome(
        hit: FrameHit | None,
        reason: AmbiguityReason | None,
        detail: str,
        *,
        target_frames: int = 0,
        longest: int = 0,
    ) -> SelectionOutcome:
        """Build an outcome carrying the diagnostics gathered so far.

        A closure rather than a dict splatted into the constructor: the dict
        version type-checked as `dict[str, object]` and quietly gave up on
        verifying any of these fields.
        """
        return SelectionOutcome(
            hit=hit,
            reason=reason,
            detail=detail,
            frames_examined=len(frames),
            frames_with_face=frames_with_face,
            frames_with_any_speaker=frames_with_any_speaker,
            frames_with_target_speaking=target_frames,
            longest_run=longest,
            required_run=required,
            speaker_codes_seen=codes_seen,
            max_faces_in_a_frame=max_faces,
        )

    if not frames:
        return outcome(
            None,
            AmbiguityReason.NO_FACE_DETECTED,
            "The detection service returned no frames for this window.",
        )

    if frames_with_face == 0:
        return outcome(
            None,
            AmbiguityReason.NO_FACE_DETECTED,
            "No face was visible anywhere in this window.",
        )

    runs = _find_runs(frames, speaker_code, cut_frames, onset_frame)
    target_frames = sum(run.length for run in runs)
    longest = max((run.length for run in runs), default=0)

    if not runs:
        spoke_before_onset = onset_frame is not None and any(
            frame.frame_id < onset_frame
            and _qualifying_face(frame, speaker_code) is not None
            for frame in frames
        )
        if spoke_before_onset:
            return outcome(
                None,
                AmbiguityReason.SPOKEN_BEFORE_ONSET,
                (
                    "The expected speaker was seen speaking only before the queried "
                    f"words begin (clip frame {onset_frame}); while the line itself "
                    "is spoken they are not detected speaking."
                ),
                target_frames=target_frames,
                longest=longest,
            )
        if frames_with_any_speaker == 0:
            return outcome(
                None,
                AmbiguityReason.NO_FACE_DETECTED,
                (
                    f"A face was visible in {frames_with_face} frames, but none was "
                    "detected speaking -- consistent with voice-over or an off-camera "
                    "speaker."
                ),
                target_frames=target_frames,
                longest=longest,
            )
        return outcome(
            None,
            AmbiguityReason.SPEAKER_MISMATCH,
            (
                f"Someone was speaking on screen in {frames_with_any_speaker} frames, "
                f"but never the expected speaker (code {speaker_code}). "
                f"Codes present: {list(codes_seen)}."
            ),
            target_frames=target_frames,
            longest=longest,
        )

    qualifying = [run for run in runs if run.length >= required]
    if not qualifying:
        return outcome(
            None,
            AmbiguityReason.SHORT_SPEAKING_RUN,
            (
                f"The longest continuous stretch of the expected speaker speaking was "
                f"{longest} frame(s); {required} are required to rule out detector "
                f"flicker at {result.fps:.3f} fps."
            ),
            target_frames=target_frames,
            longest=longest,
        )

    # Earliest qualifying run, and its first frame: the question is where the
    # dialogue begins to appear.
    run = min(qualifying, key=lambda r: r.frames[0].frame_id)
    first_frame = run.frames[0]
    face = run.faces[0]

    global_frame = clip_frame_to_global(first_frame.frame_id, result.clip_start_frame)
    mean_confidence = sum(f.face_detection_confidence for f in run.faces) / len(run.faces)

    hit = FrameHit(
        global_frame=global_frame,
        timestamp=frame_to_seconds(global_frame, result.fps),
        face=face,
        run_length=run.length,
        mean_face_confidence=mean_confidence,
    )

    reason: AmbiguityReason | None = None
    detail = ""
    if expected_speaker is None and match_speaker:
        reason = AmbiguityReason.SPEAKER_MISMATCH
        detail = (
            "No diarized speaker was available for this line, so any speaking face "
            "was accepted. The identity is unverified."
        )
    elif max_faces >= max_speakers:
        reason = AmbiguityReason.MAX_SPEAKERS_REACHED
        detail = (
            f"{max_faces} faces were reported in a single frame, at or above the "
            f"service's per-frame cap of {max_speakers}; further faces may have been "
            "dropped."
        )
    elif _simultaneous_speakers(run.frames) > 1:
        reason = AmbiguityReason.MULTIPLE_SIMULTANEOUS_SPEAKERS
        detail = "More than one face was speaking at the chosen frame."

    return outcome(hit, reason, detail, target_frames=target_frames, longest=longest)


def _simultaneous_speakers(frames: tuple[ASDFrame, ...]) -> int:
    """Most faces reported speaking in any one frame of the run."""
    return max((len(f.speaking_faces) for f in frames), default=0)
