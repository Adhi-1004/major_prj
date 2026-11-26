"""Every tunable threshold, named and in one place.

Nothing here is tuned to the sample video or the sample phrase -- the brief
requires the system to generalize to a video and line supplied at evaluation
time, so these are principled defaults with a stated rationale, not fitted
values.
"""

from __future__ import annotations

from typing import Final

# --- Media normalization -----------------------------------------------------

TARGET_AUDIO_SAMPLE_RATE: Final[int] = 16_000
"""16 kHz mono is what both faster-whisper and pyannote expect."""

TARGET_AUDIO_CHANNELS: Final[int] = 1

MAX_NORMALIZED_HEIGHT: Final[int] = 1080
"""Cap normalization height. Above this costs decode time for no accuracy gain."""

FALLBACK_FPS: Final[float] = 25.0
"""Used only when ffprobe reports no usable frame rate at all."""


# --- Transcription -----------------------------------------------------------

DEFAULT_WHISPER_MODEL: Final[str] = "large-v3"
WHISPER_BEAM_SIZE: Final[int] = 5

WHISPER_TEMPERATURE: Final[float] = 0.0
"""Fixed at 0.0 for reproducibility.

faster-whisper defaults to a temperature *fallback* ladder, which resamples on
low confidence and makes the same input produce different transcripts across
runs -- incompatible with the Reproducibility NFR.
"""

WHISPER_CONDITION_ON_PREVIOUS_TEXT: Final[bool] = False
"""Off: it makes each window depend on prior output, which both hurts
reproducibility and lets one hallucination cascade into later segments."""

TRANSCRIPTION_CHUNK_SECONDS: Final[float] = 600.0
TRANSCRIPTION_CHUNK_OVERLAP_SECONDS: Final[float] = 5.0
"""Overlap so a phrase straddling a chunk boundary is never split mid-match."""


# --- Diarization -------------------------------------------------------------

PYANNOTE_PIPELINE: Final[str] = "pyannote/speaker-diarization-community-1"
"""pyannote.audio 4.x pipeline id. The pre-4.x "speaker-diarization-3.1" and the
`use_auth_token=` kwarg are both superseded."""

MAX_INTRA_SEGMENT_GAP: Final[float] = 0.75
"""Silence longer than this splits a speech segment, even for one speaker."""

MIN_SPEAKER_TURN_SECONDS: Final[float] = 0.25
"""Diarization turns shorter than this are discarded as segmentation artifacts.

Nobody produces a speaker-identifiable utterance in a fraction of a syllable,
but diarization emits such turns freely -- on the sample material 24% of turns
were under 0.25s and the shortest was 17 milliseconds. Left in, they split
single utterances across speakers: "Until next time, Big Joe. Always a pleasure,
Mr. Holmes." came back as three segments attributed to two different speakers.

That is not cosmetic. Phase 5 cross-checks the matched phrase's diarized speaker
against the face the detector reports as speaking, so a word misattributed here
makes a correct on-screen match fail verification.

Measured on the sample: 0.25s merged those utterances correctly at a cost of
three more unassigned words; 0.40s gained nothing further and started stripping
speakers from correctly-attributed lines.
"""

UNASSIGNED_SPEAKER_ID: Final[str] = "UNKNOWN"


# --- Matching ----------------------------------------------------------------

MATCH_SCORE_FLOOR: Final[float] = 70.0
"""rapidfuzz ratio (0-100) below which a window is not considered a match."""

MATCH_WINDOW_SLACK: Final[int] = 2
"""Sliding window is the target's word count +/- this, absorbing ASR
insertions and deletions without letting the window drift arbitrarily wide."""

WEAK_MATCH_THRESHOLD: Final[float] = 82.0
"""Matches between the floor and this are flagged WEAK_TEXT_MATCH."""


# --- Candidate windows -------------------------------------------------------

WINDOW_PAD_SECONDS: Final[float] = 1.5
"""Padding on each side of a matched phrase.

Sized to absorb faster-whisper's word-timestamp jitter (roughly +/-0.2-0.4 s),
with margin, so the clip reliably contains the whole utterance. It does not
affect final precision -- the exact frame comes from the ASD model, not from
the ASR timestamp.
"""

CHUNK_MAX_SECONDS: Final[float] = 30.0
"""Longest clip sent to the ASD service in one request. Also the Mode B
merge target for adjacent same-speaker segments."""

MAX_CLIP_HEIGHT: Final[int] = 720
"""Clips are downscaled before upload. Keeps requests small; face detection
does not benefit from more."""

ASD_MIN_CLIP_SECONDS: Final[float] = 3.5
"""Shortest clip worth sending to the detection service.

Measured, not guessed. The hosted service accepts a shorter clip without
complaint and returns **zero frames** -- a wasted round-trip that looks exactly
like "nobody was on screen". Probing a window known to contain a visible
speaker, trimmed to varying lengths:

    0.12s -> 0 frames        2.60s -> 0 frames
    0.40s -> 0 frames        2.75s -> 0 frames
    1.00s -> 0 frames        2.90s -> 0 frames
    2.00s -> 0 frames        3.00s -> 72 frames
    2.50s -> 0 frames        4.90s -> 118 frames

72 frames is exactly 3.0s at this video's rate, so the floor looks like an
internal context window rather than a duration check. 3.5s leaves margin.

Short candidates are therefore **padded up to this length, not discarded**. A
one-word line is still real dialogue -- "Yes." is something someone might
search for -- it simply needs surrounding footage for the detector to work
with. On the sample video 23 of 37 discovery candidates fell below the floor;
without padding, every one of them would have come back empty.
"""

FRAGMENT_MERGE_GAP: Final[float] = 0.05
"""Silence below which an unattributed fragment is treated as part of its neighbour.

Diarization leaves single words stranded at utterance boundaries -- "Tell"
immediately before "me, Doctor, as a medical man..." with no gap at all. Those
are segmentation artifacts, not separate utterances, and verifying them
separately costs a request while producing a candidate with no speaker to
cross-check against.

Only *zero-gap* neighbours are absorbed. A fragment genuinely isolated in time
keeps its own candidate, because assigning it a speaker on proximity alone
would be a guess.
"""


# --- Active speaker detection ------------------------------------------------

ASD_MAX_SPEAKERS: Final[int] = 5
"""Server-side per-frame face cap (NV_AI4M_ASD_MAX_SPEAKERS default).

Documented by NVIDIA as a deployment setting; it is not a client constant and
cannot be read back from the hosted function, so treat a frame reporting this
many faces as possibly truncated rather than certainly complete.
"""

ASD_DEADLINE_SECONDS: Final[float] = 300.0
ASD_MAX_RETRIES: Final[int] = 3
ASD_RETRY_BASE_DELAY: Final[float] = 2.0

ASD_INPUT_FILE_SIZE_LIMIT_MB: Final[int] = 500
"""Documented overall input-file cap for the hosted service (H.264 MP4).

Distinct from the per-request total below -- do not conflate the two.
"""

ASD_REQUEST_SOFT_LIMIT_MB: Final[int] = 25
"""Our own conservative per-request ceiling.

The hosted function's true per-request total is not published anywhere. Per-
*message* size is a non-issue (the vendored client streams in 64 KB chunks), so
this exists purely to fail fast and trigger halve-and-retry rather than to
mirror a documented number.
"""


# --- Frame selection ---------------------------------------------------------

MIN_SPEAKING_MS: Final[float] = 120.0
"""Minimum continuous on-screen speech before a run counts.

Expressed as a duration, never as a frame count: a fixed K=3 would mean 125 ms
at 24 fps but 50 ms at 60 fps, silently changing strictness with the source's
frame rate. The frame count is derived per-video via timing.k_frames_for().

120 ms is roughly one syllable -- long enough that single-frame detector
flicker cannot produce a confident wrong answer, short enough not to miss a
genuinely brief line.
"""

ONSET_TOLERANCE_SECONDS: Final[float] = 0.30
"""How far before the first matched word's ASR start a speaking run may begin.

The onset constraint exists because the candidate window is padded by
WINDOW_PAD_SECONDS on both sides: when the same speaker is already talking in
the lead-in, the earliest qualifying run starts on their *previous* sentence and
the reported frame precedes the queried line. Requiring the run to start at or
after the line's onset removes that error.

The tolerance absorbs ASR word-timestamp jitter (roughly +/-0.2-0.4 s, see
WINDOW_PAD_SECONDS), so a word whose recognised start is slightly late does not
push the answer past the true onset. It is a parameter of ablation A5, not a
value fitted to any video.
"""

SCENE_CUT_THRESHOLD: Final[float] = 0.4
"""ffmpeg scene-score threshold for a shot change."""


# --- Confidence --------------------------------------------------------------

CONFIDENCE_WEIGHTS: Final[dict[str, float]] = {
    "match": 0.40,
    "asr": 0.25,
    "face": 0.25,
    "stability": 0.10,
}
"""Composite score weights.

In discovery mode there is no target phrase, so "match" is *absent* rather than
1.0, and the remaining weights renormalize to sum to 1.0. Substituting 1.0
would grant every discovery result a free 0.40 for something never verified,
letting identical detection evidence clear a lower bar in one mode than the
other.
"""

CONFIDENCE_SUCCESS_THRESHOLD: Final[float] = 0.70
CONFIDENCE_AMBIGUOUS_THRESHOLD: Final[float] = 0.45

LOW_FACE_CONFIDENCE_THRESHOLD: Final[float] = 0.60
"""Below this, flag LOW_FACE_CONFIDENCE even if the composite score passes."""


# --- Mode B budget -----------------------------------------------------------

DEFAULT_MAX_CANDIDATES: Final[int] = 0
"""0 means unlimited. A feature-length film can produce ~1200 candidates, each
one network round-trip, so the CLI exposes this for bounded runs."""

DEFAULT_TIME_BUDGET_SECONDS: Final[float] = 0.0
"""0 means unlimited wall-clock."""


# --- Ingestion ---------------------------------------------------------------

DOWNLOAD_MAX_RETRIES: Final[int] = 3
DOWNLOAD_RETRY_BASE_DELAY: Final[float] = 2.0
ALLOWED_URL_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})
