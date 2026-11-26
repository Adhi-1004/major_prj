# Stage 4: Proposed Methodology and Mathematical Model

**Builds on:** Stage 3 (gap G1–G4, objectives O1–O5, contributions C1–C6, hypotheses H1–H7).
**Source of truth:** every rule and constant below is taken from the code in `src/dialogue_extractor/`. Each subsection names its file, so the paper describes exactly what runs.
**Status:** draft for author approval. This document gives the technical content and notation. The final manuscript prose is written in Stage 6.

---

## 0. Change made in this stage (approved in Stage 3)

**Onset constraint (contribution C2, ablation A5) is implemented.**

| File | Change |
|---|---|
| `vision/frame_selector.py` | New argument `onset_frame` (clip-relative). Frames before it never join a speaking run. New failure reason when the expected speaker spoke *only* before the onset. |
| `pipeline/mode_a.py` | New `onset_frame_for(candidate, clip)`: first word's ASR start − tolerance, mapped to a clip frame. `verify_candidate`, `mode_a.run` and `mode_b.run` take `onset_constraint` (default on). |
| `config/constants.py` | `ONSET_TOLERANCE_SECONDS = 0.30`, documented as absorbing word-timestamp jitter. |
| `domain/enums.py` | New `AmbiguityReason.SPOKEN_BEFORE_ONSET`. |
| `pipeline/orchestrator.py`, `api/schemas.py`, `main.py` | `onset_constraint` request field; CLI flag `--no-onset-constraint` for the ablation. |
| `frontend/src/api/types.ts`, `frontend/src/model/labels.ts` | Label for the new reason. |
| `tests/unit/vision/test_frame_selector.py` | 8 new tests (`TestOnsetConstraint`). |
| `tests/unit/pipeline/test_onset_frame.py` | 4 new tests for the onset mapping. |

**Test status:** 696 passed, 0 failed, across the affected modules: frame selection, pipeline, API, matching, domain, and the offline Mode A/B integration tests. A full unit run shows 40 failures, all caused by heavy packages that were not installed in the isolated test environment (torch, grpc, yt-dlp, faster-whisper). None is in a file this change touched. **Please run the full suite in your normal environment before committing.** Nothing has been committed.

**Effect on earlier results:** the constraint is on by default, so the frame indices reported earlier (frame 983 and the determinism list) may change. Stage 5 re-measures everything, and the paper will report only the new runs.

---

## 1. Problem formulation and system model (Section III of the paper)

### 1.1 Notation

| Symbol | Meaning | Unit / type |
|---|---|---|
| V | Input video (URL or local file) | — |
| r | Frame rate of the normalized (constant-frame-rate) video | frames/s |
| F | Number of frames; frames indexed f ∈ {0,…,F−1} | — |
| T | Duration of the video | s |
| t(f) | Presentation time of frame f | s |
| 𝒲 = (w₁,…,w_N) | Word sequence from ASR | — |
| sᵢ, eᵢ | Start and end time of word wᵢ | s |
| πᵢ | ASR probability of word wᵢ | [0,1] |
| 𝒰 = {u} | Diarization turns, each (label ℓ_u, start u_s, end u_e) | — |
| σ(wᵢ) | Speaker label assigned to wᵢ (may be ∅) | label |
| q | Query: a dialogue line (optional) | string |
| n | Token length of the normalized query | tokens |
| m = w_{a:b} | A transcript match of q (contiguous words) | — |
| ρ(x,y) | Normalized indel similarity | [0,100] |
| σ_q | Majority speaker of the match | label |
| [a,b] | Candidate (clip) window | s |
| c | First source frame of the clip, c = ⌈a·r⌉ | frame |
| j | Clip-relative frame index; global f = c + j | frame |
| Φ(j) | Faces reported by the ASD service at clip frame j | set |
| a(φ), κ(φ), λ(φ) | Speaking flag, detection confidence and associated speaker code of face φ | {0,1}, [0,1], ℤ∪{∅} |
| ψ | Speaker codec: label → integer (ψ(ℓ) ≥ 1; 0 reserved for "unattributed") | bijection |
| τ | Minimum speaking duration | ms |
| k | Minimum run length in frames | frames |
| 𝒦 | Clip-relative frames on which a shot cut begins | set |
| j₀ | Clip-relative onset frame of the line (onset constraint) | frame |
| ε | Onset tolerance | s |
| f* | Reported frame | frame |
| C | Composite confidence | [0,1] |

### 1.2 Assumptions (stated in the paper)

| # | Assumption | Why it is needed |
|---|---|---|
| A1 | One video and one primary audio track per request | Scope; dubbing is not handled |
| A2 | Audio and video in the source are synchronized | Desync is not corrected |
| A3 | English speech by default (`language="en"`) | Thresholds are not tuned per language |
| A4 | All learned components are used pretrained, without fine-tuning | No training data for the task |
| A5 | "First frame where the line appears on screen" = the first frame of the earliest qualifying speaking run that begins no earlier than the line's onset (minus tolerance ε) | Makes an otherwise ambiguous phrase objective and reproducible |
| A6 | The hosted detector processes at most 5 faces per frame | A server-side default that cannot be read back from the client; treated as a source of ambiguity |
| A7 | Network access to the hosted ASD service and to the model hubs | A real dependency, stated rather than hidden |

### 1.3 Input / output

- **Input:** V, and optionally q. Tunable parameters: θ_m, p, τ, the scene-cut toggle, the onset-constraint toggle, and budgets (N_max, T_max).
- **Output, target mode:** one record (status, f*, t(f*), recognized text, speaker label, C with its per-term breakdown, frame image, reason).
  - Status ∈ {success, ambiguous, not_on_screen, not_found, error}.
- **Output, discovery mode:** three lists (on screen, ambiguous, not on screen), the processed and total counts, a `truncated` flag, and per-candidate errors.

### 1.4 Formal definitions

**(1) Time–frame invariant.** It holds only after forced constant-frame-rate normalization:

  t(f) = f / r

**(2) Clip offset invariant.** For a clip trimmed at time a:

  c = ⌈a·r⌉,  f = c + j

**(3) Speaker-consistent speaking predicate** at clip frame j, for target speaker code z = ψ(σ_q):

  S(j) = ∃ φ ∈ Φ(j) : a(φ) = 1 ∧ λ(φ) = z

**(4) Persistence threshold.** A duration converted per video:

  k = max(1, ⌊τ·r / 1000 + ½⌋)

With τ = 120 ms, this gives k = 3 at 23.976 and 25 fps, 4 at 30 fps, and 7 at 60 fps. These values are verified by the existing `TestFrameRateScaling` tests.

**(5) Onset frame**, with ε = 0.30 s:

  j₀ = max(0, ⌈max(a, s_a − ε)·r⌉ − c)

where s_a is the start time of the first matched word.

**(6) Qualifying runs.** A run R = (j, j+1, …, j+L−1) qualifies if all of the following hold:
- (i) S(j+i) for all i < L;
- (ii) the frame ids are consecutive;
- (iii) {j+1, …, j+L−1} ∩ 𝒦 = ∅;
- (iv) j ≥ j₀;
- (v) L ≥ k.

**(7) Answer:**

  f* = c + min{ j : (j, …, j+L−1) is a qualifying run }

If the set is empty, the line is *not on screen*, with the reason given by the first failed condition (§2.7).

**(8) Discovery.** Equations (3)–(7) are evaluated independently for every candidate built from the speech segments, with σ_q replaced by the segment's speaker.

### 1.5 Requirements, constraints and design objectives

| Kind | Item |
|---|---|
| Functional | Accept a URL or file and an optional line. Return f*, the time, the text and the image. Give every line a three-way verdict. Return a structured error on any failure. |
| Non-functional | Deterministic output. Robust to variable frame rate, resolution and codec. Visual cost bounded by the number of matches. Explainable confidence. No secrets in output. |
| Constraints (measured) | The hosted ASD returns zero frames for clips ≲ 3 s, so D_min = 3.5 s. A diarization payload that arrives after video produces wrong associations, so diarization must be sent first. Wire sentinel: −1 means unassociated, 0 is a valid speaker. 5-face cap. Per-request deadline 300 s. |
| Design objectives | O1–O5 from Stage 3 |

---

## 2. Proposed methodology (Section IV of the paper)

### 2.1 Overall architecture *(Figure 1: pipeline; Figure 3: detailed control flow)*

The design has three layers:
1. a **per-video audio layer** (acquisition → normalization → ASR → diarization → alignment), run once and cached;
2. a **per-candidate visual layer** (clip → ASD → frame selection → confidence), run once per candidate window;
3. a **mode controller** (target mode: early exit; discovery mode: budgets and buckets).

The orchestrator depends only on abstract interfaces (ports). A composition root binds the concrete implementations: `interfaces/ports.py`, `composition.py`, `pipeline/orchestrator.py`.

### 2.2 Acquisition and preprocessing (`ingestion/`)

- **URL validation.** Only http/https are accepted. Private, loopback and internal hosts are refused, including hosts that resolve to non-public addresses. A local file is accepted as a fallback.
- **Normalization in one FFmpeg pass:**
  - video: H.264 (CRF 18), yuv420p, `-fps_mode cfr -r r̂`, where r̂ is the measured average rate; height ≤ 1080; `+faststart`;
  - audio: PCM 16-bit, 16 kHz mono.

  The output is probed again, and the *measured* r is used everywhere, which is what makes eq. (1) hold.
- **Caching.** The download and normalization are cached together, keyed by the canonical URL and the normalization settings. Transcripts are cached by SHA-256(audio) together with the model, beam size, temperature and quantization. Diarization is cached by SHA-256(audio) together with the pipeline id.

### 2.3 Speech pipeline (`audio/`)

- **ASR.** Whisper large-v3 (faster-whisper / CTranslate2), word timestamps on. Determinism is enforced: temperature 0, beam 5, `condition_on_previous_text = False`, VAD filter on. Quantization is int8/float16 on GPU and int8 on CPU, and it is part of the cache key: the same audio gave 325 vs 337 words depending on quantization.
- **Diarization.** pyannote `speaker-diarization-community-1`. Turns shorter than δ_turn = 0.25 s are discarded. Measured on the sample: 24% of turns were below 0.25 s, the shortest 17 ms.
- **Alignment (eq. 9):**

  σ(wᵢ) = argmax over u ∈ 𝒰 of | [sᵢ, eᵢ] ∩ [u_s, u_e] |

  Ties go to the earlier turn. A word with no overlap gets ∅ and is **kept**, so a line that straddles a diarization gap can still be matched.
- **Segmentation.** A new segment starts at a speaker change or at a gap > 0.75 s.
- **Fragment absorption (discovery only).** A single word with no speaker is merged into an adjacent attributed segment only if the gap is ≤ 0.05 s. Otherwise it keeps its own candidate.

### 2.4 Query matching (`matching/`)

- **Normalization N(·).** Case folding; punctuation and typographic-quote folding; numeral handling. One word may expand into several tokens.
- **Search space.** For each start index a, the window w_{a:b} is grown word by word. It is scored when its token count lies in [max(1, n−δ), n+δ], with δ = 2.
- **Score (eq. 10):**

  ρ(x, y) = 100 · ( 1 − d_ID(x, y) / (|x| + |y|) )

  where d_ID is the character insertion/deletion distance (the RapidFuzz `ratio`). This measure is **order-sensitive** by design: token-set measures would score a scrambled sentence as equal to the original.
- **Acceptance and suppression.** Keep windows with ρ ≥ θ_m = 70. Sort them by (−ρ, start, length) and greedily drop any window that overlaps a kept one. Report the survivors in time order; repeated occurrences are all kept.
- **Match speaker (eq. 11).** The majority label among the match's words, ignoring ∅, with ties going to the earliest speaker:

  σ_q = argmax over ℓ of |{ i ∈ [a, b) : σ(wᵢ) = ℓ }|

### 2.5 Candidate construction (`matching/candidate_builder.py`)

- **Target mode.**
  1. Pad: [a, b] = [max(0, s_a − p), min(T, e_b + p)], with p = 1.5 s.
  2. Widen to D_min = 3.5 s around the centre; at the end of the video, shift the window back rather than shorten it.
  3. Cap at D_max = 30 s, anchored on the utterance onset; leftover room goes to lead-in first.
  4. Drop windows wholly contained in an earlier one.
- **Discovery mode.** Merge adjacent same-speaker segments while the span stays ≤ D_max. Never merge across a change of speaker. Then apply the same widening, capping and de-duplication.

### 2.6 Speaker-consistent ASD request (`vision/clip_builder.py`, `audio/diarization_export.py`, `vision/providers/nvidia_asd.py`)

1. **Exact trim.** Re-encode rather than stream-copy, since stream copy cuts on keyframes. Keep CFR at r, height ≤ 720, `+faststart` (verified by reading the MP4 atom order). Record c = ⌈a·r⌉.
2. **Codec ψ.** Number the speakers present in the window from 1 in first-seen order. Emit them as `speaker_<ψ(ℓ)>`, because the service parses `\d+` from the label. 0 is reserved for words with no speaker.
3. **Diarization payload.** Words inside the window, with times rebased to the clip start, in seconds.
4. **Stream order.** Config, then **all** diarization batches (100 words each), then the video chunks (64 KB), then the audio chunks.
5. **Response mapping.** For every clip frame j, the faces (bbox, a, κ, λ). Wire λ < 0 maps to ∅; λ = 0 is kept as a speaker.
6. **Retry.** Only for UNAVAILABLE and DEADLINE_EXCEEDED: 3 attempts with exponential back-off. Auth, invalid input and quota errors are not retried.

### 2.7 Frame selection (`vision/frame_selector.py`): decision rule and failure taxonomy

The selector evaluates eqs. (3)–(7). When several faces qualify in one frame, the one with the highest κ is used. When no run qualifies, the first matching row below gives the reason:

| Order | Condition | Reported reason | Meaning |
|---|---|---|---|
| 1 | No frames returned | `no_face_detected` | Empty response, e.g. clip too short |
| 2 | No face in any frame | `no_face_detected` | Nobody visible |
| 3 | The target spoke only before j₀ | `spoken_before_onset` | The previous sentence, not this line *(new)* |
| 4 | Faces visible, none speaking | `no_face_detected` (detail: "consistent with voice-over") | The voice-over signature |
| 5 | Someone else speaking | `speaker_mismatch` | A different visible person |
| 6 | Runs exist, all shorter than k | `short_speaking_run` | Flicker, or too brief |

When a run qualifies, the result is still flagged as *ambiguous* in three cases:
- z is undefined, so any speaking face was accepted (`speaker_mismatch`, "identity unverified");
- max |Φ(j)| ≥ 5 (`max_speakers_reached`);
- more than one face is speaking in the run (`multiple_simultaneous_speakers`).

### 2.8 Confidence model (`pipeline/confidence.py`, `pipeline/result_builder.py`)

Terms, each clamped to [0, 1]:
- x_match = ρ / 100 (target mode only);
- x_asr = the mean πᵢ over the matched words;
- x_face = the mean κ over the run's selected faces;
- x_stab = min(1, L / 2k) (eq. 12).

Weights: w = (0.40, 0.25, 0.25, 0.10).

**Eq. 13:**

  C = Σ_{i∈A} wᵢ·xᵢ / Σ_{i∈A} wᵢ,  with A = {asr, face, stab} ∪ {match if q is given}

Status:
- C ≥ 0.70 → success;
- 0.45 ≤ C < 0.70 → ambiguous, with the weakest term named (match < 82, face < 0.60, stability < 0.5);
- otherwise → not on screen.

A caveat raised during selection (§2.7) downgrades a success to ambiguous.

*Honest note for the paper:* the weights and bands are set by design, not learned or calibrated. Stage 5 will report how C relates to correctness, so the reader can judge them.

### 2.9 Algorithms (pseudocode for the paper)

**Algorithm 1: Target-mode localization**
```
Input: V, q, parameters (θm, δ, p, Dmin, Dmax, τ, ε)
1  M ← NormalizeCFR(V);  r ← M.fps                       ▹ cached by canonical URL
2  W ← Align(ASR(M.audio), Diarize(M.audio))              ▹ eq. 9
3  Q ← Match(W, q, θm, δ) in time order                   ▹ eqs. 10–11
4  if Q = ∅ then return NOT_FOUND
5  E ← ∅
6  for each m ∈ Q do
7      [a,b] ← Window(m, p, Dmin, Dmax);  (clip, c) ← Trim(M.video, a, b)
8      ψ ← Codec(words in [a,b]);  D ← Serialize(words in [a,b], ψ, a)
9      R ← ASD(config, D, clip)                           ▹ diarization before media
10     j0 ← Onset(m, a, c, r, ε);  K ← Cuts(clip)         ▹ eq. 5
11     h ← Select(R, ψ(σm), k(τ,r), K, j0)                ▹ eqs. 3–7
12     if h found then
13         f* ← c + h.j;  return (f*, f*/r, Confidence(m,h), Image(f*))
14     E ← E ∪ {h.reason}
15 return NOT_ON_SCREEN with most informative reason in E
```

**Algorithm 2: Discovery sweep**
```
Input: V, parameters, budget (Nmax, Tmax)
1  lines 1–2 of Algorithm 1
2  C ← Candidates(Segments(W))                            ▹ absorb fragments, merge same speaker
3  for i, cand in enumerate(C) while within budget do
4      outcome ← Verify(cand)  (lines 7–11 of Alg. 1); on ASD error record and continue
5      if not found: add to NOT_ON_SCREEN with reason and line start time
6      else: score; add to ON_SCREEN if C ≥ 0.70 else AMBIGUOUS
7  if processed ≥ 3 and no window returned any speaker association: raise DiarizationDropped
8  return lists, processed/total, truncated flag, errors
```

### 2.10 Computational complexity and cost model

| Step | Time complexity | Notes |
|---|---|---|
| Normalization | O(F) decode + encode | Dominant first-run cost (about 100 s on the sample); cached |
| ASR, diarization | O(T) model inference | GPU-bound; cached |
| Alignment | O(N·\|𝒰\|) worst case; close to O(N + \|𝒰\|) in practice | Turns sorted; scan breaks once turns start after the word |
| Matching | O(N·(n+δ)·ℓ_q) | ℓ_q = query length in characters; bit-parallel indel computation |
| Overlap suppression | O(M log M + M·M′) | M = raw windows above the floor, M′ = kept matches |
| Frame selection (per clip) | O(J·\|Φ\|) | J ≤ D_max·r frames, \|Φ\| ≤ 5 |
| ASD requests | Target: ≤ \|Q\| (stops at the first verified one); Discovery: \|C\| | The cost driver that counts |

**Visual cost (eq. 14)**, in seconds of video sent for verification:

  V_A = Σ_{i ≤ i*} (b_i − a_i) ≤ |Q| · D_max,   V_B = Σ_{i ∈ C} (b_i − a_i)

where i* is the first candidate that verifies. **V_A does not depend on T.** That is the property hypothesis H7 tests.

### 2.11 Implementation strategy (Section V of the paper)

- Python 3.11, about 10 k lines, organized as ports and adapters. Heavy libraries are imported lazily, so `/health` never loads torch.
- An error boundary turns any exception into structured JSON that names the stage, without forwarding tracebacks.
- Entry points: a CLI (exit codes 0 / 1 / 2), a FastAPI job API (202 + polling) and a React UI.
- The hosted ASD sits behind a provider registry, with a placeholder local provider (Light-ASD).
- More than 1,000 unit and integration tests.

---

## 3. Figure and equation plan

| ID | Content | Source | Purpose |
|---|---|---|---|
| Fig. 1 | Pipeline overview: per-video layer and per-candidate layer, with the early-exit loop | TikZ (exists in `main.tex`; add the onset step) | Architecture at a glance |
| Fig. 2 | Candidate window and frame selection: padding, D_min, clip offset c, onset j₀, a cut, and the selected run | TikZ (exists; **add j₀ and a pre-onset run that is rejected**) | Explains eqs. 2 and 5–7 |
| Fig. 3 | Full control flow with error paths | `figures/architecture.png` | Completeness, reproducibility |
| Fig. 4 | ASD clip-length probe (frames returned vs duration) | pgfplots, measured data | Justifies D_min |
| Fig. 5 (Stage 5) | Ablation results (bar chart) | pgfplots, measured | H2–H6 |
| Fig. 6 (Stage 5) | Onset-error distribution (histogram or CDF) | pgfplots, measured | O1 / RQ2 |
| Alg. 1, Alg. 2 | Above | `algorithmic` | Reproducibility |

Equations 1–14 as numbered above, renumbered in order of appearance in the manuscript.

---

## 4. Points to confirm

1. The notation table and assumptions A1–A7 are what the paper will state. Tell me if your institution or the venue expects different conventions.
2. ε = 0.30 s for the onset tolerance is a design value, chosen from the documented word-timestamp jitter of about ±0.2–0.4 s. Stage 5 can add a small sensitivity check (ε ∈ {0, 0.15, 0.30, 0.45}) if you want. Each extra value costs one more discovery sweep per video.
