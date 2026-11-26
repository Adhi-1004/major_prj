# Stage 3: Research Gap, Objectives and Contributions

**Builds on:** Stage 1 (problem analysis) and Stage 2 (verified literature, IDs F1–F8, R1–R20).
**Status:** draft for author approval. No manuscript text yet.

---

## 1. Final gap check (additional search)

Stage 2 left one area open: broadcast speaker naming, and "who said what" retrieval in news and TV archives. A targeted search found an established line of work on **naming speakers in TV broadcast**. That work links transcripts, overlaid on-screen names, faces and voices in order to answer *who* is speaking. It includes the REPERE challenge and several talk-show and news systems. Two records were verified and added:

| ID | Title | Authors | Year | Venue | DOI | Status |
|---|---|---|---|---|---|---|
| F9 | Taking the Bite out of Automated Naming of Characters in TV Video | M. Everingham, J. Sivic, A. Zisserman | 2009 | Image and Vision Computing, vol. 27, pp. 545–559 | 10.1016/j.imavis.2008.04.018 | ✅ |
| F10 | Look Who's Talking: Active Speaker Detection in the Wild (ASW dataset) | Y. J. Kim, H.-S. Heo, S. Choe, S.-W. Chung, Y. Kwon, B.-J. Lee, Y. Kwon, J. S. Chung | 2021 | Interspeech, pp. 3675–3679 | 10.21437/Interspeech.2021-2041 | ✅ |

**What this line of work changes.** It answers "who is this speaker". It does not answer "is the speaker of this line visible at the moment it is said, and from which frame". The gap therefore stands. The related-work section will describe speaker naming as a separate family, citing F9 as its representative, so that reviewers do not read the omission as a lack of awareness.

F10 is also useful for definitions. It treats a speaker as *active* when the face is visible **and** the voice is audible at the same time. That is the same predicate this paper formalizes (Stage 1, eq. for S_q).

---

## 2. Research problem statement (final)

> Searching video for a line of dialogue usually returns the moments where the line is **heard**. Captions, transcripts and speaker-attributed transcripts cannot say whether the person speaking the line is **visible** when they say it. Edited media separates the two routinely: voice-over, narration, internal monologue, reaction shots and reused audio. Given a video and a verbatim dialogue line, the problem is to return the **earliest source-video frame** at which that line is spoken by a **visible, actively speaking person who is also the diarized speaker of the line**. When no such frame exists, the system must state that the line is spoken off camera, or not spoken at all. Without a query, every line in the video must receive an on-screen, ambiguous or off-camera verdict. Each answer must carry an interpretable confidence and must be computed without manual inspection, for videos of arbitrary length, resolution and frame rate.

---

## 3. Research gap (final)

The gap has four parts. Each part is supported by specific reviewed works, and the wording is limited to the reviewed literature.

| # | Gap | Evidence |
|---|---|---|
| G1 | **No reviewed method takes a verbatim, multi-word dialogue line as its query and returns a frame-exact on-camera onset.** Query-driven temporal localization exists for visual-semantic moments (F6, R9) and for isolated keywords on a talking-face video (F5). | F5, F6, R9 |
| G2 | **Off-screen speech is treated as something to correct or reassign, never as a result in its own right.** Audio-visual diarization and subtitling systems assign off-screen lines to an identity by voice (R14, R17). They correct for them in ASD scoring (R3), or recover registered speakers who are not visible (R20). None reports "this line is delivered off camera" as an output. | R3, R14, R17, R20 |
| G3 | **The answer depends on per-frame ASD output that is reliable on benchmarks but degrades out of domain.** AVA mAP reaches 95–97% (R10, R11, R16, R19, R20), yet the best reported UniTalk mAP is about 88–89% (R18–R20). Systems that turn per-frame scores into a single decision need explicit rules: speaker consistency, temporal persistence and shot boundaries. These rules are not specified or evaluated in the subtitling and diarization literature. | R18, R19, R20, R2 |
| G4 | **There is no task-appropriate evaluation.** Existing metrics are per-face mAP, per-speaker DER/JER/CDER and moment IoU. None measures (i) the onset error, in frames, of a queried line, or (ii) the line-level precision of an on-screen vs off-camera verdict. Cost, meaning how much footage is sent to visual verification and the cost of repeated queries, is not reported either. | R1–R20 (metrics used); R14, R17 (whole-episode processing) |

**Gap statement for the manuscript (draft wording):**
> Query-driven localization in video has been studied for visual-semantic moments [F6, R9] and for isolated keywords on talking faces [F5]. Audio-visual speaker attribution for edited media has reached practical accuracy [R12, R14, R17]. However, the reviewed methods do not take a verbatim dialogue line as a query and return the frame at which a *visible* speaker begins to say it. In these systems, off-camera delivery is treated as an attribution problem rather than reported as an outcome. This work addresses that combination and proposes metrics suited to it.

---

## 4. Research questions

- **RQ1.** Can pretrained ASR, diarization and ASD, joined by explicit decision rules, separate on-camera delivery of a line from off-camera delivery with useful precision across different kinds of video?
- **RQ2.** How accurately, in frames and in milliseconds, can the on-camera onset of a queried line be localized?
- **RQ3.** Which decision rules carry the result? Candidates: speaker consistency, the duration-based persistence threshold, shot-cut segmentation, minimum-clip padding, and constraining selection to the match onset.
- **RQ4.** How does visual-verification cost scale under localize-then-verify, compared with sweeping every line?

---

## 5. Research objectives

| # | Objective | Addresses | Measured by (Stage 5) |
|---|---|---|---|
| O1 | Localize the earliest on-camera frame f* of a queried line | G1 | Onset error \|f* − f_gt\| in frames and ms; hit rate within ±k frames |
| O2 | Give every line a verdict (on screen / ambiguous / off camera) with a stated reason | G2 | Line-level precision, recall and F1 for the on-screen class and the off-camera class; ambiguous rate |
| O3 | Make decisions robust to detector flicker, shot cuts and frame-rate variation | G3 | Ablations: each rule removed in turn |
| O4 | Bound the visual cost by the number of matches, not by video length | G4 | ASD requests and % of footage sent, per query and per sweep |
| O5 | Produce deterministic, explainable outputs | G4 | Run-to-run agreement; the confidence breakdown in each output |

---

## 6. Contributions (final, with honest scope)

### 6.1 What is **not** claimed
- No new ASR, diarization or ASD model. Whisper (R4), pyannote (R6, R7) and the hosted NVIDIA ASD service (S1) are used **as published**.
- Using pretrained components together is **not** claimed as new. R14 and R17 already combine ASR, diarization and visual speaker evidence.
- No claim that "no existing system" addresses the problem. The claim is limited to the reviewed literature.

### 6.2 What **is** claimed

| # | Contribution | Type | Novelty (honest) | Evidence required |
|---|---|---|---|---|
| C1 | **Task formulation and metrics.** On-screen dialogue localization defined as the earliest frame satisfying a speaker-consistent, persistent speaking predicate, plus a three-way per-line verdict. Two task metrics: onset-frame error and line-level on/off-screen precision, recall and F1. | Problem definition and evaluation | **Moderate.** It sets the task apart from KWS (F5), moment retrieval (F6, R9) and subtitling (R14, R17). | Formal definitions (Stage 4) plus the annotated evaluation (Stage 5) |
| C2 | **Speaker-consistent temporal verification rule.** A frame qualifies only if the speaking face is associated with the diarized speaker of the queried words. The run must also last τ ms, converted per video using its own frame rate, must not cross a detected shot cut, must be contiguous, and must start at or after the match onset.¹ Each failed condition reports its own reason. | Method | **Moderate.** The individual checks are standard ideas. The contribution is the explicit rule set, how it is specified, and its ablation. | Ablation of each condition (Stage 5) |
| C3 | **Localize-then-verify with cost analysis.** Candidate windows come from the transcript, padded to a measured minimum clip length, and are verified in time order with early exit. The visual cost depends on the number of matches, not on video length. | System design and efficiency | **Incremental to moderate.** Early exit is simple, but it has not been analysed for this task. | Cost measurements across all benchmark videos (Stage 5) |
| C4 | **Mode-consistent, interpretable confidence.** A weighted evidence score that *removes and renormalizes* the text-match term when no query exists, so that target and discovery results share one scale. Explicit overrides force "ambiguous": no diarized speaker, the 5-face cap reached, or simultaneous speakers. | Method | **Incremental.** Needs a calibration check to count as more than a heuristic. | Score distribution against correctness; a reliability check if data allows (Stage 5) |
| C5 | **Empirical integration findings for hosted ASD.** Clips under ~3 s return zero frames (measured); diarization must be sent before any video; the unassociated sentinel is −1 while 0 is a valid speaker; forced CFR is needed for frame/timestamp invertibility. | Reproducibility and engineering knowledge | **Practical, not scientific novelty.** Useful to practitioners. Presented as findings, not as a main contribution. | Existing measurements (clip probe) plus code references |
| C6 | **Open implementation and annotated evaluation set.** A CLI, REST API and web UI, plus labelled clips and an evaluation script. | Artifact | Depends on licensing of the clips (see §9) | Release plan (Stage 5) |

¹ **Onset constraint: a proposed change, not yet implemented.** In the current code (`vision/frame_selector.py`), the selected frame can come before the queried words when the same speaker is already talking inside the ±1.5 s padding. Adding the constraint "run starts at or after the match onset, minus a small tolerance" is a small, testable change. Its effect would be reported as an ablation (with vs without). **This needs your approval because it changes project code.**

### 6.3 Contribution statement for the manuscript (draft)
> The main contributions of this work are: (i) a formulation of on-screen dialogue localization as a query-driven search for the earliest speaker-consistent on-camera onset, together with task-specific evaluation metrics; (ii) a verification rule that converts per-frame active-speaker output into a single frame decision through speaker consistency, duration-defined persistence and shot-boundary constraints, each with an explicit failure reason; (iii) a localize-then-verify pipeline whose visual-verification cost depends on the number of transcript matches rather than video length; and (iv) an evaluation on [N] annotated videos of varied genre, including ablations of each decision rule and comparison with transcript-only and speaker-agnostic baselines.

[N] stays a placeholder until the Stage 5 data exists.

---

## 7. Positioning against the closest works

| Work | Input | Output | Off-camera handling | Exact onset frame for a query | Needs per-video priors |
|---|---|---|---|---|---|
| KWS-Net (F5) | Talking-face video + keyword | Whether/when the keyword is spoken | Not addressed | Keyword timing on the face track | No |
| Spoken video grounding (R9) | Video + spoken query | Moment span (visual event) | Not applicable | No | No |
| Character-aware subtitling (R12, **R14**) | TV episode + cast list + gallery | Speaker name per segment | Assigned to a character by voice | No | **Yes** (cast, gallery) |
| CineSRD (R17) | Film/TV video + subtitles | Diarization (who spoke when) | Assigned by timbre / registered as new speaker | No | No |
| ROAM-ASD-CR (R20) | Face tracks + audio + enrolment | Per-face speaking scores incl. registered off-screen speakers | Recovers registered speakers | No | **Yes** (enrolment) |
| **This work** | Video URL + optional dialogue line | Onset frame, timestamp, text, image, confidence; or a per-line verdict | **Reported as an outcome with a reason** | **Yes** | No |

---

## 8. Hypotheses for the Stage 5 experiments

| # | Hypothesis | Comparison |
|---|---|---|
| H1 | Transcript-only search (every text match counted as on screen) has on-screen precision equal to the base rate of on-screen lines. This is substantially below the proposed system. | Baseline B1 vs full system |
| H2 | Accepting *any* speaking face lowers off-camera precision compared with requiring speaker consistency, especially in multi-person shots. | Ablation A1 (no speaker check) |
| H3 | Removing the persistence threshold (τ = 0) raises false on-screen verdicts caused by detector flicker. | Ablation A2 |
| H4 | Removing shot-cut segmentation produces runs credited across cuts. | Ablation A3 |
| H5 | Not padding short windows to D_min lowers coverage, because short windows return no detections. | Ablation A4 (expected from the clip probe) |
| H6 | The onset constraint lowers the mean onset error without lowering the hit rate. | Ablation A5 (needs the code change, footnote 1) |
| H7 | Target-mode visual cost stays at about one request per transcript match, regardless of video duration. | Cost across videos of different lengths |

These are hypotheses. Any that the data does not support will be reported as such.

---

## 9. Risks to the contribution and mitigations

| Risk | Effect on review | Mitigation |
|---|---|---|
| The evaluation set is small (8–15 clips) | Generalization is questioned | Report per-genre results and confidence intervals (bootstrap over lines); state the limitation openly |
| Copyrighted film/TV clips | Clips cannot be redistributed | Release annotations, timestamps and source URLs only, not media. Prefer Creative-Commons or public-domain clips where possible |
| Dependence on a proprietary hosted ASD | Reproducibility is questioned | Report the service version and date. Note the swappable provider interface and the LR-ASD (R15) path as future work |
| No end-to-end trained baseline for the task | Reviewers ask "compared to what?" | Use transcript-only (B1) and speaker-agnostic ASD (A1) as the most relevant baselines that can be reproduced. Explain why R14/R17 are not directly comparable (different output) |
| Contributions are incremental | Possible rejection at top-tier venues | Target an applied IEEE conference. Make C1 (task + metrics + data) the centrepiece |

---

## 10. Title (final candidates)

1. **Query-Driven Localization of On-Camera Spoken Dialogue in Video via Speaker-Consistent Audio-Visual Verification** *(recommended: precise and searchable)*
2. Separating On-Camera Dialogue from Voice-Over: Frame-Accurate Query Localization with Speaker-Consistent Active Speaker Verification
3. Where Is It Said on Camera? Localizing the On-Screen Onset of a Dialogue Line in Edited Video

Option 1 replaces "On-Screen" with "On-Camera" because it is less ambiguous: "on-screen" can also mean on-screen text or captions, which is a different problem.

---

## 11. Decisions needed from the authors

1. **Title:** approve option 1, or choose another.
2. **Onset-constraint code change (footnote 1):** approve implementing it in Stage 4 and evaluating it as ablation A5. It is a small change in `frame_selector.py` and its caller, with tests.
3. **Contribution set C1–C6:** approve, or remove any you do not want to claim (for example C6, if the clips cannot be released).
