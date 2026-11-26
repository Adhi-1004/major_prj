# Stage 2: Literature Review and Source Verification

**Project:** query-driven localization of on-screen spoken dialogue in video
**Status:** draft for author approval. No manuscript text has been written in this stage.
**Date of verification:** 2026-11-18

---

## 1. Search strategy and verification method

**Databases and records used.**
- **Crossref** (the DOI registration agency) was used to verify title, authors, venue, year, pages and DOI.
- **arXiv** abstract pages were used for preprints.
- **Publisher and proceedings pages** were checked: IEEE (CVPR, ICASSP), Springer (ECCV, ACCV, IJCV), ACM (MM), ISCA (Interspeech), ACL Anthology, BMVA (BMVC), PMLR (ICML) and CVF.
- **Search engines** were used only for discovery. Every bibliographic fact in this file was then checked against one of the records above.
- **OpenAlex** returned HTTP 503 during this session and was not used.

**Inclusion criteria.**
- Work that addresses one of three questions: (a) *what was said and when*, (b) *who said it*, (c) *whether the speaker is visibly speaking*.
- Work that localizes a query in video time.
- Recent work (2022–2026) is prioritized. Foundational work is included where later papers build on it.

**Verification legend.**

| Mark | Meaning |
|---|---|
| ✅ | Peer-reviewed. DOI (or official proceedings URL) verified against Crossref or the publisher record. |
| 🟡 | Preprint or accepted paper without a registered DOI yet. Verified against the arXiv record only. Not peer-reviewed unless stated. |
| (abs) | Numbers taken from the paper's abstract. |
| (full) | Numbers taken from the full text (HTML or PDF). |
| (cit.) | Numbers taken from another paper's comparison table, which is named. |
| n.v. | Not verified in this session. Must not be stated in the manuscript until checked. |

---

## 2. Bibliographic record (verified)

### 2.1 Foundational work (before 2022)

| ID | Title | Authors | Year | Venue | DOI / official URL | Status |
|---|---|---|---|---|---|---|
| F1 | AVA Active Speaker: An Audio-Visual Dataset for Active Speaker Detection | J. Roth, S. Chaudhuri, O. Klejch, R. Marvin, A. Gallagher, L. Kaver, S. Ramaswamy, A. Stopczynski, C. Schmid, Z. Xi, C. Pantofaru | 2020 | IEEE ICASSP | 10.1109/ICASSP40776.2020.9053900 | ✅ |
| F2 | Is Someone Speaking? Exploring Long-term Temporal Features for Audio-visual Active Speaker Detection | R. Tao, Z. Pan, R. K. Das, X. Qian, M. Z. Shou, H. Li | 2021 | ACM MM | 10.1145/3474085.3475587 | ✅ |
| F3 | Out of Time: Automated Lip Sync in the Wild | J. S. Chung, A. Zisserman | 2017 (ACCV 2016 Workshops) | LNCS, ACCV Workshops, pp. 251–263 | 10.1007/978-3-319-54427-4_19 | ✅ |
| F4 | Self-supervised Learning of Audio-Visual Objects from Video | T. Afouras, A. Owens, J. S. Chung, A. Zisserman | 2020 | ECCV (LNCS), pp. 208–224 | 10.1007/978-3-030-58523-5_13 | ✅ |
| F5 | Seeing Wake Words: Audio-visual Keyword Spotting | L. Momeni, T. Afouras, T. Stafylakis, S. Albanie, A. Zisserman | 2020 | BMVC | 10.5244/C.34.10 | ✅ |
| F6 | Localizing Moments in Video with Natural Language | L. A. Hendricks, O. Wang, E. Shechtman, J. Sivic, T. Darrell, B. Russell | 2017 | IEEE ICCV, pp. 5804–5813 | 10.1109/ICCV.2017.618 | ✅ |
| F7 | Putting a Face to the Voice: Fusing Audio and Visual Signals Across a Video to Determine Speakers | K. Hoover, S. Chaudhuri, C. Pantofaru, M. Slaney, I. Sturdy | 2017 | arXiv:1706.00079 | https://arxiv.org/abs/1706.00079 | 🟡 |
| F8 | Binary Codes Capable of Correcting Deletions, Insertions, and Reversals | V. I. Levenshtein | 1966 | Soviet Physics Doklady 10(8), pp. 707–710 | No DOI (historical journal) | Bibliographic record standard; not re-checked online |

### 2.2 Recent work (2022–2026)

| ID | Title | Authors | Year | Venue | DOI / official URL | Status |
|---|---|---|---|---|---|---|
| R1 | Learning Long-Term Spatial-Temporal Graphs for Active Speaker Detection (SPELL) | K. Min, S. Roy, S. Tripathi, T. Guha, S. Majumdar | 2022 | ECCV (LNCS) | 10.1007/978-3-031-19833-5_22 | ✅ |
| R2 | AVA-AVD: Audio-Visual Speaker Diarization in the Wild | E. Z. Xu, Z. Song, S. Tsutsui, C. Feng, M. Ye, M. Z. Shou | 2022 | ACM MM, pp. 3838–3847 | 10.1145/3503161.3548027 | ✅ |
| R3 | Unsupervised Active Speaker Detection in Media Content Using Cross-Modal Information | R. Sharma, S. Narayanan | 2022 | arXiv:2209.11896 (listed as under review at IEEE TIP) | https://arxiv.org/abs/2209.11896 | 🟡 ¹ |
| R4 | Robust Speech Recognition via Large-Scale Weak Supervision (Whisper) | A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, I. Sutskever | 2023 | ICML, PMLR vol. 202, pp. 28492–28518 | https://proceedings.mlr.press/v202/radford23a.html | ✅ (PMLR; no DOI) |
| R5 | WhisperX: Time-Accurate Speech Transcription of Long-Form Audio | M. Bain, J. Huh, T. Han, A. Zisserman | 2023 | Interspeech, pp. 4489–4493 | 10.21437/Interspeech.2023-78 | ✅ |
| R6 | pyannote.audio 2.1 Speaker Diarization Pipeline: Principle, Benchmark, and Recipe | H. Bredin | 2023 | Interspeech, pp. 1983–1987 | 10.21437/Interspeech.2023-105 | ✅ |
| R7 | Powerset Multi-Class Cross Entropy Loss for Neural Speaker Diarization | A. Plaquet, H. Bredin | 2023 | Interspeech, pp. 3222–3226 | 10.21437/Interspeech.2023-205 | ✅ |
| R8 | A Light Weight Model for Active Speaker Detection (Light-ASD) | J. Liao, H. Duan, K. Feng, W. Zhao, Y. Yang, L. Chen | 2023 | IEEE/CVF CVPR, pp. 22932–22941 | 10.1109/CVPR52729.2023.02196 | ✅ |
| R9 | Weakly-Supervised Spoken Video Grounding via Semantic Interaction Learning | Y. Wang, W. Lin, S. Zhang, T. Jin, L. Li, X. Cheng, Z. Zhao | 2023 | ACL (Long Papers), pp. 10914–10932 | 10.18653/v1/2023.acl-long.611 | ✅ |
| R10 | LoCoNet: Long-Short Context Network for Active Speaker Detection | X. Wang, F. Cheng, G. Bertasius | 2024 | IEEE/CVF CVPR, pp. 18462–18472 | 10.1109/CVPR52733.2024.01747 | ✅ |
| R11 | TalkNCE: Improving Active Speaker Detection with Talk-Aware Contrastive Learning | C. Jung, S. Lee, K. Nam, K. Rho, Y. J. Kim, Y. Jang, J. S. Chung | 2024 | IEEE ICASSP, pp. 8391–8395 | 10.1109/ICASSP48485.2024.10448124 | ✅ |
| R12 | Look, Listen and Recognise: Character-Aware Audio-Visual Subtitling | B. Korbar, J. Huh, A. Zisserman | 2024 | IEEE ICASSP, pp. 2975–2979 | 10.1109/ICASSP48485.2024.10446480 | ✅ |
| R13 | Synchformer: Efficient Synchronization from Sparse Cues | V. Iashin, W. Xie, E. Rahtu, A. Zisserman | 2024 | IEEE ICASSP, pp. 5325–5329 | 10.1109/ICASSP48485.2024.10448489 | ✅ |
| **R14** | **Character-Aware Audio-Visual Subtitling in Context** (base paper) | **J. Huh, A. Zisserman** | **2024** | **ACCV (LNCS), pp. 365–383** | **10.1007/978-981-96-0908-6_21** | ✅ |
| R15 | LR-ASD: Lightweight and Robust Network for Active Speaker Detection | J. Liao, H. Duan, K. Feng, W. Zhao, Y. Yang, L. Chen, Y. Chen | 2025 | International Journal of Computer Vision | 10.1007/s11263-025-02399-2 | ✅ |
| R16 | Dual-Stream Decoupled Learning for Temporal Consistency and Speaker Interaction in AVSD (D²Stream) | J. Xiao, S. Feng, Z. Wu, J. Yu, H. Yao, Z. Ma, J. Li, Y. Bao, Y. Chen | 2025 (v2 2026) | arXiv:2512.19130 (submitted to ACM MM 2026) | https://arxiv.org/abs/2512.19130 | 🟡 |
| R17 | CineSRD: Leveraging Visual, Acoustic, and Linguistic Cues for Open-World Visual Media Speaker Diarization | L. Huang, X. Liao, C. Cui, S. Wang, Z. Huang, Y. Du, W. Mao | 2026 | IEEE/CVF CVPR 2026 (accepted; arXiv:2603.16966) | https://arxiv.org/abs/2603.16966 ; https://cvpr.thecvf.com/virtual/2026/poster/38404 | 🟡 ² |
| R18 | Revisiting Active Speaker Detection: An In-the-Wild Benchmark for Generalization and Robustness (UniTalk) | L. T. P. Nguyen, Z. Yu, K. Q. N. Cao, Y. Guo, T. H. M. Pham, T. T. Nguyen, T. N. D. Vo, L. Poon, T. K. Nguyen, S. Lee, Y. J. Lee | 2026 | Interspeech 2026 (accepted; arXiv:2505.21954) | https://arxiv.org/abs/2505.21954 | 🟡 ² |
| R19 | ROAM-ASD: Robust Open-World Active Speaker Detection with Flexible Multimodal Fusion | P. Wang, H. Van hamme | 2026 | arXiv:2609.26648 (submitted to ICASSP 2027) | https://arxiv.org/abs/2609.26648 | 🟡 |
| R20 | Beyond Visible Speakers: Context-Aware Active Speaker Detection with Speaker Registration | P. Wang, H. Van hamme | 2026 | arXiv:2610.33127 (submitted to ICASSP 2027) | https://arxiv.org/abs/2610.33127 | 🟡 |

¹ A related peer-reviewed paper by the same authors exists: R. Sharma and S. Narayanan, "Audio-Visual Activity Guided Cross-Modal Identity Association for Active Speaker Detection," *IEEE Open J. Signal Process.*, vol. 4, pp. 225–232, 2023, DOI 10.1109/OJSP.2023.3267269 ✅. Its reference list cites R3 as a separate work, so it is **not** the published version of R3. Cite R3 as an arXiv preprint.
² Accepted at the named venue according to the arXiv comments and the venue page. A proceedings DOI was not found during verification. Check again before submission.

### 2.3 Software and services the system depends on (cited as resources, not as research)

| ID | Resource | Official URL |
|---|---|---|
| S1 | NVIDIA Active Speaker Detection NIM documentation | https://docs.nvidia.com/nim/maxine/active-speaker-detection/latest/index.html |
| S2 | NVIDIA-Maxine nim-clients (vendored gRPC client and proto definitions) | https://github.com/NVIDIA-Maxine/nim-clients |
| S3 | faster-whisper (CTranslate2 Whisper runtime) | https://github.com/SYSTRAN/faster-whisper |
| S4 | pyannote speaker-diarization-community-1 pipeline | https://huggingface.co/pyannote/speaker-diarization-community-1 |
| S5 | FFmpeg | https://ffmpeg.org |
| S6 | yt-dlp | https://github.com/yt-dlp/yt-dlp |
| S7 | RapidFuzz | https://github.com/rapidfuzz/RapidFuzz |
| S8 | PyAV | https://github.com/PyAV-Org/PyAV |

---

## 3. Technical comparison (per paper)

"Relevance" describes how each work relates to the proposed system: what it supplies, what it competes with, or what gap it leaves.

### 3.1 Theme A: Active speaker detection (is this visible face speaking?)

| ID | Objective | Method / algorithms | Experimental setup | Results | Limitations | Relevance |
|---|---|---|---|---|---|---|
| F1 | Provide a benchmark for audio-visual ASD | Dense per-face, per-frame speaking labels on movie clips | Movie footage | Dataset paper. Counts n.v. | Older-film domain; R18 reports that it overstates real-world accuracy | Standard benchmark for every ASD model below |
| F2 | ASD with long-term temporal context | Audio and visual temporal encoders with cross-attention (TalkNet) | AVA-ActiveSpeaker | 92.0% mAP as reproduced in R11 Table 2 (cit.) | Needs face tracks; per-face output only | ASD model used inside R17 |
| R1 | Long-range spatio-temporal reasoning at low cost | Graph per video: person-frame nodes, temporal and spatial edges; node classification | AVA-ActiveSpeaker val | 94.2% mAP; SPELL+ 94.9% (full). Visual encoder about 0.7 GFLOPs / 11.2 M params vs ASDNet 13.2 GFLOPs / 48.6 M (full) | Over-smoothing when distant frames are linked; visual-only 84.9%, audio-only 55.6% (full) | Typical per-face, per-frame classifier: no query, no transcript |
| R8 | Lightweight ASD | Fewer candidate faces; split 2D/3D convolutions; GRU cross-modal modelling | AVA-ActiveSpeaker; Columbia | 94.1% mAP with 1.0 M params and 0.6 GFLOPs (abs) | About 0.1 mAP below the state of the art at the time; needs face tracks | The system's registered local fallback provider (not implemented) |
| R10 | Joint long-term intra-speaker and short-term inter-speaker context | Interleaved self/cross-attention (LIM) and convolutional inter-speaker blocks (SIM) | AVA, Talkies, Ego4D, Columbia | AVA 95.2%, Talkies 97.2%, Ego4D 68.4% (CVPR abs); +3.0 mAP in multi-speaker scenes | Computationally heavier; the arXiv and CVPR versions report different numbers (cite CVPR) | Upper bound of conventional ASD accuracy in 2024 |
| R11 | Better audio-visual embeddings for ASD | Talk-aware contrastive loss applied before fusion, only on active segments | AVA val; ASW | TalkNet 92.0→92.5; LoCoNet 95.2→95.5 mAP; ASW 98.8 (val) / 99.3 (test) (full) | Training add-on only; sensitive to loss placement and weight | Shows ASD gains on AVA have become marginal |
| R15 | Real-time, robust lightweight ASD | Journal extension of R8 | AVA; cross-dataset Talkies, Columbia, RealVAD | 94.5% mAP, 0.84 M params, 0.51 GFLOPs, 0.1–3.9 ms per frame (abs/full) | About 1 mAP below the heaviest models; needs face tracks | Candidate local replacement for the hosted detector |
| R16 | Resolve the conflict between temporal and social cues | Two decoupled branches (ITC, ISR), motivated by an 86.1° gradient-direction conflict | AVA; Columbia | 95.6% mAP on AVA (abs) | Preprint; two datasets only; params/FLOPs not in the abstract (n.v.) | Evidence of saturation on AVA |
| R18 | Test ASD under realistic conditions | New benchmark (UniTalk), comparable in scale to AVA; under-represented languages, noise, crowded scenes | Cross-dataset evaluation of state-of-the-art models | Models near saturation on AVA do not saturate on UniTalk; training on UniTalk transfers better to Talkies and ASW (abs) | Benchmark only; exact size n.v. | Motivates conservative use of detector output (persistence, ambiguity reporting) |
| R19 | Robust ASD when some inputs are missing | Joint self-attention over audio, face and mouth crop; modality-agnostic query tokens; modality dropout | WASD, UniTalk, AVA, ASW, Talkies | WASD 98.8, UniTalk 87.9, AVA 96.5, ASW 99.3, Talkies 98.2 mAP (abs) | Preprint | State of the art per face; still no query or line-level output |
| R20 | ASD for co-visible and registered speakers, including off-screen | ROAM-ASD-C reasons over co-visible faces; ROAM-ASD-CR adds registered-speaker embeddings from enrolment speech | AVA, WASD, UniTalk, Talkies, ASW | ROAM-ASD-C: AVA 96.8, WASD 98.9, UniTalk 88.8, Talkies 98.7, ASW 99.5 mAP; up to +13.3 mAP with 75% of the target's visuals removed (abs) | Preprint; needs enrolment speech for each speaker | Closest ASD work to the off-screen question, but answers "is a registered speaker talking", not "is this queried line spoken on camera, at which frame" |

### 3.2 Theme B: Audio-visual synchronization and talking-face localization

| ID | Objective | Method | Setup | Results | Limitations | Relevance |
|---|---|---|---|---|---|---|
| F3 | Detect audio-video offset; find the speaking face | Two-stream CNN learning a joint audio-visual sync embedding (SyncNet) | TV broadcast footage | n.v. | Needs a face crop; short temporal window | Background for lip-sync-based on-screen verification |
| F4 | Self-supervised localization of audio-visual objects (talking faces) | Audio-visual attention over video (LWTNet) | Talking-face videos | n.v. | n.v. | Used by R14 to find the talking face |
| R13 | Efficient audio-visual synchronization from sparse cues | Transformer synchronizer (Synchformer) | Open-domain video | n.v. | n.v. | Alternative synchrony backend |

### 3.3 Theme C: Speech recognition and speaker diarization (what, when, who, from audio only)

| ID | Objective | Method | Setup | Results | Limitations | Relevance |
|---|---|---|---|---|---|---|
| R4 | Robust zero-shot ASR | Encoder-decoder Transformer trained on 680 k hours of weakly supervised web audio (abs) | Many ASR benchmarks | Robust zero-shot transcription (abs; figures n.v.) | Coarse native timestamps; hallucination on long audio (reported in R5) | The ASR model used (large-v3, via S3) |
| R5 | Time-accurate long-form transcription | VAD cut-and-merge into about 30 s chunks; batched Whisper; wav2vec2 phoneme forced alignment | AMI, Switchboard, TED-LIUM, Kincaid46 | Word-segmentation P/R: AMI 84.1/60.3 vs Whisper 78.9/52.1; SWB 93.2/65.4 vs 85.4/62.8. WER: TED-LIUM 9.7 vs 10.5; Kincaid46 11.8 vs 12.5. About 11.8× faster (full) | Hallucination and repetition remain; no translation alignment; multilingual not benchmarked | Word-timestamp accuracy bounds candidate-window quality; candidate for an ablation |
| R6 | A reproducible neural diarization pipeline | Segmentation + embedding + clustering (pyannote 2.1) | Multiple diarization benchmarks | n.v. | Audio only; anonymous speaker labels | Basis of the diarizer used (S4) |
| R7 | Handle overlapping speech in diarization | Powerset multi-class formulation instead of multi-label | Diarization benchmarks | n.v. | Audio only | Part of the pyannote model family used |

### 3.4 Theme D: Audio-visual diarization and character-aware subtitling (who speaks, with visual evidence)

| ID | Objective | Method | Setup | Results | Limitations | Relevance |
|---|---|---|---|---|---|---|
| F7 | Link faces and voices without task-specific training | Weak audio speech clusters fused with visual signals across a video | YouTube videos, human-rated | About 71% accuracy vs human labels (abs). The paper body describes an "off-screen speaker" tag (search excerpt; n.v. in full text) | Preprint; early and weak cues | Early statement of the on-screen/off-screen distinction |
| R2 | Audio-visual diarization in the wild | Dataset (117 AVA films, 351 clips, about 29 h) plus AVR-Net with a face-visibility modality mask | AVA-AVD | DER 20.57 (oracle VAD + ASD labels) vs WST 42.04 (full) | Large drop with predicted VAD; ASD adds about 3.5 DER; at ≥80% missing faces GMU beats it; dubbed video excluded (full) | Shows off-screen speakers are the core difficulty for AV methods |
| R3 | Unsupervised ASD in media content | Speech–face assignment using cross-modal identity structure; simple off-screen speech handling | VPCD, AVA-ActiveSpeaker, Columbia | Competitive with supervised methods (abs; figures n.v.) | Preprint; off-screen handling is a correction step, not an output | Treats off-screen speech as a nuisance to correct |
| R12 | Character-aware subtitling | AV cues select high-precision per-character voice exemplars; segments classified by nearest exemplar | Seinfeld, Frasier, Scrubs | Reproduced by R14: DER 26.4/28.0/26.7%, 6.8% "unknown" (cit. R14) | Short segments unresolved; needs a cast list; voice identity does not show visibility | Precursor of the base paper |
| **R14** | **Character-aware subtitling using dialogue context** | WhisperX + Silero VAD; LWTNet talking-face heatmap; CLIP-based character recogniser; ECAPA-TDNN exemplars; short segments resolved with local embeddings and Llama-3-70B over ±15 lines; pyannote 2.1 overlap detection (full) | LLR-TV (Frasier, Seinfeld, Scrubs) and Bazinga!-gold-TV (9 shows); 12 series in total; about 71% of segments are under 2 s (full) | DER/CDER on LLR-TV: Frasier 20.3/28.8 vs 26.4/39.1; Seinfeld 23.3/33.6 vs 28.0/40.7; Scrubs 25.7/37.0 vs 26.7/40.3. Character accuracy 84–89%. "Unknown" 6.8% → 0.77% (full) | Needs a per-show cast list and gallery; the LLM guesses when unsure (lower precision); overlap handled by heuristic; grid-searched hyperparameters (full) | **Base paper.** Closest system-level work. It outputs a speaker identity per segment, not an on/off-screen verdict, an exact onset frame or a query result |
| R17 | Speaker diarization for films and TV | TalkNet active-speaker anchors clustered into registered speakers; audio LLM for speaker-turn detection; off-screen lines assigned by timbre or registered as new speakers; training-free (full) | SubtitleSD (Chinese, Chinese-Hard, English; Youku film/TV); AVA-AVD | SubtitleSD DER (AVT): 0.076 / 0.095 / 0.089; AVA-AVD DER 0.1898 vs AVR-Net 0.2057 (full) | Multi-stage; weak on animation; small English subset (full) | Recent proof that ASD + audio + text fusion works on film. Off-screen lines are absorbed into identities rather than reported |

### 3.5 Theme E: Query-driven temporal localization in video

| ID | Objective | Method | Setup | Results | Limitations | Relevance |
|---|---|---|---|---|---|---|
| F6 | Localize moments described in natural language | Moment Context Network; joint video-language embedding | DiDeMo | n.v. | Semantic, clip-level moments; no speech or speaker reasoning | Text-to-moment retrieval targets *visual semantics*, not verbatim speech |
| R9 | Localize moments given a *spoken* query | Weakly supervised speech–video semantic interaction | Spoken video grounding benchmarks | n.v. | The query is spoken, but the target is a visual event | Different task; shows speech queries for grounding do exist |
| F5 | Decide *whether and when* a target word is spoken by a talking face | KWS-Net: similarity map between keyword and lip-video features, then CNN pattern detection; works with or without audio (abs) | Lip-reading benchmarks (exact sets n.v.) | Beats previous visual keyword spotting and a lip-reading baseline; audio helps substantially in noise; transfers to French and German (abs) | Visually similar words ("may"/"pay"/"bay") are confusable without audio; single keywords; assumes a talking-face video (not verified as an explicit assumption) | **Closest query-driven prior work.** It localizes a keyword on a talking face. It does not handle multi-word lines in edited multi-speaker footage, does not separate off-screen speech from a visible but silent face, and does not return an onset frame in the source video |

---

## 4. Synthesis

### 4.1 What existing work already achieves
1. **Per-face speaking detection is mature on movie benchmarks.** AVA mAP rose from 94.1–94.2% (R1, R8) to 95.2–95.6% (R10, R11, R16) and 96.5–96.8% (R19, R20).
2. **Word-level transcription with usable timestamps is available off the shelf** (R4, R5). So is neural diarization (R6, R7).
3. **Audio-visual speaker attribution for TV and film works at the system level** (R12, R14, R17). Off-screen speech is handled by assigning it to an identity by voice.
4. **Query-driven localization exists in two neighbouring forms:** visual-semantic moments (F6, R9) and single keywords on talking faces (F5).

### 4.2 Problems that remain open
1. **The answer has a different shape.** The systems in Theme D output *who* spoke each segment. None of the reviewed works outputs, for a queried line, the earliest source-video frame at which that line is spoken by a visible speaker.
2. **Off-screen speech is treated as noise to correct** (R3, R17, R14) or as an identity to recover (R20). It is not reported as a result in its own right. Telling on-camera delivery apart from voice-over is the product this project needs.
3. **Accuracy drops out of domain.** ASD scores on AVA approach saturation, but R18 shows a clear drop on modern in-the-wild video (best reported UniTalk mAP about 88–89%, R19/R20). Downstream systems must therefore treat per-frame detector output conservatively.
4. **Cost is not a design variable.** The subtitling and diarization systems process whole episodes. None of the reviewed works evaluates verifying only the transcript-localized windows, or the cost of answering repeated queries.
5. **No metric fits the task.** Evaluations use mAP (per face), DER/JER/CDER (per speaker) or moment IoU. None measures the *onset-frame error of a queried line* or the *precision of an on/off-screen verdict per line*.

### 4.3 Preliminary research gap (to be finalized in Stage 3)
> Among the works reviewed here, query-driven localization exists only for visual-semantic moments (F6, R9) and for isolated keywords on a talking face (F5). Audio-visual speaker attribution for edited media (R12, R14, R17) assigns identities to speech but does not report whether the speaker is visible when a given line is spoken, or at which frame. The gap is a method that takes a verbatim dialogue line as its query and returns the earliest source-video frame at which that line is spoken by a visible, speaker-consistent face. The same method should report off-camera delivery explicitly, with a confidence value, and be evaluated with onset-frame and on/off-screen metrics.

The wording deliberately avoids "no existing system". The claim is limited to the reviewed literature. Stage 3 should search once more for broadcast and media-monitoring work, such as quote verification and speaker-attributed retrieval in news archives.

---

## 5. Information that could not be verified in this session

| Item | Status | Action needed before citing |
|---|---|---|
| Exact numeric results of F2 (own paper), F3, F4, R4, R6, R7, R13 | n.v. | Cite these only for their method, or check the full text first |
| KWS-Net datasets and numbers (F5) | Abstract gives no numbers | Read the BMVC paper (the 10 MB PDF exceeded the fetch limit) |
| F7 "off-screen speaker" tag | From a search excerpt of the paper body | Confirm in the full text, or cite F7 only for face–voice association |
| R3 numbers | Abstract gives no numbers | Read the full text |
| D²Stream parameter count and FLOPs (R16) | Not in the abstract | Read the full text, or omit |
| UniTalk dataset size (R18) | Abstract only says "on par with AVA" | Read the full text, or omit |
| DOIs for R17 (CVPR 2026) and R18 (Interspeech 2026) | Not yet registered or found | Check CVF/ISCA again before submission |
| The claim that NVIDIA's ASD model is trained on AVA-ActiveSpeaker | Appears in `docs/approach.md`; not found in an official NVIDIA record during this session | Do not state it in the manuscript unless the NIM documentation confirms it |
| Transpotter (Prajwal et al., "Visual Keyword Spotting with Attention," BMVC 2021, DOI 10.5244/C.35.212) | Metadata verified (✅); content not read | Optional addition to Theme E after reading |

---

## 6. Proposed use in the manuscript

| Manuscript section | Sources |
|---|---|
| Introduction (problem and motivation) | R4, R5, F1, R18, R14 |
| Related Work A: ASD | F1, F2, R1, R8, R10, R11, R15, R16, R19, R20, R18 |
| Related Work B: sync / talking face | F3, F4, R13 |
| Related Work C: ASR and diarization | R4, R5, R6, R7 |
| Related Work D: AV diarization and subtitling | F7, R2, R3, R12, **R14**, R17 |
| Related Work E: query-driven localization | F6, R9, F5 |
| Method (components used) | R4, S3, R6, S4, S1, S2, S5, S6, S7, S8, F8 |
| Discussion (robustness, local detector) | R15, R18, R19 |
| Comparison table (capabilities) | R1/R8/R10/R15 (grouped), R20, R5, R2, R12, R14, R17, F5 |
