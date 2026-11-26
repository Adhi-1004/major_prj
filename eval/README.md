# Evaluation

This measures how often the system is right, on 14 public-domain and
Creative-Commons clips across six genres. Ground truth comes **only** from people
watching the clips. The system's answers are never shown to annotators.

## What gets measured

| Metric | Question it answers |
|---|---|
| Precision / recall / F1, on-screen class | When the system says "on screen", is it right, and does it find the on-screen lines? |
| Off-camera precision / recall | When it says "off camera" (voice-over, narration, off-screen speaker), is it right? |
| Onset error (frames, ms; % within ±3 / ±12 frames) | How close is the returned frame to the true first on-camera frame of a queried line? |
| Cost | Detector requests sent, and share of footage analysed, per sweep and per query |
| Latency | Seconds per stage, first run vs repeat run |
| Confidence vs correctness | Does a higher confidence score mean a more trustworthy answer? |
| Cohen's κ | Do two annotators agree? This is the ceiling for any system. |

**Baselines and ablations.** B1 is transcript-only search, which calls every heard
line "on screen". The ablations are: A1 no speaker check, A2 no minimum speaking
duration, A3 runs may cross shot cuts, A4 no padding of short windows, and A5 no
onset constraint. Detector responses are cached, so A1, A2, A3 and A5 reuse
byte-identical detector output: they cost no extra requests, and any difference
is caused by the rule itself.

## Steps

All commands run from the repository root, with the project's Python environment
active and `.env` filled in.

```bash
# 1. Download and trim the clips (only the needed segment is fetched)
python -m dialogue_extractor.eval.prepare

# 2. Sweep every clip under every configuration, and write blind labelling files
python -m dialogue_extractor.eval.runner

# 3. LABEL (people, not code) -- see below

# 4. Target-mode runs on the labelled queries (for onset error and per-query cost)
python -m dialogue_extractor.eval.runner --targets --configs full A5_no_onset

# 5. Score everything; tables and plot data land in eval/report/
python -m dialogue_extractor.eval.report
```

## Labelling (step 3)

1. Open `eval/labeler/index.html` in Chrome or Edge.
2. Type your name, then load `eval/labelling/<clip>.json` and
   `eval/data/label_videos/<clip>.mp4`.
3. For each line, watch it (Space) and press 1–5. The definitions are in the tool.
   For lines marked ⏱, step frame by frame to the first frame where the speaker is
   visibly saying the line and press **O**.
4. Press **Save labels**, then move the downloaded file into `eval/labels/`. Keep
   its name `<clip>__<yourname>.json`.

**Two annotators per clip** are recommended, at least for a shared subset. The
report then computes agreement (κ) and scores only the lines the annotators agree
on. Disagreements are listed so they can be settled in a
`<clip>__adjudicated.json` file.

**Rules that keep the numbers honest:**
- Label before looking at any system output, including `eval/runs/`.
- Do not change a label after seeing the report. If a definition was unclear, fix
  the definition and relabel every clip.
- Do not adjust clip segments after labelling has started.

## Files

| Path | Kept in git | Contents |
|---|---|---|
| `manifest.json` | yes | Clips, sources, licences, segments |
| `labelling/` | yes | Blind files for annotators (lines, no verdicts) |
| `labels/` | yes | Human ground truth |
| `runs/` | yes | System output per configuration, plus `prepared.json` clip fingerprints |
| `report/` | yes | Generated tables (`*.tex`), plot data (`*.dat`), `summary.md` |
| `data/` | **no** | Media: clips, normalized videos, work files |
