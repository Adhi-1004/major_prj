"""Score the runs against human labels and write every table the paper needs.

    python -m dialogue_extractor.eval.report

Reads eval/runs/<config>/*.json and eval/labels/*.json. Writes to eval/report/:

    summary.md            human-readable results, every number traceable
    lines.csv             one row per (config, line): truth, verdict, confidence
    ablation.tex          Table: verdict metrics per configuration (+ baseline)
    genre.tex             Table: full system per genre
    onset.tex, cost.tex   Tables: onset error; cost and latency
    *.dat                 pgfplots data: ablation F1, onset CDF, reliability

Nothing here invents a number: a metric with no labelled data behind it is
written as "--" and the reason is stated in summary.md.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from dialogue_extractor.eval import metrics as M
from dialogue_extractor.eval.dataset import (
    CONFIGS,
    LABELLING_DIR,
    REPORT_DIR,
    RUNS_DIR,
    load_manifest,
)
from dialogue_extractor.eval.labels import (
    annotator_files,
    disagreements,
    load_ground_truth,
)


def _fmt(value: float, digits: int = 3) -> str:
    return (
        "--"
        if value is None or (isinstance(value, float) and math.isnan(value))
        else f"{value:.{digits}f}"
    )


def _tex(text: str) -> str:
    """Escape underscores for LaTeX table cells."""
    return text.replace("_", r"\_")


def _line_index(clip_id: str) -> dict[float, str]:
    """line_start -> line_id, from the blind labelling file."""
    path = LABELLING_DIR / f"{clip_id}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        round(float(ln["line_start"]), 3): ln["line_id"]
        for ln in data["lines"]
        if ln["line_start"] is not None
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def collect_pairs() -> tuple[dict[str, list[M.Pair]], list[dict[str, Any]], dict[str, int]]:
    """Join every configuration's verdicts with the ground truth."""
    pairs: dict[str, list[M.Pair]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)

    for clip in load_manifest():
        truth = load_ground_truth(clip.id)
        index = _line_index(clip.id)
        for config in CONFIGS:
            run = _load_json(RUNS_DIR / config.name / f"{clip.id}.discovery.json")
            if run is None:
                continue
            for line in run["lines"]:
                key = (
                    round(float(line["line_start"]), 3) if line["line_start"] is not None else None
                )
                line_id = index.get(key) if key is not None else None
                gt = truth.get(line_id) if line_id else None
                if config.name == "full":
                    counts["lines"] += 1
                    if gt is None:
                        counts["unlabelled_or_disputed"] += 1
                    elif not gt.scored:
                        counts[f"excluded_{gt.label}"] += 1
                    if line["status"] == "error":
                        counts["system_errors"] += 1
                if gt is None or not gt.scored:
                    continue
                pairs[config.name].append(
                    M.Pair(clip.id, gt.is_on, line["status"], line.get("confidence"))
                )
                rows.append(
                    {
                        "config": config.name,
                        "clip_id": clip.id,
                        "genre": clip.genre,
                        "line_id": line_id,
                        "truth": gt.label,
                        "status": line["status"],
                        "confidence": line.get("confidence"),
                        "reason": line.get("reason"),
                        "frame_number": line.get("frame_number"),
                        "gt_onset_frame": gt.onset_frame,
                    }
                )
    return pairs, rows, counts


def onset_errors(config: str) -> list[tuple[int | None, float]]:
    errors: list[tuple[int | None, float]] = []
    for clip in load_manifest():
        run = _load_json(RUNS_DIR / config / f"{clip.id}.targets.json")
        if run is None:
            continue
        for q in run["queries"]:
            fps = float(q.get("fps") or 0) or 25.0
            produced = q["status"] in ("success", "ambiguous") and q["frame_number"] is not None
            errors.append(((q["frame_number"] - q["gt_onset_frame"]) if produced else None, fps))
    return errors


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pairs, rows, counts = collect_pairs()
    write_csv(REPORT_DIR / "lines.csv", rows)
    md: list[str] = ["# Evaluation report", ""]

    # --- dataset and labels --------------------------------------------------
    clips = load_manifest()
    md += ["## Data", "", f"- Clips in manifest: {len(clips)}"]
    md += [f"- {k.replace('_', ' ')}: {v}" for k, v in sorted(counts.items())]
    kappas, all_a, all_b = [], [], []
    for clip in clips:
        per = annotator_files(clip.id)
        if len(per) >= 2:
            a, b = sorted(per)[:2]
            shared = [i for i in per[a] if i in per[b] and per[a][i].scored and per[b][i].scored]
            va = [per[a][i].is_on for i in shared]
            vb = [per[b][i].is_on for i in shared]
            all_a += va
            all_b += vb
            kappas.append(
                (clip.id, len(shared), M.cohen_kappa(va, vb), len(disagreements(clip.id)))
            )
    if kappas:
        md += ["", "### Inter-annotator agreement (Cohen's kappa, on vs off)", ""]
        md += [f"- {c}: n={n}, kappa={_fmt(k)}, disagreements={d}" for c, n, k, d in kappas]
        md += [f"- **Overall: kappa={_fmt(M.cohen_kappa(all_a, all_b))} over {len(all_a)} lines**"]
    else:
        md += ["", "Inter-annotator agreement: not computed (fewer than two annotators per clip)."]

    # --- ablation table ------------------------------------------------------
    full_pairs = pairs.get("full", [])
    md += ["", "## Line-level verdicts (positive class = on screen)", ""]
    header = (
        "| Configuration | Policy | P | R | F1 | F1 95% CI | Acc | Off-P | Off-R | Coverage | n |"
    )
    md += [header, "|" + "---|" * 11]
    tex_rows, dat_rows = [], []

    def add_row(name: str, policy: str, sc: M.Scores, ci: tuple[float, float]) -> None:
        md.append(
            f"| {name} | {policy} | {_fmt(sc.precision)} | {_fmt(sc.recall)} | {_fmt(sc.f1)} | "
            f"[{_fmt(ci[0])}, {_fmt(ci[1])}] | {_fmt(sc.accuracy)} | {_fmt(sc.off_precision)} | "
            f"{_fmt(sc.off_recall)} | {_fmt(sc.coverage)} | {sc.n} |"
        )

    if full_pairs:
        base = M.transcript_only_baseline(full_pairs)
        add_row("B1 transcript-only", "--", base, (float("nan"), float("nan")))
        tex_rows.append(("B1: transcript only", base))
    for config in CONFIGS:
        cp = pairs.get(config.name, [])
        if not cp:
            continue
        for policy in M.POLICIES:
            sc = M.score(cp, policy)
            ci = M.cluster_bootstrap(cp, lambda s, pol=policy: M.score(s, pol).f1)
            add_row(config.name, policy, sc, ci)
            if policy == "strict":
                tex_rows.append((_tex(config.name), sc))
                dat_rows.append((config.name, sc.f1, ci[0], ci[1]))
    if not full_pairs:
        md += ["", "_No labelled lines yet: verdict metrics cannot be computed._"]

    (REPORT_DIR / "ablation.tex").write_text(
        "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
        "Configuration & P & R & F1 & Acc & Off-P & Off-R \\\\\n\\midrule\n"
        + "".join(
            f"{name} & {_fmt(s.precision)} & {_fmt(s.recall)} & {_fmt(s.f1)} & "
            f"{_fmt(s.accuracy)} & {_fmt(s.off_precision)} & {_fmt(s.off_recall)} \\\\\n"
            for name, s in tex_rows
        )
        + "\\bottomrule\n\\end{tabular}\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "ablation_f1.dat").write_text(
        "config f1 lo hi\n"
        + "".join(f"{n} {_fmt(f)} {_fmt(lo)} {_fmt(hi)}\n" for n, f, lo, hi in dat_rows),
        encoding="utf-8",
    )

    # --- per genre (full system) --------------------------------------------
    genre_of = {c.id: c.genre for c in clips}
    by_genre: dict[str, list[M.Pair]] = defaultdict(list)
    for p in full_pairs:
        by_genre[genre_of[p.clip_id]].append(p)
    if by_genre:
        md += [
            "",
            "## Full system by genre (strict policy)",
            "",
            "| Genre | Clips | Lines | On-rate | P | R | F1 | Off-P |",
            "|" + "---|" * 8,
        ]
        genre_tex = []
        for genre, gp in sorted(by_genre.items()):
            sc = M.score(gp, "strict")
            on_rate = sum(p.truth_on for p in gp) / len(gp)
            n_clips = len({p.clip_id for p in gp})
            md.append(
                f"| {genre} | {n_clips} | {len(gp)} | {_fmt(on_rate)} | {_fmt(sc.precision)} | {_fmt(sc.recall)} | {_fmt(sc.f1)} | {_fmt(sc.off_precision)} |"
            )
            genre_tex.append(
                f"{genre} & {n_clips} & {len(gp)} & {_fmt(on_rate)} & {_fmt(sc.precision)} & {_fmt(sc.recall)} & {_fmt(sc.f1)} \\\\\n"
            )
        (REPORT_DIR / "genre.tex").write_text(
            "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\nGenre & Clips & Lines & On-rate & P & R & F1 \\\\\n\\midrule\n"
            + "".join(genre_tex)
            + "\\bottomrule\n\\end{tabular}\n",
            encoding="utf-8",
        )

    # --- onset ---------------------------------------------------------------
    md += ["", "## Onset error, target mode", ""]
    onset_tex = []
    cdf_lines = ["error_frames config_full config_A5"]
    errs_by_cfg = {}
    for name in ("full", "A5_no_onset"):
        errs = onset_errors(name)
        errs_by_cfg[name] = errs
        if not errs:
            md.append(f"- {name}: no target runs (needs labelled queries; run `runner --targets`).")
            continue
        st = M.onset_stats(errs)
        md.append(
            f"- **{name}**: queries={st.n}, frame produced={st.found}, "
            f"mean |err|={_fmt(st.mean_abs_frames, 2)} frames ({_fmt(st.mean_abs_ms, 1)} ms), "
            f"median |err|={_fmt(st.median_abs_frames, 1)}, mean signed={_fmt(st.mean_signed_frames, 2)}, "
            + ", ".join(f"within ±{t}={_fmt(v)}" for t, v in st.within.items())
        )
        tex_name = _tex(name)
        onset_tex.append(
            f"{tex_name} & {st.n} & {st.found} & {_fmt(st.mean_abs_frames, 2)} & "
            f"{_fmt(st.mean_abs_ms, 1)} & {_fmt(st.median_abs_frames, 1)} & "
            f"{_fmt(st.within.get(3, float('nan')))} & {_fmt(st.within.get(12, float('nan')))} \\\\\n"
        )
    if onset_tex:
        (REPORT_DIR / "onset.tex").write_text(
            "\\begin{tabular}{@{}lccccccc@{}}\n\\toprule\nConfig & Queries & Found & Mean $|e|$ (fr) & Mean $|e|$ (ms) & Median $|e|$ & $\\le$3 fr & $\\le$12 fr \\\\\n\\midrule\n"
            + "".join(onset_tex)
            + "\\bottomrule\n\\end{tabular}\n",
            encoding="utf-8",
        )
        for t in range(0, 61, 3):
            vals = []
            for name in ("full", "A5_no_onset"):
                errs = errs_by_cfg.get(name) or []
                vals.append(
                    _fmt(sum(1 for e, _ in errs if e is not None and abs(e) <= t) / len(errs))
                    if errs
                    else "nan"
                )
            cdf_lines.append(f"{t} {vals[0]} {vals[1]}")
        (REPORT_DIR / "onset_cdf.dat").write_text("\n".join(cdf_lines) + "\n", encoding="utf-8")

    # --- cost and latency ----------------------------------------------------
    md += [
        "",
        "## Cost and latency (full configuration)",
        "",
        "| Clip | Genre | Duration (s) | Lines | ASD sent (sweep) | Footage verified (sweep) | Target queries | Mean windows/query | Mean footage/query | Cold run (s) |",
        "|" + "---|" * 10,
    ]
    cost_tex = []
    for clip in clips:
        disc = _load_json(RUNS_DIR / "full" / f"{clip.id}.discovery.json")
        targ = _load_json(RUNS_DIR / "full" / f"{clip.id}.targets.json")
        if disc is None:
            continue
        dur = float(disc["diagnostics"].get("duration") or 0)
        sweep_frac = (
            (disc["diagnostics"].get("verified_seconds") or 0) / dur if dur else float("nan")
        )
        q = targ["queries"] if targ else []
        mean_windows = (
            sum(x.get("candidates_verified") or 0 for x in q) / len(q) if q else float("nan")
        )
        mean_frac = (
            sum((x.get("verified_seconds") or 0) / dur for x in q) / len(q)
            if q and dur
            else float("nan")
        )
        cold = disc["elapsed_seconds"] if disc.get("cold_run") else float("nan")
        md.append(
            f"| {clip.id} | {clip.genre} | {_fmt(dur, 1)} | {len(disc['lines'])} | {disc['asd_requests_sent']} | {_fmt(sweep_frac)} | {len(q)} | {_fmt(mean_windows, 2)} | {_fmt(mean_frac, 4)} | {_fmt(cold, 1)} |"
        )
        tex_id = _tex(clip.id)
        cost_tex.append(
            f"{tex_id} & {clip.genre} & {_fmt(dur, 0)} & {len(disc['lines'])} & {_fmt(sweep_frac, 2)} & {_fmt(mean_windows, 2)} & {_fmt(mean_frac, 4)} \\\\\n"
        )
    if cost_tex:
        (REPORT_DIR / "cost.tex").write_text(
            "\\begin{tabular}{@{}llcccccc@{}}\n\\toprule\nClip & Genre & Dur. (s) & Lines & Sweep footage & Windows/query & Footage/query \\\\\n\\midrule\n"
            + "".join(cost_tex)
            + "\\bottomrule\n\\end{tabular}\n",
            encoding="utf-8",
        )

    # --- reliability ---------------------------------------------------------
    if full_pairs:
        rel = M.reliability(full_pairs)
        md += [
            "",
            "## Confidence vs correctness (full, discovery)",
            "",
            "| C range | Lines | Precision |",
            "|---|---|---|",
        ]
        md += [f"| [{lo:.2f}, {hi:.2f}) | {n} | {_fmt(p)} |" for lo, hi, n, p in rel]
        (REPORT_DIR / "reliability.dat").write_text(
            "lo hi n precision\n"
            + "".join(f"{lo:.2f} {hi:.2f} {n} {_fmt(p)}\n" for lo, hi, n, p in rel),
            encoding="utf-8",
        )

    md += ["", "## Configurations", ""] + [f"- **{c.name}**: {c.description}" for c in CONFIGS]
    (REPORT_DIR / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
