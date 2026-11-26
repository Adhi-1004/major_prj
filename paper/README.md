# IEEE paper: Query-Driven Localization of On-Screen Spoken Dialogue

## Files
| File | What it is |
|---|---|
| `main.tex` | The paper (IEEEtran conference format, two columns) |
| `refs.bib` | Bibliography, 27 entries, IEEE style via `IEEEtran.bst` |
| `figures/architecture.png` | Detailed control-flow diagram (Fig. 3) |
| `figures/system-flow.png`, `figures/user-flow.png` | Extra diagrams from the project, not used in the paper; available if you want them |

Figs. 1, 2 and 4 are drawn in the source with TikZ/pgfplots, so they need no image files.

## Compile on Overleaf
1. Overleaf → **New Project** → **Upload Project** → choose `paper.zip`.
2. Menu → Settings: Compiler **pdfLaTeX**, TeX Live version **2023 or newer**.
3. Click **Recompile**. Overleaf runs BibTeX automatically.

## Before submitting
- Replace the placeholder author names, department, institution and emails in `main.tex`.
- Check the page count. It should land at about 6–7 pages. To shorten it, reduce the
  architecture figure (`height=0.42\textheight`), or remove the Determinism
  subsection (VII-C). To lengthen it, add the `user-flow.png` diagram in Section V.
- `wang2026beyond`, `wang2026roam` and `xiao2025d2stream` are arXiv preprints, and
  `jung2023talknce` is cited from arXiv. Update them if they have been published since.
- Keep the AI-assistance statement in the Acknowledgment, as required by IEEE policy.
