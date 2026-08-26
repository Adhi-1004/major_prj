import type { JobSnapshot, Mode } from "../api/types";
import { STAGE_ORDER, formatDuration, stageLabel } from "../model/labels";
import { partialCards } from "../model/view";
import { ResultCardView } from "./ResultCardView";

/**
 * What a run looks like while it is happening.
 *
 * There is deliberately no percentage. Active-speaker detection is one stage
 * that consumes most of a discovery run, so any denominator would be a lie, and
 * a lying progress bar is worse than none. The elapsed clock, the stage rail and
 * — for a sweep — the live candidate counter carry the reassurance instead.
 */
export function RunMonitor({
  snapshot,
  mode,
  elapsed,
  onCancel,
}: {
  snapshot: JobSnapshot | null;
  mode: Mode;
  elapsed: number;
  onCancel: () => void;
}) {
  const stages = STAGE_ORDER[mode];
  const current = snapshot?.stage;
  const index = current ? stages.indexOf(current as (typeof stages)[number]) : -1;
  const progress = snapshot?.progress;
  const partial = snapshot?.partial ?? [];
  const jobId = snapshot?.job_id ?? "";

  return (
    <>
      <section className="panel" aria-live="polite">
        <div className="progress__head">
          <div>
            <p className="eyebrow">
              {mode === "target" ? "Target search" : "Discovery sweep"}
            </p>
            <p className="progress__counter" style={{ marginTop: 4 }}>
              {snapshot?.url ?? ""}
            </p>
          </div>
          <span className="progress__clock">{formatDuration(elapsed)}</span>
        </div>

        <div className="rail">
          {stages.map((stage, i) => (
            <span
              key={stage}
              className="rail__seg"
              data-state={i < index ? "done" : i === index ? "active" : "todo"}
            />
          ))}
        </div>
        <div className="rail__labels">
          {stages.map((stage, i) => (
            <span key={stage} data-state={i === index ? "active" : "todo"}>
              {stageLabel(stage)}
            </span>
          ))}
        </div>

        <p className="progress__status">
          › {snapshot?.message ?? "starting"}
          {progress ? (
            <span className="progress__counter">
              {"  ·  "}
              {progress.current} of {progress.total} candidates
            </span>
          ) : null}
        </p>

        <div className="progress__foot">
          <span className="mono">{jobId ? `job ${jobId}` : "submitting"}</span>
          <button type="button" className="btn btn--ghost" onClick={onCancel}>
            Stop watching
          </button>
        </div>
      </section>

      {partial.length > 0 ? (
        <section className="section">
          <div className="section__head">
            <p className="eyebrow">Found so far</p>
            <span className="section__count">{partial.length}</span>
          </div>
          <div className="grid">
            {partialCards(partial, jobId).map((card) => (
              <ResultCardView key={card.key} card={card} />
            ))}
          </div>
        </section>
      ) : null}
    </>
  );
}
