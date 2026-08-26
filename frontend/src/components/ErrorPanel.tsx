import type { AppError } from "../api/errors";
import { stageLabel } from "../model/labels";

const TITLE: Record<string, string> = {
  transport: "Cannot reach the API",
  job_missing: "That job is gone",
  frame_missing: "Frame not found",
  malformed: "Unreadable response",
  validation: "The request was rejected",
  stage: "The run failed",
};

export function ErrorPanel({
  error,
  evicted,
  onReset,
}: {
  error: AppError;
  evicted: boolean;
  onReset: () => void;
}) {
  return (
    <section className="err">
      <p className="eyebrow">
        {stageLabel(error.stage)} · <span className="mono">{error.reason}</span>
      </p>
      <h2 className="err__title">
        {evicted ? "This job was dropped" : TITLE[error.kind] ?? "Something went wrong"}
      </h2>

      <p className="err__detail">
        {evicted
          ? "The server keeps only the 100 most recent jobs, and this one was evicted while it was still being watched. The run itself may well have finished."
          : error.detail}
      </p>

      {error.remediation ? (
        <p className="err__fix">
          <strong>Try this:</strong> {error.remediation}
        </p>
      ) : null}

      <div className="divider" />

      <details>
        <summary
          style={{
            cursor: "pointer",
            fontSize: "var(--text-xs)",
            letterSpacing: "0.09em",
            textTransform: "uppercase",
            color: "var(--text-faint)",
          }}
        >
          Raw error
        </summary>
        <pre className="well" style={{ marginTop: "var(--s3)" }}>
          {JSON.stringify(error, null, 2)}
        </pre>
      </details>

      <div className="actions" style={{ marginTop: "var(--s4)" }}>
        <button type="button" className="btn btn--primary" onClick={onReset}>
          Start over
        </button>
      </div>
    </section>
  );
}
