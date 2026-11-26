import { useState } from "react";

import type { WireCandidateError, WireLine } from "../api/types";
import { formatTimecode, type RunView } from "../model/view";
import { CheckYourself } from "./CheckYourself";
import { OffCameraRow, ResultCardView } from "./ResultCardView";

type Discovery = Extract<RunView, { kind: "discovery" }>;

export function DiscoveryResult({
  view,
  onReset,
  lines,
  videoUrl,
}: {
  view: Discovery;
  onReset: () => void;
  lines?: WireLine[] | undefined;
  videoUrl?: string | undefined;
}) {
  const [checking, setChecking] = useState(false);
  const { onScreen, uncertain, offCamera, processed, total, truncated, errors } = view;
  const canCheck = Boolean(videoUrl && lines && lines.length > 0);

  if (checking && canCheck) {
    return (
      <CheckYourself
        lines={lines as WireLine[]}
        videoUrl={videoUrl as string}
        onClose={() => setChecking(false)}
      />
    );
  }
  const filed = onScreen.length + uncertain.length + offCamera.length;
  const unprocessed = Math.max(0, total - processed);

  const share = (n: number) => (filed + unprocessed > 0 ? (n / (filed + unprocessed)) * 100 : 0);

  return (
    <>
      <section className="panel">
        <div className="summary">
          <div className="counts">
            <Count n={onScreen.length} label="On screen" verdict="verified" />
            <Count n={uncertain.length} label="Uncertain" verdict="uncertain" />
            <Count n={offCamera.length} label="Off camera" verdict="off_camera" />
          </div>
          <div style={{ textAlign: "right" }}>
            <p className="eyebrow">Candidates</p>
            <p className="mono" style={{ margin: 0 }}>
              {processed} of {total}
            </p>
            {canCheck ? (
              <button
                type="button"
                className="btn btn--primary"
                style={{ marginTop: "var(--s3)" }}
                onClick={() => setChecking(true)}
              >
                Check it yourself
              </button>
            ) : null}
          </div>
        </div>

        <div className="stacked" aria-hidden="true">
          <span style={{ width: `${share(onScreen.length)}%`, background: "var(--verified)" }} />
          <span style={{ width: `${share(uncertain.length)}%`, background: "var(--uncertain)" }} />
          <span style={{ width: `${share(offCamera.length)}%`, background: "var(--offcam)" }} />
        </div>
      </section>

      {truncated ? (
        <div className="banner">
          <span aria-hidden="true">⚠</span>
          <span>
            <strong>The sweep stopped early.</strong> {unprocessed} candidate
            {unprocessed === 1 ? " was" : "s were"} not examined, so this is not
            the complete set of on-screen lines. Raise the budget to finish it.
          </span>
        </div>
      ) : null}

      <section className="section">
        <div className="section__head" data-verdict="verified">
          <p className="eyebrow">On screen</p>
          <span className="section__count">{onScreen.length}</span>
        </div>
        {onScreen.length > 0 ? (
          <div className="grid">
            {onScreen.map((card) => (
              <ResultCardView key={card.key} card={card} />
            ))}
          </div>
        ) : (
          <p className="empty">No line was verified as spoken on camera.</p>
        )}
      </section>

      {uncertain.length > 0 ? (
        <section className="section">
          <div className="section__head" data-verdict="uncertain">
            <p className="eyebrow">Uncertain</p>
            <span className="section__count">{uncertain.length}</span>
          </div>
          <p className="field__hint">
            A frame was found, but something about the evidence is unresolved.
            Includes lines that produced a frame yet fell below the on-screen
            threshold — each row states its own verdict.
          </p>
          <div className="grid">
            {uncertain.map((card) => (
              <ResultCardView key={card.key} card={card} />
            ))}
          </div>
        </section>
      ) : null}

      {/*
        Off camera gets a band of its own rather than a tab or a footnote.
        Telling these apart from on-screen dialogue is the thing this system
        does that a caption search cannot, so hiding them would hide the point.
      */}
      <section className="offband" data-verdict="off_camera">
        <div className="section__head" style={{ borderBottom: "none" }}>
          <p className="eyebrow">Off camera</p>
          <span className="section__count">{offCamera.length}</span>
        </div>
        <p className="offband__lede">
          Heard but not seen — voice-over, narration, and lines spoken by someone
          the camera is not on. A transcript search returns these as matches;
          separating them is what this system is for.
        </p>
        {offCamera.length > 0 ? (
          offCamera.map((card) => <OffCameraRow key={card.key} card={card} />)
        ) : (
          <p className="empty">Every line examined was spoken on camera.</p>
        )}
      </section>

      {errors.length > 0 ? <CandidateErrors errors={errors} /> : null}

      <div className="actions">
        <button type="button" className="btn btn--ghost" onClick={onReset}>
          New run
        </button>
      </div>
    </>
  );
}

function Count({
  n,
  label,
  verdict,
}: {
  n: number;
  label: string;
  verdict: string;
}) {
  return (
    <div className="count" data-verdict={verdict}>
      <span className="count__n">{n}</span>
      <span className="count__label">{label}</span>
    </div>
  );
}

function CandidateErrors({ errors }: { errors: WireCandidateError[] }) {
  return (
    <details className="advanced">
      <summary>
        {errors.length} candidate{errors.length === 1 ? "" : "s"} failed
      </summary>
      <div style={{ padding: "0 var(--s4) var(--s4)" }}>
        {errors.map((error) => (
          <div key={error.candidate} className="offrow">
            {/* `window` is raw seconds here, unlike every other time in the API. */}
            <span className="offrow__time">
              {formatTimecode(error.window[0])}
            </span>
            <div>
              <p className="offrow__text">
                Candidate {error.candidate + 1} · {error.reason}
              </p>
              <p className="offrow__why">{error.detail}</p>
            </div>
          </div>
        ))}
      </div>
    </details>
  );
}
