import { useState } from "react";

import type { WireDiagnostics } from "../api/types";
import { VERDICT_BLURB, VERDICT_LABEL, speakerLabel, reasonLabel } from "../model/labels";
import type { ResultCard } from "../model/view";
import { ConfidenceBars } from "./ConfidenceBars";
import { FrameStage, Lightbox, MetaRow, VerdictChip } from "./Primitives";

/** "1920x1080" -> "1920 / 1080", so a non-16:9 source letterboxes correctly. */
function aspectFrom(diagnostics: WireDiagnostics | null): string | undefined {
  const resolution = diagnostics?.resolution;
  if (!resolution) return undefined;
  const [w, h] = resolution.split("x").map(Number);
  if (!w || !h) return undefined;
  return `${w} / ${h}`;
}

export function TargetResult({
  card,
  diagnostics,
  onReset,
}: {
  card: ResultCard;
  diagnostics: WireDiagnostics | null;
  onReset: () => void;
}) {
  const [open, setOpen] = useState(false);
  const aspect = aspectFrom(diagnostics);

  // `not_found` is a different answer from `not_on_screen`: one means the line
  // was never said, the other that it was said off camera. Giving it its own
  // panel keeps them from being read as the same outcome.
  if (card.verdict === "not_found") {
    return (
      <section className="err" data-verdict="not_found" style={{ borderColor: "var(--line)" }}>
        <p className="eyebrow">✕ No match in transcript</p>
        <h2 className="err__title">
          {card.text ? <>“{card.text}”</> : "That phrase was not found"}
        </h2>
        <p className="err__detail">{card.detail}</p>
        <div className="actions">
          <button type="button" className="btn btn--primary" onClick={onReset}>
            Try another line
          </button>
        </div>
      </section>
    );
  }

  return (
    <section className="result" data-verdict={card.verdict}>
      <div>
        <FrameStage
          src={card.evidence?.imageUrl ?? null}
          alt={card.text ?? "extracted frame"}
          {...(card.evidence
            ? {
                timecode: card.evidence.timecode,
                frameNumber: card.evidence.frameNumber,
                onOpen: () => setOpen(true),
              }
            : {})}
          chip={<VerdictChip verdict={card.verdict} />}
          eager
          {...(aspect ? { aspect } : {})}
        />

        {card.text ? <p className="result__quote">“{card.text}”</p> : null}

        {card.detail ? (
          <div className="result__callout">
            <strong>{VERDICT_LABEL[card.verdict]}.</strong>{" "}
            {card.reason ? <>{reasonLabel(card.reason)} — </> : null}
            {card.detail}
          </div>
        ) : (
          <div className="result__callout">{VERDICT_BLURB[card.verdict]}</div>
        )}
      </div>

      <aside className="panel">
        <div className="meta">
          {card.evidence ? (
            <>
              <p className="eyebrow">Timestamp</p>
              <p className="meta__hero mono" style={{ margin: 0 }}>
                {card.evidence.timecode}
              </p>
              <MetaRow label="Frame" value={String(card.evidence.frameNumber)} />
              <MetaRow label="Image" value={card.evidence.imageName} />
            </>
          ) : (
            <p className="field__hint">
              No frame was selected, so there is no timestamp to report.
            </p>
          )}

          {card.speaker ? (
            <MetaRow label="Speaker" value={speakerLabel(card.speaker)} />
          ) : null}
          {card.faceConfidence !== null ? (
            <MetaRow label="Face" value={card.faceConfidence.toFixed(4)} />
          ) : null}
          {diagnostics?.fps ? (
            <MetaRow label="FPS" value={diagnostics.fps.toFixed(3)} />
          ) : null}
          {diagnostics?.resolution ? (
            <MetaRow label="Source" value={diagnostics.resolution} />
          ) : null}
        </div>

        {card.breakdown ? (
          <>
            <div className="divider" />
            <ConfidenceBars breakdown={card.breakdown} />
          </>
        ) : card.confidence !== null ? (
          <>
            <div className="divider" />
            <MetaRow label="Confidence" value={card.confidence.toFixed(4)} />
            <p className="conf__note">No per-term breakdown was reported.</p>
          </>
        ) : null}

        <div className="divider" />
        <div className="actions">
          <button type="button" className="btn btn--ghost" onClick={onReset}>
            New run
          </button>
        </div>
      </aside>

      {open && card.evidence ? (
        <Lightbox
          src={card.evidence.imageUrl}
          caption={`${card.evidence.timecode} · frame ${card.evidence.frameNumber}`}
          onClose={() => setOpen(false)}
        />
      ) : null}
    </section>
  );
}
