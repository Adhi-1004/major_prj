import { useState } from "react";

import { reasonLabel, speakerLabel } from "../model/labels";
import type { ResultCard } from "../model/view";
import { FrameStage, Lightbox, VerdictChip } from "./Primitives";

/**
 * One result in a discovery grid.
 *
 * It reads `card.verdict`, never which array the card came from. The `ambiguous`
 * bucket can hold an item whose own status is `not_on_screen`, so labelling by
 * provenance would mislabel it. Sections say where a result sits; the chip says
 * what it is.
 */
export function ResultCardView({ card }: { card: ResultCard }) {
  const [open, setOpen] = useState(false);

  return (
    <article className="card" data-verdict={card.verdict}>
      <FrameStage
        src={card.evidence?.imageUrl ?? null}
        alt={card.text ?? "extracted frame"}
        {...(card.evidence ? { onOpen: () => setOpen(true) } : {})}
      />

      <div className="card__meta">
        <span>{card.timecode ?? "—"}</span>
        <span>
          {card.confidence !== null ? card.confidence.toFixed(3) : "—"}
        </span>
      </div>

      <div className="card__meta">
        <VerdictChip verdict={card.verdict} />
        {card.speaker ? <span>{speakerLabel(card.speaker)}</span> : null}
      </div>

      {card.text ? <p className="card__text">{card.text}</p> : null}

      {card.reason ? (
        <span className="chip chip--muted">{reasonLabel(card.reason)}</span>
      ) : null}

      {open && card.evidence ? (
        <Lightbox
          src={card.evidence.imageUrl}
          caption={`${card.evidence.timecode} · frame ${card.evidence.frameNumber}`}
          onClose={() => setOpen(false)}
        />
      ) : null}
    </article>
  );
}

/**
 * One off-camera finding.
 *
 * Rendered as a row rather than a card, deliberately. These entries have no
 * image — no frame was ever selected — so the transcript text is the content,
 * and rows suit text where grids suit pictures. The structural difference also
 * means the band is identifiable without relying on colour.
 */
export function OffCameraRow({ card }: { card: ResultCard }) {
  return (
    <div className="offrow" data-verdict="off_camera">
      <span className="offrow__time">{card.timecode ?? "—"}</span>
      <div>
        {card.text ? <p className="offrow__text">“{card.text}”</p> : null}
        <div className="offrow__why">
          {card.reason ? (
            <span className="chip chip--muted">{reasonLabel(card.reason)}</span>
          ) : null}
          {card.speaker ? <span>{speakerLabel(card.speaker)}</span> : null}
          {card.detail ? <span>{card.detail}</span> : null}
        </div>
      </div>
    </div>
  );
}
