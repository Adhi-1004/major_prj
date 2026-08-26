import { useState } from "react";

import type { Verdict } from "../model/view";
import { VERDICT_GLYPH, VERDICT_LABEL } from "../model/labels";

export function VerdictChip({ verdict }: { verdict: Verdict }) {
  return (
    <span className="chip" data-verdict={verdict}>
      <span aria-hidden="true">{VERDICT_GLYPH[verdict]}</span>
      {VERDICT_LABEL[verdict]}
    </span>
  );
}

export function MetaRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="meta__row">
      <span className="meta__key">{label}</span>
      <span className="meta__val">{value}</span>
    </div>
  );
}

/**
 * The frame plate.
 *
 * `evidence === null` renders a slate rather than an empty box or a broken
 * image, because "no frame was selected" is a real answer the system gives and
 * it deserves to look deliberate.
 */
export function FrameStage({
  src,
  alt,
  timecode,
  frameNumber,
  chip,
  onOpen,
  aspect,
  eager = false,
}: {
  src: string | null;
  alt: string;
  timecode?: string;
  frameNumber?: number;
  chip?: React.ReactNode;
  onOpen?: () => void;
  aspect?: string;
  /**
   * Load immediately rather than on scroll.
   *
   * True for the single hero frame, which is the answer the user has been
   * waiting minutes for -- deferring that one until a scroll heuristic notices
   * it would be absurd. False for grid thumbnails, where deferring is the whole
   * point: a sweep can produce a dozen 1920x1080 stills.
   */
  eager?: boolean;
}) {
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState(false);

  const style = aspect ? { aspectRatio: aspect } : undefined;

  if (src === null || failed) {
    return (
      <div className="frame" style={style}>
        <div className="frame__slate">
          <span className="frame__slate-glyph" aria-hidden="true">
            {failed ? "⚠" : "◍"}
          </span>
          <span>{failed ? "Image unavailable" : "No verified frame"}</span>
        </div>
        {chip ? <div className="frame__plate frame__plate--tl">{chip}</div> : null}
      </div>
    );
  }

  const body = (
    <div className="frame" style={style}>
      <img
        className="frame__img"
        src={src}
        alt={alt}
        data-loaded={loaded}
        loading={eager ? "eager" : "lazy"}
        fetchPriority={eager ? "high" : "low"}
        decoding="async"
        onLoad={() => setLoaded(true)}
        onError={() => setFailed(true)}
      />
      {chip ? <div className="frame__plate frame__plate--tl">{chip}</div> : null}
      {timecode ? (
        <div className="frame__plate frame__plate--bl">{timecode}</div>
      ) : null}
      {frameNumber !== undefined ? (
        <div className="frame__plate frame__plate--br">frame {frameNumber}</div>
      ) : null}
    </div>
  );

  if (!onOpen) return body;
  return (
    <button
      type="button"
      className="card__btn"
      onClick={onOpen}
      aria-label={`Enlarge ${alt}`}
    >
      {body}
    </button>
  );
}

export function Lightbox({
  src,
  caption,
  onClose,
}: {
  src: string;
  caption: string;
  onClose: () => void;
}) {
  return (
    <div
      className="lightbox"
      role="dialog"
      aria-modal="true"
      aria-label={caption}
      onClick={onClose}
    >
      <div onClick={(event) => event.stopPropagation()}>
        <img src={src} alt={caption} />
        <div className="lightbox__bar">
          <span>{caption}</span>
          <button type="button" className="btn btn--ghost" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
