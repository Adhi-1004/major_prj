import { useEffect, useMemo, useRef, useState } from "react";

import type { WireLine } from "../api/types";
import { reasonLabel, speakerLabel } from "../model/labels";
import { formatTimecode } from "../model/view";

/**
 * "Check it yourself": the demo version of human evaluation.
 *
 * The viewer watches each line and says whether the speaker is on camera.
 * Only *after* they answer is the system's verdict revealed, with a running
 * tally of how often the two agree. Asking first is the point -- shown the
 * verdict up front, anyone would tend to agree with it.
 *
 * This is a demonstration, not the paper's evaluation. The paper's ground truth
 * comes from eval/labeler, which never reveals verdicts and is filled in by the
 * authors before any results are looked at.
 */

type Answer = "on" | "off" | "unsure";

const SYSTEM_SAYS: Record<string, "on" | "off" | "unsure" | "failed"> = {
  success: "on",
  not_on_screen: "off",
  ambiguous: "unsure",
  error: "failed",
  not_found: "failed",
};

const SAYS_LABEL = { on: "On camera", off: "Off camera", unsure: "Uncertain", failed: "No verdict" };

export function CheckYourself({
  lines,
  videoUrl,
  onClose,
}: {
  lines: WireLine[];
  videoUrl: string;
  onClose: () => void;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const [index, setIndex] = useState(0);
  const [answers, setAnswers] = useState<Record<number, Answer>>({});
  const line = lines[index];

  // Play the current line, with a little lead-in so the viewer sees who is in shot.
  const play = () => {
    const el = video.current;
    if (!el || !line) return;
    el.currentTime = Math.max(0, line.start - 0.7);
    void el.play();
  };
  useEffect(() => {
    const el = video.current;
    if (!el || !line) return;
    const stop = () => {
      if (el.currentTime >= line.end + 0.5) el.pause();
    };
    el.addEventListener("timeupdate", stop);
    return () => el.removeEventListener("timeupdate", stop);
  }, [line]);
  useEffect(play, [index]); // eslint-disable-line react-hooks/exhaustive-deps

  const score = useMemo(() => {
    let compared = 0, agree = 0, systemOn = 0, systemOnRight = 0, systemOff = 0, systemOffRight = 0;
    for (const [i, answer] of Object.entries(answers)) {
      const says = SYSTEM_SAYS[lines[Number(i)]?.status ?? "error"];
      if (answer === "unsure" || says === "unsure" || says === "failed") continue;
      compared += 1;
      if (answer === says) agree += 1;
      if (says === "on") { systemOn += 1; if (answer === "on") systemOnRight += 1; }
      if (says === "off") { systemOff += 1; if (answer === "off") systemOffRight += 1; }
    }
    return { compared, agree, systemOn, systemOnRight, systemOff, systemOffRight };
  }, [answers, lines]);

  if (!line) return null;
  const mine = answers[index];
  const says = SYSTEM_SAYS[line.status] ?? "failed";
  const pct = (a: number, b: number) => (b > 0 ? `${Math.round((a / b) * 100)}%` : "—");

  const answer = (value: Answer) => setAnswers((prev) => ({ ...prev, [index]: value }));
  const go = (delta: number) => setIndex((i) => Math.min(lines.length - 1, Math.max(0, i + delta)));

  return (
    <section className="panel check">
      <div className="check__head">
        <div>
          <p className="eyebrow">Check it yourself</p>
          <p className="field__hint" style={{ margin: 0 }}>
            Watch each line and decide: is the person saying it visibly speaking on camera?
            The system's answer is revealed only after yours.
          </p>
        </div>
        <button type="button" className="btn btn--ghost" onClick={onClose}>
          Back to results
        </button>
      </div>

      <div className="check__body">
        <div>
          <video ref={video} className="check__video" src={videoUrl} controls preload="auto" />
          <div className="check__line">
            <span className="mono">
              Line {index + 1} of {lines.length} · {formatTimecode(line.start)}
              {line.speaker ? ` · ${speakerLabel(line.speaker)}` : ""}
            </span>
            <p className="result__quote">“{line.text}”</p>
          </div>

          <div className="check__answers">
            <button type="button" className="btn" onClick={play}>▶ Replay line</button>
            <span className="check__sep" />
            {(["on", "off", "unsure"] as Answer[]).map((value) => (
              <button
                key={value}
                type="button"
                className={`btn${mine === value ? " btn--primary" : ""}`}
                onClick={() => answer(value)}
              >
                {value === "on" ? "On camera" : value === "off" ? "Off camera" : "Can't tell"}
              </button>
            ))}
          </div>

          {mine ? (
            <div className="check__reveal" data-verdict={says === "on" ? "verified" : says === "off" ? "off_camera" : "uncertain"}>
              <strong>System said: {SAYS_LABEL[says]}</strong>
              {mine !== "unsure" && (says === "on" || says === "off") ? (
                <span className={mine === says ? "check__agree" : "check__disagree"}>
                  {mine === says ? "✓ agrees with you" : "✗ disagrees with you"}
                </span>
              ) : null}
              {line.reason ? <span className="field__hint">{reasonLabel(line.reason)}</span> : null}
              {line.confidence !== null ? (
                <span className="mono field__hint">confidence {line.confidence.toFixed(3)}</span>
              ) : null}
            </div>
          ) : (
            <p className="field__hint">Answer to see what the system decided.</p>
          )}

          <div className="actions" style={{ justifyContent: "space-between" }}>
            <button type="button" className="btn btn--ghost" onClick={() => go(-1)} disabled={index === 0}>
              ← Previous
            </button>
            <button type="button" className="btn btn--ghost" onClick={() => go(1)} disabled={index === lines.length - 1}>
              Next →
            </button>
          </div>
        </div>

        <aside className="check__score">
          <p className="eyebrow">Your score vs the system</p>
          <div className="counts">
            <div className="count" data-verdict="verified">
              <span className="count__n">{pct(score.agree, score.compared)}</span>
              <span className="count__label">agreement ({score.agree}/{score.compared})</span>
            </div>
          </div>
          <dl className="check__stats">
            <dt>When it said “on camera”</dt>
            <dd className="mono">{pct(score.systemOnRight, score.systemOn)} right ({score.systemOnRight}/{score.systemOn})</dd>
            <dt>When it said “off camera”</dt>
            <dd className="mono">{pct(score.systemOffRight, score.systemOff)} right ({score.systemOffRight}/{score.systemOff})</dd>
            <dt>Lines answered</dt>
            <dd className="mono">{Object.keys(answers).length} / {lines.length}</dd>
          </dl>
          <p className="field__hint">
            Lines you could not judge, and lines the system left uncertain, are not counted.
            A transcript search would call every line “on camera”.
          </p>
          <ol className="check__list">
            {lines.map((l, i) => {
              const a = answers[i];
              const s = SYSTEM_SAYS[l.status] ?? "failed";
              const mark = !a ? "" : a === "unsure" || s === "unsure" || s === "failed" ? "·" : a === s ? "✓" : "✗";
              return (
                <li key={l.index} className={i === index ? "is-current" : ""}>
                  <button type="button" onClick={() => setIndex(i)}>
                    <span className="mono">{formatTimecode(l.start).slice(3, 8)}</span>
                    <span className="check__text">{l.text}</span>
                    <span className={mark === "✓" ? "check__agree" : mark === "✗" ? "check__disagree" : ""}>{mark}</span>
                  </button>
                </li>
              );
            })}
          </ol>
        </aside>
      </div>
    </section>
  );
}
