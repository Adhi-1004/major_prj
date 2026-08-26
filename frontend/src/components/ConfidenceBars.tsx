import type { Breakdown } from "../model/view";
import { termBlurb, termLabel } from "../model/labels";

const SWATCHES = [
  "var(--term-a)",
  "var(--term-b)",
  "var(--term-c)",
  "var(--term-d)",
];

/**
 * A contribution bar, not four independent meters.
 *
 * Independent meters cannot show how the score was reached. Decomposing it means
 * the segments literally sum to the score, and the unclaimed remainder is what
 * makes a weak result *look* weak.
 *
 * The 0.45 and 0.70 marks are the most explanatory thing here: they are why a
 * result can carry a real frame and still be reported as off camera.
 */
export function ConfidenceBars({ breakdown }: { breakdown: Breakdown }) {
  return (
    <div>
      <div className="meta__row">
        <span className="meta__key">Confidence</span>
        <span className="meta__val mono">{breakdown.score.toFixed(4)}</span>
      </div>

      <div className="conf__bar">
        {breakdown.terms.map((term, i) => (
          <span
            key={term.name}
            className="conf__seg"
            style={{
              width: `${term.contribution * 100}%`,
              background: SWATCHES[i % SWATCHES.length],
            }}
            title={`${termLabel(term.name)} contributes ${term.contribution.toFixed(4)}`}
          />
        ))}
        <span className="conf__mark" style={{ left: "45%" }} />
        <span className="conf__mark" style={{ left: "70%" }} />
      </div>

      <div className="conf__marks" aria-hidden="true">
        <span className="conf__mark-label" style={{ left: "45%" }}>
          0.45
        </span>
        <span className="conf__mark-label" style={{ left: "70%" }}>
          0.70
        </span>
      </div>

      <table className="conf__table">
        <thead>
          <tr>
            <th>Term</th>
            <th>Value</th>
            <th>Weight</th>
            <th>Adds</th>
          </tr>
        </thead>
        <tbody>
          {breakdown.terms.map((term, i) => (
            <tr key={term.name} title={termBlurb(term.name)}>
              <td>
                <span
                  className="conf__swatch"
                  style={{ background: SWATCHES[i % SWATCHES.length] }}
                />
                {termLabel(term.name)}
              </td>
              <td>{term.value.toFixed(4)}</td>
              <td>{term.weight.toFixed(2)}</td>
              <td>{term.contribution.toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {breakdown.renormalized ? (
        <p className="conf__note">
          Discovery has no target phrase to match against, so the weights sum to{" "}
          {breakdown.weightSum.toFixed(2)} and contributions are renormalized.
          Both modes still score on the same 0–1 scale.
        </p>
      ) : null}
    </div>
  );
}
