import { useState } from "react";

import type { ExtractRequest } from "../api/types";

/** The backend's own defaults and ranges, mirrored so a 422 is unreachable. */
const DEFAULTS = {
  max_candidates: 0,
  time_budget_seconds: 0,
  match_score_floor: 70,
  window_pad_seconds: 1.5,
  min_speaking_ms: 120,
  detect_scene_cuts: true,
};

interface Props {
  busy: boolean;
  fieldErrors: Record<string, string> | null;
  onSubmit: (request: ExtractRequest) => void;
}

export function ExtractForm({ busy, fieldErrors, onSubmit }: Props) {
  const [url, setUrl] = useState("");
  const [dialogue, setDialogue] = useState("");
  const [advanced, setAdvanced] = useState(DEFAULTS);

  // Mirrors the backend's own rule exactly: whitespace-only is not a target,
  // so the badge can never disagree with the mode the server picks.
  const isTarget = dialogue.trim().length > 0;

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    onSubmit({
      url: url.trim(),
      dialogue: dialogue.trim() || null,
      ...advanced,
    });
  };

  const num =
    (key: keyof typeof DEFAULTS) => (event: React.ChangeEvent<HTMLInputElement>) => {
      const value = Number(event.target.value);
      setAdvanced((prev) => ({ ...prev, [key]: Number.isFinite(value) ? value : 0 }));
    };

  return (
    <form className="panel form" onSubmit={submit}>
      <div className="field">
        <label className="eyebrow" htmlFor="url">
          Source
        </label>
        <input
          id="url"
          className="input"
          value={url}
          onChange={(event) => setUrl(event.target.value)}
          placeholder="https://www.youtube.com/watch?v=…  or a local file path"
          aria-invalid={Boolean(fieldErrors?.["url"])}
          aria-describedby={fieldErrors?.["url"] ? "url-error" : undefined}
          autoComplete="off"
          spellCheck={false}
          required
        />
        {fieldErrors?.["url"] ? (
          <p className="field__error" id="url-error">
            {fieldErrors["url"]}
          </p>
        ) : null}
      </div>

      <div className="field">
        <label className="eyebrow" htmlFor="dialogue">
          Dialogue
        </label>
        <input
          id="dialogue"
          className="input"
          value={dialogue}
          onChange={(event) => setDialogue(event.target.value)}
          placeholder="my mind rebels at stagnation"
          autoComplete="off"
        />
        <p className="field__hint">
          Leave empty to sweep the whole video for every on-screen line.
        </p>
      </div>

      <div className="mode-note">
        <span className="mode-note__name">
          {isTarget ? "◉ Target" : "◎ Discovery"}
        </span>
        <span className="mode-note__body">
          {isTarget
            ? "Finds the first frame where this line is spoken on camera, then stops. Usually one detection request."
            : "Checks every spoken line and sorts them into on screen, uncertain, and off camera. One request per candidate, so this takes minutes."}
        </span>
      </div>

      <details className="advanced">
        <summary>Advanced</summary>
        <div className="advanced__grid">
          {!isTarget ? (
            <>
              <NumberField
                id="max_candidates"
                label="Max candidates"
                hint="0 = no limit"
                value={advanced.max_candidates}
                min={0}
                step={1}
                onChange={num("max_candidates")}
              />
              <NumberField
                id="time_budget_seconds"
                label="Time budget (s)"
                hint="0 = no limit"
                value={advanced.time_budget_seconds}
                min={0}
                step={10}
                onChange={num("time_budget_seconds")}
              />
            </>
          ) : (
            <NumberField
              id="match_score_floor"
              label="Match floor"
              hint="0–100, default 70"
              value={advanced.match_score_floor}
              min={0}
              max={100}
              step={1}
              onChange={num("match_score_floor")}
            />
          )}
          <NumberField
            id="window_pad_seconds"
            label="Window pad (s)"
            hint="0–30, default 1.5"
            value={advanced.window_pad_seconds}
            min={0}
            max={30}
            step={0.5}
            onChange={num("window_pad_seconds")}
          />
          <NumberField
            id="min_speaking_ms"
            label="Min speaking (ms)"
            hint="≈ one syllable at 120"
            value={advanced.min_speaking_ms}
            min={10}
            max={5000}
            step={10}
            onChange={num("min_speaking_ms")}
          />
          <label className="checkline">
            <input
              type="checkbox"
              checked={advanced.detect_scene_cuts}
              onChange={(event) =>
                setAdvanced((prev) => ({
                  ...prev,
                  detect_scene_cuts: event.target.checked,
                }))
              }
            />
            Split at scene cuts
          </label>
        </div>
      </details>

      <div className="actions">
        <button
          type="submit"
          className="btn btn--primary"
          disabled={busy || url.trim().length === 0}
        >
          {busy ? "Running…" : "Run extraction"}
        </button>
      </div>
    </form>
  );
}

/**
 * A number input whose `min` and `step` must be chosen together.
 *
 * The browser derives its valid values from `min` stepping by `step`, so a
 * default that is not on that grid makes the control invalid -- and because
 * these live inside a collapsed <details>, the browser cannot focus it to show
 * why. Submit then does nothing at all, with only a console message. `min={1}`
 * with `step={10}` did exactly that to a default of 120.
 */
function NumberField({
  id,
  label,
  hint,
  value,
  min,
  max,
  step,
  onChange,
}: {
  id: string;
  label: string;
  hint: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <div className="field">
      <label className="eyebrow" htmlFor={id}>
        {label}
      </label>
      <input
        id={id}
        className="input"
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={onChange}
      />
      <p className="field__hint">{hint}</p>
    </div>
  );
}
