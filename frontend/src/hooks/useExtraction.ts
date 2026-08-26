import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, fetchJob, submitExtraction } from "../api/client";
import type { AppError } from "../api/errors";
import type { ExtractRequest, JobSnapshot, Mode } from "../api/types";
import { toRunView, type RunView } from "../model/view";

/**
 * One extraction, from submit to result.
 *
 * The phases are deliberately explicit rather than derived from a pile of
 * booleans, because "submitted but not yet accepted" and "accepted but not yet
 * finished" behave differently and conflating them is how a UI ends up showing
 * a spinner forever.
 */
export type Phase = "idle" | "submitting" | "running" | "done" | "failed";

export interface ExtractionState {
  phase: Phase;
  jobId: string | null;
  mode: Mode | null;
  snapshot: JobSnapshot | null;
  view: RunView | null;
  error: AppError | null;
  /** 422 field messages, so the form can annotate the offending inputs. */
  fieldErrors: Record<string, string> | null;
  /** The job vanished from the server's 100-job cache mid-run. */
  evicted: boolean;
  elapsed: number;
}

const INITIAL: ExtractionState = {
  phase: "idle",
  jobId: null,
  mode: null,
  snapshot: null,
  view: null,
  error: null,
  fieldErrors: null,
  evicted: false,
  elapsed: 0,
};

/**
 * How often to ask for progress.
 *
 * Fast at the start, where stages turn over in seconds and an early failure is
 * most likely; slower later, where a single stage can occupy minutes and there
 * is nothing new to learn. A discovery sweep also reports per-candidate, so
 * there is real movement to catch throughout.
 */
function pollInterval(elapsedMs: number): number {
  if (elapsedMs < 15_000) return 1_000;
  if (elapsedMs < 60_000) return 2_000;
  return 3_000;
}

const MAX_TRANSPORT_FAILURES = 3;

export function useExtraction() {
  const [state, setState] = useState<ExtractionState>(INITIAL);
  const abortRef = useRef<AbortController | null>(null);
  const timerRef = useRef<number | undefined>(undefined);
  const startedRef = useRef<number>(0);

  const stop = useCallback(() => {
    window.clearTimeout(timerRef.current);
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  useEffect(() => stop, [stop]);

  const reset = useCallback(() => {
    stop();
    setState(INITIAL);
  }, [stop]);

  const submit = useCallback(
    async (request: ExtractRequest) => {
      stop();
      startedRef.current = Date.now();
      setState({ ...INITIAL, phase: "submitting" });

      let jobId: string;
      let mode: Mode;
      try {
        const accepted = await submitExtraction(request);
        jobId = accepted.job_id;
        mode = accepted.mode;
      } catch (cause) {
        const error =
          cause instanceof ApiError
            ? cause.error
            : ({
                kind: "malformed",
                httpStatus: null,
                stage: "api",
                reason: "unknown",
                detail: String(cause),
              } as AppError);

        // A validation failure belongs on the form, not on a full-page error:
        // the user's cursor is already in the field that needs fixing.
        setState({
          ...INITIAL,
          phase: error.kind === "validation" ? "idle" : "failed",
          error: error.kind === "validation" ? null : error,
          fieldErrors: error.fields ?? null,
        });
        return;
      }

      setState((prev) => ({ ...prev, phase: "running", jobId, mode }));

      // --- the polling loop ---------------------------------------------------
      let transportFailures = 0;
      let sawSnapshot = false;

      const tick = async (): Promise<void> => {
        const controller = new AbortController();
        abortRef.current = controller;

        try {
          const snapshot = await fetchJob(jobId, controller.signal);
          sawSnapshot = true;
          transportFailures = 0;

          const elapsed = (Date.now() - startedRef.current) / 1000;

          if (snapshot.state === "done" || snapshot.state === "failed") {
            const result = snapshot.result;
            setState((prev) => ({
              ...prev,
              phase: snapshot.state === "failed" ? "failed" : "done",
              snapshot,
              elapsed: snapshot.elapsed_seconds ?? elapsed,
              view: result
                ? toRunView(result, jobId, snapshot.diagnostics ?? null)
                : null,
              error:
                snapshot.state === "failed" && result
                  ? (toRunView(result, jobId, null) as { error?: AppError }).error ??
                    null
                  : null,
            }));
            return; // terminal: stop polling
          }

          setState((prev) => ({ ...prev, snapshot, elapsed }));
        } catch (cause) {
          if (cause instanceof DOMException && cause.name === "AbortError") return;

          const error = cause instanceof ApiError ? cause.error : null;

          // A 404 after a successful poll means the job was evicted from the
          // server's fixed-size cache, which is a different story from a job id
          // that was never valid.
          if (error?.kind === "job_missing") {
            setState((prev) => ({
              ...prev,
              phase: "failed",
              error,
              evicted: sawSnapshot,
            }));
            return;
          }

          // A blip during a four-minute run must not discard the run.
          if (error?.kind === "transport") {
            transportFailures += 1;
            if (transportFailures < MAX_TRANSPORT_FAILURES) {
              timerRef.current = window.setTimeout(
                () => void tick(),
                1_000 * 2 ** transportFailures,
              );
              return;
            }
          }

          setState((prev) => ({
            ...prev,
            phase: "failed",
            error:
              error ??
              ({
                kind: "malformed",
                httpStatus: null,
                stage: "api",
                reason: "unknown",
                detail: String(cause),
              } as AppError),
          }));
          return;
        }

        timerRef.current = window.setTimeout(
          () => void tick(),
          pollInterval(Date.now() - startedRef.current),
        );
      };

      void tick();
    },
    [stop],
  );

  return { state, submit, reset };
}
