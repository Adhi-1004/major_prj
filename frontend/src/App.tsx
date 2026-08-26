import { useExtraction } from "./hooks/useExtraction";
import { useHealth } from "./hooks/useHealth";
import { DiscoveryResult } from "./components/DiscoveryResult";
import { ErrorPanel } from "./components/ErrorPanel";
import { ExtractForm } from "./components/ExtractForm";
import { RunMonitor } from "./components/RunMonitor";
import { TargetResult } from "./components/TargetResult";

export default function App() {
  const { state, submit, reset } = useExtraction();
  const health = useHealth();

  const busy = state.phase === "submitting" || state.phase === "running";

  return (
    <div className="shell">
      <header className="masthead">
        <h1 className="masthead__title">
          <span className="masthead__mark" aria-hidden="true">
            ◉
          </span>
          Dialogue Frame Extractor
          <span className="masthead__sub">
            the exact frame a line is spoken on screen
          </span>
        </h1>
        <HealthPill />
      </header>

      <main className="main">
        {state.phase === "idle" || state.phase === "submitting" ? (
          <>
            {!health.loading && !health.reachable ? (
              <div className="banner">
                <span aria-hidden="true">⚠</span>
                <span>
                  <strong>The API is not responding.</strong> Start it with{" "}
                  <code className="mono">python serve.py --port 8000</code>.
                </span>
              </div>
            ) : null}

            {health.missing.length > 0 ? (
              <div className="banner">
                <span aria-hidden="true">⚠</span>
                <span>
                  <strong>
                    {health.missing.join(", ")} not configured.
                  </strong>{" "}
                  Runs will fail partway through — after transcription, which
                  takes a few minutes. Set them in <code className="mono">.env</code>{" "}
                  and restart the API.
                </span>
              </div>
            ) : null}

            <ExtractForm
              busy={busy}
              fieldErrors={state.fieldErrors}
              onSubmit={submit}
            />
          </>
        ) : null}

        {state.phase === "running" ? (
          <RunMonitor
            snapshot={state.snapshot}
            mode={state.mode ?? "target"}
            elapsed={state.elapsed}
            onCancel={reset}
          />
        ) : null}

        {state.phase === "done" && state.view?.kind === "target" ? (
          <TargetResult
            card={state.view.card}
            diagnostics={state.view.diagnostics}
            onReset={reset}
          />
        ) : null}

        {state.phase === "done" && state.view?.kind === "discovery" ? (
          <DiscoveryResult view={state.view} onReset={reset} />
        ) : null}

        {state.phase === "failed" ? (
          <ErrorPanel
            error={
              state.error ?? {
                kind: "malformed",
                httpStatus: null,
                stage: "api",
                reason: "unknown",
                detail: "The run failed without reporting a reason.",
              }
            }
            evicted={state.evicted}
            onReset={reset}
          />
        ) : null}
      </main>
    </div>
  );

  function HealthPill() {
    if (health.loading) {
      return (
        <span className="health">
          <span className="health__dot" />
          checking
        </span>
      );
    }
    if (!health.reachable) {
      return (
        <span className="health" style={{ color: "var(--error)" }}>
          <span className="health__dot" style={{ ["--c" as string]: "var(--error)" }} />
          API offline
        </span>
      );
    }
    const ok = health.missing.length === 0;
    return (
      <span className="health">
        <span
          className="health__dot"
          style={{ ["--c" as string]: ok ? "var(--verified)" : "var(--uncertain)" }}
        />
        {ok ? "ready" : `${health.missing.length} credential(s) missing`}
      </span>
    );
  }
}
