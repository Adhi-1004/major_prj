import { useEffect, useState } from "react";

import { fetchHealth } from "../api/client";
import type { HealthResponse } from "../api/types";

export interface HealthState {
  data: HealthResponse | null;
  reachable: boolean;
  /** Credentials the server does not have, named so the UI can warn precisely. */
  missing: string[];
  loading: boolean;
}

const CREDENTIAL_LABEL: Record<string, string> = {
  nvidia_api_key: "NVIDIA_API_KEY",
  asd_function_id: "ASD_FUNCTION_ID",
  hf_token: "HF_TOKEN",
};

/**
 * One check on mount, no polling.
 *
 * Worth doing because a missing credential is the most common setup mistake and
 * it otherwise surfaces three minutes into a run, after transcription has
 * already completed. Learning it before submitting saves the whole wait.
 */
export function useHealth(): HealthState {
  const [state, setState] = useState<HealthState>({
    data: null,
    reachable: true,
    missing: [],
    loading: true,
  });

  useEffect(() => {
    const controller = new AbortController();
    let live = true;

    fetchHealth(controller.signal)
      .then((data) => {
        if (!live) return;
        const missing = Object.entries(data.credentials)
          .filter(([, present]) => !present)
          .map(([key]) => CREDENTIAL_LABEL[key] ?? key);
        setState({ data, reachable: true, missing, loading: false });
      })
      .catch(() => {
        if (!live) return;
        setState({ data: null, reachable: false, missing: [], loading: false });
      });

    return () => {
      live = false;
      controller.abort();
    };
  }, []);

  return state;
}
