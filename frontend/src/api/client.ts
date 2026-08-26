import { normalizeError, transportError, type AppError } from "./errors";
import type {
  ExtractRequest,
  HealthResponse,
  JobAccepted,
  JobSnapshot,
} from "./types";

/**
 * Paths are origin-relative. In development Vite proxies them to the Python
 * service; in production that service serves this bundle itself. Either way the
 * browser makes same-origin requests and cross-origin rules never apply.
 */

export class ApiError extends Error {
  constructor(readonly error: AppError) {
    super(error.detail);
    this.name = "ApiError";
  }
}

/** Tolerates a non-JSON body, which is what a misconfigured proxy returns. */
async function readBody(response: Response): Promise<unknown> {
  const text = await response.text();
  if (text.length === 0) return {};
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return { __raw: text.slice(0, 200) };
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === "AbortError") throw cause;
    throw new ApiError(
      transportError("Could not reach the API. It may not be running."),
    );
  }

  const body = await readBody(response);
  if (!response.ok) {
    throw new ApiError(normalizeError(response.status, body));
  }
  return body as T;
}

export function submitExtraction(
  body: ExtractRequest,
  signal?: AbortSignal,
): Promise<JobAccepted> {
  const init: RequestInit = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  if (signal) init.signal = signal;
  return request<JobAccepted>("/extract", init);
}

export function fetchJob(id: string, signal?: AbortSignal): Promise<JobSnapshot> {
  return request<JobSnapshot>(`/jobs/${encodeURIComponent(id)}`, signal ? { signal } : undefined);
}

export function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return request<HealthResponse>("/health", signal ? { signal } : undefined);
}

/**
 * URL for one extracted frame.
 *
 * The job id rides along as a query parameter the server ignores. Frames are
 * named only by frame number, so two runs on different videos can produce the
 * same filename; scoping the browser cache per job stops a stale image from a
 * previous run being shown for this one.
 */
export function frameUrl(filename: string, jobId: string): string {
  return `/frames/${encodeURIComponent(filename)}?j=${encodeURIComponent(jobId)}`;
}
