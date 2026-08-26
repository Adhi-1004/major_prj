import type { Stage } from "./types";

/**
 * The API fails in three structurally different ways, and this is the only
 * place that knows it.
 *
 *   422  { detail: [ {loc, msg, ...}, ... ] }   <- detail is an ARRAY
 *   404  { detail: {status, stage, reason, ...} } <- detail is an OBJECT
 *   400  { status, stage, reason, detail, ... }   <- FLAT, top level
 *   500  same flat shape
 *
 * 404 and 422 both use the key `detail`, so the only reliable discriminator is
 * whether that value is an array. Everything downstream consumes `AppError`.
 */

export type ErrorKind =
  | "validation"
  | "stage"
  | "job_missing"
  | "frame_missing"
  | "transport"
  | "malformed";

export interface AppError {
  kind: ErrorKind;
  httpStatus: number | null;
  stage: Stage;
  reason: string;
  detail: string;
  remediation?: string;
  /** 422 only: message per offending field, so the form can annotate inputs. */
  fields?: Record<string, string>;
}

function asString(value: unknown, fallback: string): string {
  return typeof value === "string" && value.length > 0 ? value : fallback;
}

function fromStructured(source: Record<string, unknown>, kind: ErrorKind, httpStatus: number | null): AppError {
  const error: AppError = {
    kind,
    httpStatus,
    stage: asString(source["stage"], "api"),
    reason: asString(source["reason"], "unknown_error"),
    detail: asString(source["detail"], "The server reported an error."),
  };
  const remediation = source["remediation"];
  if (typeof remediation === "string" && remediation.length > 0) {
    error.remediation = remediation;
  }
  return error;
}

/** `unknown_job` and `unknown_frame` get their own kinds so the UI can explain them. */
function kindForReason(reason: string): ErrorKind {
  if (reason === "unknown_job") return "job_missing";
  if (reason === "unknown_frame") return "frame_missing";
  return "stage";
}

export function normalizeError(httpStatus: number | null, body: unknown): AppError {
  const envelope = (body ?? {}) as Record<string, unknown>;
  const detail = envelope["detail"];

  // 422 — pydantic's validation array.
  if (Array.isArray(detail)) {
    const fields: Record<string, string> = {};
    for (const entry of detail) {
      const item = (entry ?? {}) as Record<string, unknown>;
      const loc = Array.isArray(item["loc"]) ? (item["loc"] as unknown[]) : [];
      // loc is ["body", "url"]; it can be just ["body"] when the body itself is bad.
      const key = String(loc[loc.length - 1] ?? "body");
      fields[key] = asString(item["msg"], "Invalid value");
    }
    const messages = Object.values(fields);
    return {
      kind: "validation",
      httpStatus,
      stage: "validation",
      reason: "invalid_request",
      detail: messages.length > 0 ? messages.join(" ") : "The request was rejected.",
      fields,
    };
  }

  // 404 — the documented shape, nested one level under `detail`.
  if (detail !== null && typeof detail === "object") {
    const inner = detail as Record<string, unknown>;
    return fromStructured(inner, kindForReason(asString(inner["reason"], "")), httpStatus);
  }

  // 400 / 500 — the same shape, flat.
  if (typeof envelope["status"] === "string") {
    return fromStructured(envelope, "stage", httpStatus);
  }

  // Anything else: almost always the dev proxy returning Vite's HTML index.
  return {
    kind: "malformed",
    httpStatus,
    stage: "api",
    reason: "unreadable_response",
    detail: "The server returned a response this client could not parse.",
    remediation:
      "Check that the API is running on port 8000 and that the dev proxy points at it.",
  };
}

export function transportError(detail: string): AppError {
  return {
    kind: "transport",
    httpStatus: null,
    stage: "api",
    reason: "unreachable",
    detail,
    remediation: "Start the API with: python serve.py --port 8000",
  };
}
