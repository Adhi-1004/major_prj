import { frameUrl } from "../api/client";
import type { AppError } from "../api/errors";
import type {
  AmbiguityReason,
  ResultStatus,
  WireBreakdown,
  WireCandidateError,
  WireDiagnostics,
  WireDiscoveryResult,
  WireExtractionResult,
  WireResult,
} from "../api/types";
import { isDiscoveryResult, isErrorResult } from "../api/types";

/**
 * The shapes components actually render.
 *
 * The wire types cannot be a useful discriminated union, because `status` does
 * not determine the shape. `not_on_screen` has two different producers: a
 * verified frame that scored below 0.45 (which carries a frame, an image and a
 * full breakdown), and a window where the speaker was never on camera (which
 * carries none of those). So the mapping is done once, here, along two
 * independent axes:
 *
 *   verdict  — what the system concluded, taken from `status` and nothing else
 *   evidence — whether there is a frame to show, taken from the fields present
 *
 * A component can then never ask "is this off camera?" in order to decide
 * whether to render an image, which is the mistake that would mislabel results.
 */

export type Verdict = "verified" | "uncertain" | "off_camera" | "not_found";

const VERDICT_FOR: Record<ResultStatus, Verdict> = {
  success: "verified",
  ambiguous: "uncertain",
  not_on_screen: "off_camera",
  not_found: "not_found",
  error: "not_found",
};

export interface FrameEvidence {
  frameNumber: number;
  imageName: string;
  imageUrl: string;
  timecode: string;
}

export interface BreakdownTerm {
  name: string;
  value: number;
  weight: number;
  /** value * weight / sum(weights) — the four contributions sum to the score. */
  contribution: number;
}

export interface Breakdown {
  score: number;
  weightSum: number;
  /** True in discovery mode, where the weights sum to 0.60 rather than 1.0. */
  renormalized: boolean;
  terms: BreakdownTerm[];
}

export interface ResultCard {
  key: string;
  verdict: Verdict;
  status: ResultStatus;
  /** null means no frame was selected — render a slate, never a broken image. */
  evidence: FrameEvidence | null;
  /** Can be present without evidence: discovery records when an off-camera line was said. */
  timecode: string | null;
  seconds: number | null;
  text: string | null;
  speaker: string | null;
  /** null only when the key was absent. 0 is a real, meaningful value. */
  confidence: number | null;
  faceConfidence: number | null;
  reason: AmbiguityReason | null;
  detail: string | null;
  breakdown: Breakdown | null;
}

export type RunView =
  | { kind: "target"; card: ResultCard; diagnostics: WireDiagnostics | null }
  | {
      kind: "discovery";
      onScreen: ResultCard[];
      uncertain: ResultCard[];
      offCamera: ResultCard[];
      processed: number;
      total: number;
      truncated: boolean;
      errors: WireCandidateError[];
      diagnostics: WireDiagnostics | null;
    }
  | { kind: "error"; error: AppError };

// --- helpers -----------------------------------------------------------------

/** "HH:MM:SS.mmm" -> seconds. Split from the right; hours are not width-capped. */
export function parseTimecode(timecode: string): number | null {
  const parts = timecode.split(":");
  if (parts.length < 2) return null;
  const seconds = Number(parts[parts.length - 1]);
  const minutes = Number(parts[parts.length - 2]);
  const hours = parts.length >= 3 ? Number(parts[parts.length - 3]) : 0;
  if (!Number.isFinite(seconds) || !Number.isFinite(minutes) || !Number.isFinite(hours)) {
    return null;
  }
  return hours * 3600 + minutes * 60 + seconds;
}

export function formatTimecode(totalSeconds: number): string {
  const safe = Math.max(0, totalSeconds);
  const ms = Math.floor((safe % 1) * 1000);
  const whole = Math.floor(safe);
  const hh = String(Math.floor(whole / 3600)).padStart(2, "0");
  const mm = String(Math.floor((whole % 3600) / 60)).padStart(2, "0");
  const ss = String(whole % 60).padStart(2, "0");
  return `${hh}:${mm}:${ss}.${String(ms).padStart(3, "0")}`;
}

export function buildBreakdown(wire: WireBreakdown): Breakdown {
  // Iterate `weights`, not `terms`: the backend already filtered weights down to
  // the active set, so `match` is simply absent in discovery mode.
  const entries = Object.entries(wire.weights);
  const weightSum = entries.reduce((total, [, weight]) => total + weight, 0);
  const terms: BreakdownTerm[] = entries.map(([name, weight]) => {
    const value = wire.terms[name] ?? 0;
    return {
      name,
      value,
      weight,
      contribution: weightSum > 0 ? (value * weight) / weightSum : 0,
    };
  });
  terms.sort((a, b) => b.contribution - a.contribution);
  return {
    score: wire.score,
    weightSum,
    renormalized: weightSum < 0.999,
    terms,
  };
}

function toCard(
  wire: WireExtractionResult,
  jobId: string,
  key: string,
): ResultCard {
  const timecode = wire.timestamp ?? null;
  // Evidence requires all three: a frame number alone cannot be displayed.
  const evidence: FrameEvidence | null =
    wire.frame_number !== undefined && wire.image !== undefined && timecode !== null
      ? {
          frameNumber: wire.frame_number,
          imageName: wire.image,
          imageUrl: frameUrl(wire.image, jobId),
          timecode,
        }
      : null;

  return {
    key,
    verdict: VERDICT_FOR[wire.status] ?? "not_found",
    status: wire.status,
    evidence,
    timecode,
    seconds: timecode !== null ? parseTimecode(timecode) : null,
    text: wire.text ?? null,
    speaker: wire.speaker ?? null,
    confidence: wire.confidence ?? null,
    faceConfidence: wire.face_detection_confidence ?? null,
    reason: wire.ambiguity_reason ?? null,
    detail: wire.detail ?? null,
    breakdown: wire.confidence_breakdown
      ? buildBreakdown(wire.confidence_breakdown)
      : null,
  };
}

function bucket(
  items: WireExtractionResult[],
  jobId: string,
  prefix: string,
): ResultCard[] {
  const cards = items.map((item, index) => toCard(item, jobId, `${prefix}-${index}`));
  // Backend order is already chronological; sorting makes that guaranteed
  // rather than incidental. Entries with no timestamp sort last.
  return cards.sort((a, b) => (a.seconds ?? Infinity) - (b.seconds ?? Infinity));
}

// --- the one entry point -----------------------------------------------------

export function toRunView(
  result: WireResult,
  jobId: string,
  diagnostics: WireDiagnostics | null,
): RunView {
  if (isErrorResult(result)) {
    return {
      kind: "error",
      error: {
        kind: "stage",
        httpStatus: null,
        stage: result.stage,
        reason: result.reason,
        detail: result.detail,
        ...(result.remediation ? { remediation: result.remediation } : {}),
      },
    };
  }

  if (isDiscoveryResult(result)) {
    const discovery = result as WireDiscoveryResult;
    return {
      kind: "discovery",
      onScreen: bucket(discovery.results, jobId, "on"),
      uncertain: bucket(discovery.ambiguous, jobId, "un"),
      offCamera: bucket(discovery.not_on_screen, jobId, "off"),
      processed: discovery.candidates_processed,
      total: discovery.candidates_total,
      truncated: Boolean(discovery.truncated),
      errors: discovery.errors ?? [],
      diagnostics,
    };
  }

  return { kind: "target", card: toCard(result, jobId, "target"), diagnostics };
}

/** Partial results streamed during a sweep, rendered with the same component. */
export function partialCards(
  items: WireExtractionResult[],
  jobId: string,
): ResultCard[] {
  return bucket(items, jobId, "partial");
}
