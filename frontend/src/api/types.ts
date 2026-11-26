/**
 * A 1:1 mirror of what the Python API actually sends.
 *
 * Every optional field here is `?:` — absent — never `| null`. That matches the
 * backend exactly: `ExtractionResult.to_dict` and `Job.to_dict` both build their
 * payload with `if x is not None: payload[k] = x`, so a missing value means a
 * missing *key*. Modelling those as nullable would invent a shape the server
 * never produces.
 *
 * Where a union looks closed but is not, it is left open on purpose and the
 * reason is stated. Nothing here should ever be consumed by an exhaustive
 * `switch` without a fallback.
 */

// --- enumerations ------------------------------------------------------------

export type JobState = "queued" | "running" | "done" | "failed";

export type Mode = "target" | "discovery";

export type ResultStatus =
  | "success"
  | "ambiguous"
  | "not_on_screen"
  | "not_found"
  | "error";

/**
 * The twelve real pipeline stages.
 *
 * Left open (`| (string & {})`) because the API also emits the literal `"api"`
 * from its 404 and 500 handlers, and that is not a StageName member. Autocomplete
 * still offers the known twelve.
 */
export type KnownStage =
  | "ingestion"
  | "validation"
  | "normalization"
  | "transcription"
  | "diarization"
  | "alignment"
  | "matching"
  | "clip_build"
  | "active_speaker_detection"
  | "frame_selection"
  | "frame_extraction"
  | "config";

export type Stage = KnownStage | (string & {});

export type KnownAmbiguityReason =
  | "low_confidence"
  | "low_face_confidence"
  | "weak_text_match"
  | "short_speaking_run"
  | "multiple_simultaneous_speakers"
  | "max_speakers_reached"
  | "speaker_mismatch"
  | "no_face_detected"
  | "spoken_before_onset";

export type AmbiguityReason = KnownAmbiguityReason | (string & {});

// --- request -----------------------------------------------------------------

export interface ExtractRequest {
  url: string;
  /** Omitted or null selects discovery mode. Whitespace-only is treated as null. */
  dialogue: string | null;
  max_candidates: number;
  time_budget_seconds: number;
  match_score_floor: number;
  window_pad_seconds: number;
  min_speaking_ms: number;
  detect_scene_cuts: boolean;
}

export interface JobAccepted {
  job_id: string;
  state: JobState;
  mode: Mode;
  poll: string;
}

// --- results -----------------------------------------------------------------

export interface WireBreakdown {
  score: number;
  /** `match` present only in target mode. */
  terms: Record<string, number>;
  /** Sums to 1.0 in target mode but **0.60 in discovery** — never assume a unit total. */
  weights: Record<string, number>;
}

export interface WireExtractionResult {
  status: ResultStatus;
  /** Formatted "HH:MM:SS.mmm" — a string, not a number of seconds. */
  timestamp?: string;
  frame_number?: number;
  text?: string;
  /** JSON key is `speaker`, not `speaker_id`. May be the literal "UNKNOWN". */
  speaker?: string;
  /** Can legitimately be `0.0`. Never test this for truthiness. */
  confidence?: number;
  face_detection_confidence?: number;
  /** A bare filename such as "frame_983.png", to be handed to /frames/{name}. */
  image?: string;
  ambiguity_reason?: AmbiguityReason;
  detail?: string;
  confidence_breakdown?: WireBreakdown;
}

export interface WireCandidateError {
  /** 0-based index into the candidate list. */
  candidate: number;
  /** Raw seconds, unlike every other time value in the API. */
  window: [number, number];
  reason: string;
  detail: string;
}

export interface WireTargetResult extends WireExtractionResult {
  mode?: "target";
}

export interface WireDiscoveryResult {
  /** Only ever these two at the discovery top level. */
  status: "success" | "not_found";
  results: WireExtractionResult[];
  /**
   * Holds items whose own status is `"ambiguous"` **or** `"not_on_screen"` — a
   * verified frame scoring below 0.45 lands here. Read each item's own status;
   * the array only says which section it belongs in.
   */
  ambiguous: WireExtractionResult[];
  not_on_screen: WireExtractionResult[];
  candidates_processed: number;
  candidates_total: number;
  /** Only ever `true`. Absent means false — the backend never serialises false. */
  truncated?: true;
  errors?: WireCandidateError[];
  mode?: "discovery";
}

export interface WireErrorResult {
  status: "error";
  stage: Stage;
  reason: string;
  detail: string;
  remediation?: string;
}

export type WireResult =
  | WireTargetResult
  | WireDiscoveryResult
  | WireErrorResult;

// --- job envelope ------------------------------------------------------------

export interface WireDiagnostics {
  mode?: Mode;
  fps?: number;
  duration?: number;
  frame_count?: number;
  /** "1920x1080" */
  resolution?: string;
  words?: number;
  segments?: number;
  speakers?: number;
  candidates?: number;
  /** Target mode only. */
  matches?: number;
  /** Target mode only. */
  candidates_verified?: number;
}

export interface WireProgress {
  current: number;
  total: number;
}

export interface JobSnapshot {
  job_id: string;
  state: JobState;
  url: string;
  mode: Mode;
  created_at: string;
  dialogue?: string;
  stage?: Stage;
  message?: string;
  started_at?: string;
  finished_at?: string;
  /** Always paired with `finished_at`; never sent while a job is running. */
  elapsed_seconds?: number;
  /** Discovery only, while sweeping. */
  progress?: WireProgress;
  /** Discovery only: results found so far, before the run finishes. */
  partial?: WireExtractionResult[];
  result?: WireResult;
  /** Absent when empty, which is always the case on an error path. */
  diagnostics?: WireDiagnostics;
  /** Discovery only, once finished: every line with its time span and verdict. */
  lines?: WireLine[];
  /** Discovery only: path of the normalized video, for the "check it yourself" demo. */
  video?: string;
}

/** One line of a finished discovery sweep. Times are raw seconds. */
export interface WireLine {
  index: number;
  start: number;
  end: number;
  text: string;
  speaker: string | null;
  /** "error" when detection failed for this line. */
  status: ResultStatus;
  frame_number: number | null;
  timestamp: string | null;
  confidence: number | null;
  reason: string | null;
}

export interface HealthResponse {
  status: string;
  version: string;
  /** Despite the name this is the resolved binary *path*, or null. */
  ffmpeg: string | null;
  providers: string[];
  credentials: Record<string, boolean>;
  jobs: Record<string, number>;
}

// --- narrowing ---------------------------------------------------------------

/**
 * Order matters. `mode` cannot be the discriminator: the orchestrator appends
 * it only on success paths, so an error payload has no `mode` key at all.
 */
export function isErrorResult(r: WireResult): r is WireErrorResult {
  return r.status === "error";
}

export function isDiscoveryResult(r: WireResult): r is WireDiscoveryResult {
  return Array.isArray((r as WireDiscoveryResult).results);
}

export function isTargetResult(r: WireResult): r is WireTargetResult {
  return !isErrorResult(r) && !isDiscoveryResult(r);
}
