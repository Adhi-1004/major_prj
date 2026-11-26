import type { KnownStage, Mode } from "../api/types";
import type { Verdict } from "./view";

/** Machine slugs are precise but not readable. This is the only place they become prose. */

export const VERDICT_LABEL: Record<Verdict, string> = {
  verified: "On screen",
  uncertain: "Uncertain",
  off_camera: "Off camera",
  not_found: "No match",
};

export const VERDICT_GLYPH: Record<Verdict, string> = {
  verified: "◉",
  uncertain: "◐",
  off_camera: "○",
  not_found: "✕",
};

export const VERDICT_BLURB: Record<Verdict, string> = {
  verified: "The speaker was visible and speaking in this frame.",
  uncertain: "A frame was found, but something about the evidence is unresolved.",
  off_camera: "The line is spoken, but not by anyone visible in the shot.",
  not_found: "The phrase does not appear in the transcript.",
};

const STAGE_LABEL: Record<KnownStage, string> = {
  ingestion: "Acquiring",
  validation: "Validating",
  normalization: "Normalizing",
  transcription: "Transcribing",
  diarization: "Identifying speakers",
  alignment: "Aligning",
  matching: "Matching",
  clip_build: "Building clip",
  active_speaker_detection: "Detecting speaker",
  frame_selection: "Selecting frame",
  frame_extraction: "Extracting frame",
  config: "Configuration",
};

/** Never exhaustive: the API also emits the literal "api" from its error handlers. */
export function stageLabel(stage: string): string {
  return (
    STAGE_LABEL[stage as KnownStage] ??
    stage.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase())
  );
}

/** The stages the orchestrator actually announces, in order. */
export const STAGE_ORDER: Record<Mode, KnownStage[]> = {
  target: [
    "ingestion",
    "normalization",
    "transcription",
    "diarization",
    "alignment",
    "matching",
    "active_speaker_detection",
  ],
  // Discovery never matches a phrase, so it has one stage fewer.
  discovery: [
    "ingestion",
    "normalization",
    "transcription",
    "diarization",
    "alignment",
    "active_speaker_detection",
  ],
};

const REASON_LABEL: Record<string, string> = {
  low_confidence: "Low overall confidence",
  low_face_confidence: "Low face confidence",
  weak_text_match: "Weak transcript match",
  short_speaking_run: "Short speaking run",
  multiple_simultaneous_speakers: "Several people speaking",
  max_speakers_reached: "Too many faces in frame",
  speaker_mismatch: "A different person was speaking",
  no_face_detected: "No face detected",
  spoken_before_onset: "Speaker seen only before the line began",
};

export function reasonLabel(reason: string): string {
  return (
    REASON_LABEL[reason] ??
    reason.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase())
  );
}

const TERM_LABEL: Record<string, string> = {
  match: "Match",
  asr: "Speech",
  face: "Face",
  stability: "Stability",
};

const TERM_BLURB: Record<string, string> = {
  match: "How closely the transcript matched the phrase you asked for",
  asr: "How confident the recognizer was about those words",
  face: "How confident the detector was that it saw a face",
  stability: "How much longer than the minimum the speaking run lasted",
};

export function termLabel(name: string): string {
  return TERM_LABEL[name] ?? name;
}

export function termBlurb(name: string): string {
  return TERM_BLURB[name] ?? "";
}

/** Diarization labels words that fall in a gap as the literal "UNKNOWN". */
export function speakerLabel(speaker: string): string {
  return speaker === "UNKNOWN" ? "Unattributed" : speaker.replace(/_/g, " ");
}

export function formatDuration(seconds: number): string {
  const whole = Math.floor(Math.max(0, seconds));
  const mm = String(Math.floor(whole / 60)).padStart(2, "0");
  const ss = String(whole % 60).padStart(2, "0");
  return `${mm}:${ss}`;
}
