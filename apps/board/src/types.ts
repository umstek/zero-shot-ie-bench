// Shapes of the lake build output (build/*.json), as produced by packages/lake.
// Plain structural types — the lake already validated the source data.

export type Track = "decision" | "ner";
export type DeploymentKind = "hosted" | "local";

export interface PerLanguageStat {
  lang: string;
  items: number | null;
  correct: number | null;
  accuracy: number | null;
}

export interface RunWhere {
  kind: DeploymentKind;
  provider?: string;
  machine?: string;
}

export interface Run {
  run_id: string;
  deployment: string;
  model_version: string;
  suite: string;
  suite_version: number;
  status: string;
  source?: string;
  started_at: string;
  where: RunWhere;
  items: number;
  correct: number | null;
  accuracy: number | null;
  cost_usd: number;
  errors: number;
  p50_ms: number | null;
  p95_ms: number | null;
  per_language: PerLanguageStat[] | null;
  v0_tiers?: Record<string, number> | null;
  v0_cells?: Record<string, string> | null;
}

/** A run tagged with the track it belongs to (decision + ner lakes merged). */
export interface RunWithTrack extends Run {
  track: Track;
}

export interface DeploymentLatestRun {
  run_id: string;
  suite: string;
  started_at: string;
  accuracy: number | null;
  cost_usd: number;
}

export interface DeploymentClaims {
  tracks: string[];
  modalities: string[];
  languages: string | string[] | null;
}

export interface DeploymentRequires {
  /** null = not estimated in the catalog yet */
  vram_gb: number | null;
  ram_gb: number | null;
  accel: string;
  note?: string;
}

export interface Deployment {
  deployment_id: string;
  model_id: string;
  display: string;
  /** Absent on a few catalog entries — treat as "unknown". */
  zdr_safe?: boolean;
  family: string;
  org: string;
  license: string;
  open_weights: boolean;
  status: string;
  claims: DeploymentClaims;
  kind: DeploymentKind;
  version: string;
  /** hosted only */
  provider?: string;
  /** local only */
  runner?: string;
  requires?: DeploymentRequires;
  feasible_on?: Record<string, boolean>;
  latest_run?: DeploymentLatestRun;
}
