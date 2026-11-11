import type { Deployment } from "../types";

export function fmtPct(v: number | null | undefined, digits = 1): string {
  if (v == null) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

export function fmtMs(ms: number | null | undefined): string {
  if (ms == null) return "—";
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
}

/** ISO timestamp (mixed offsets in the lake) → plain YYYY-MM-DD. */
export function fmtDate(iso: string): string {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return iso;
  return new Date(t).toISOString().slice(0, 10);
}

/** compact $/1k for table cells; fmtCostPer1k is the verbose card variant. */
export function fmtUsdCompact(perItem: number): string {
  const per1k = perItem * 1000;
  return `$${per1k < 0.01 ? per1k.toFixed(4) : per1k.toFixed(2)}`;
}

export function fmtCostPer1k(perItem: number): string {
  return `${fmtUsdCompact(perItem)} / 1k items`;
}

/** provider for hosted deployments, runner for local ones. */
export function platformOf(d: Deployment): string {
  return d.provider ?? d.runner ?? "—";
}

export function zdrLabel(v: boolean | undefined): string {
  if (v === true) return "ZDR safe";
  if (v === false) return "not ZDR";
  return "unknown";
}
