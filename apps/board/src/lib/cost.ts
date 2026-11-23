import type { Run } from "../types";

/** Per-item cost figure for a run with measured positive spend: the legacy-v0
 *  per-question bookkeeping when present (house rule, see bench costTable),
 *  else run cost ÷ items. null when the run carries no measured spend —
 *  free-tier and unpriced runs never rank on price. */
export function measuredPerItem(run: Run): number | null {
  if (run.cost_usd <= 0) return null;
  return run.v0_per_question ?? run.cost_usd / Math.max(run.items, 1);
}
