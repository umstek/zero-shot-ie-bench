// Pure aggregation over run items — no filesystem here so tests can drive it.
// Docs: docs/plan/02-architecture.md (metrics per track), 04 (run shapes).
import type { RunItemOut as RunItem, RunMetaOut as RunMeta } from "@zsib/schema";

export type LangMetrics = {
  lang: string;
  items: number;
  correct: number | null;
  accuracy: number | null;
};

export type RunAggregate = {
  run_id: string;
  deployment: string;
  model_version: string;
  suite: string;
  suite_version: number;
  status: string;
  source: string;
  started_at: string;
  where: { kind: string; provider?: string; machine?: string };
  items: number;
  correct: number | null;
  accuracy: number | null;
  cost_usd: number;
  errors: number;
  p50_ms: number | null;
  p95_ms: number | null;
  per_language: LangMetrics[];
  // legacy-v0 bookkeeping: row order in the v0 README table (arrival order),
  // the sort tiebreak where v0 used the repo's arrival history
  v0_rank: number | null;
};

export function percentile(sortedValues: number[], p: 50 | 95): number | null {
  if (sortedValues.length === 0) return null;
  // nearest-rank percentile
  const rank = Math.ceil((p / 100) * sortedValues.length);
  const idx = Math.min(Math.max(rank, 1), sortedValues.length) - 1;
  return sortedValues[idx] ?? null;
}

export function aggregateRun(meta: RunMeta, items: RunItem[]): RunAggregate {
  const latencies = items
    .map((i) => i.latency_ms)
    .filter((v): v is number => typeof v === "number")
    .sort((a, b) => a - b);
  const scored = items.filter((i) => typeof i.correct === "boolean");
  const correct = scored.filter((i) => i.correct === true).length;

  // legacy summary runs (source legacy-v0) may carry only totals — e.g. the
  // v0 NER columns recorded rates, not per-item predictions. Per-item truth
  // wins when present; recorded totals are the fallback.
  const totalsCorrect =
    scored.length === 0 && typeof meta.totals.correct === "number" ? meta.totals.correct : null;
  const totalsItems = typeof meta.totals.items === "number" ? meta.totals.items : 0;
  const effectiveCorrect = scored.length > 0 ? correct : totalsCorrect;
  const effectiveDenominator = scored.length > 0 ? scored.length : totalsItems;
  const accuracy =
    effectiveCorrect !== null && effectiveDenominator > 0
      ? effectiveCorrect / effectiveDenominator
      : null;

  const byLang = new Map<string, { items: number; correct: number; scored: number }>();
  for (const i of items) {
    const lang = i.lang ?? "unknown";
    const agg = byLang.get(lang) ?? { items: 0, correct: 0, scored: 0 };
    agg.items += 1;
    if (typeof i.correct === "boolean") {
      agg.scored += 1;
      if (i.correct) agg.correct += 1;
    }
    byLang.set(lang, agg);
  }

  return {
    run_id: meta.run_id,
    deployment: meta.deployment,
    model_version: meta.model_version,
    suite: meta.suite,
    suite_version: meta.suite_version,
    status: meta.status,
    source: meta.source,
    started_at: meta.started_at,
    where: {
      kind: meta.where.kind,
      provider: meta.where.kind === "hosted" ? meta.where.provider : undefined,
      machine: meta.where.kind === "local" ? meta.where.machine : undefined,
    },
    items: items.length,
    correct: effectiveCorrect,
    accuracy,
    cost_usd: items.reduce((sum, i) => sum + (i.cost_usd ?? 0), 0) || meta.totals.cost_usd,
    errors: items.filter((i) => typeof i.error === "string").length,
    p50_ms: percentile(latencies, 50) ?? meta.totals.p50_ms ?? null,
    p95_ms: percentile(latencies, 95) ?? meta.totals.p95_ms ?? null,
    v0_rank:
      typeof (meta as { v0_rank?: unknown }).v0_rank === "number"
        ? ((meta as { v0_rank?: number }).v0_rank ?? null)
        : null,
    per_language: [...byLang.entries()]
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([lang, agg]) => ({
        lang,
        items: agg.items,
        correct: agg.scored > 0 ? agg.correct : null,
        accuracy: agg.scored > 0 ? agg.correct / agg.scored : null,
      })),
  };
}
