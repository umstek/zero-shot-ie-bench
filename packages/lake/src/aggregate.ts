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
    correct: scored.length > 0 ? correct : null,
    accuracy: scored.length > 0 ? correct / scored.length : null,
    cost_usd: items.reduce((sum, i) => sum + (i.cost_usd ?? 0), 0),
    errors: items.filter((i) => typeof i.error === "string").length,
    p50_ms: percentile(latencies, 50),
    p95_ms: percentile(latencies, 95),
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
