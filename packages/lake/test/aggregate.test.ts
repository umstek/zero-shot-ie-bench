import { describe, expect, test } from "bun:test";
import type { RunItemOut as RunItem, RunMetaOut as RunMeta } from "@zsib/schema";
import { aggregateRun, percentile } from "../src/aggregate.ts";

const meta = {
  run_id: "r1",
  status: "ok",
  deferred_reason: null,
  deployment: "x.y",
  model_version: "x@1",
  served_version: null,
  where: { kind: "hosted", provider: "openrouter" },
  suite: "decision/mixed-pool",
  suite_version: 1,
  suite_hash: "sha256:0",
  params: {},
  params_hash: null,
  bench: { version: "0.1.0", git_sha: "000" },
  started_at: "2026-10-24T00:00:00Z",
  finished_at: null,
  totals: { items: 0, correct: 0, cost_usd: 0, p50_ms: 0, p95_ms: 0, errors: 0 },
  source: "bench",
} as unknown as RunMeta;

const item = (over: Partial<RunItem>): RunItem =>
  ({
    item_id: "i",
    lang: "en",
    prediction: null,
    gold: null,
    correct: true,
    score: null,
    prob: null,
    latency_ms: null,
    compute_ms: null,
    cost_usd: 0,
    usage: null,
    error: null,
    ...over,
  }) as RunItem;

describe("percentile", () => {
  test("nearest-rank on small arrays", () => {
    expect(percentile([1, 2, 3, 4], 50)).toBe(2);
    expect(percentile([1, 2, 3, 4], 95)).toBe(4);
    expect(percentile([], 50)).toBeNull();
  });
});

describe("aggregateRun", () => {
  test("accuracy, cost, latency, per-language split", () => {
    const items = [
      item({ lang: "en", correct: true, latency_ms: 10, cost_usd: 0.001 }),
      item({ lang: "en", correct: false, latency_ms: 30, cost_usd: 0.001 }),
      item({ lang: "si", correct: true, latency_ms: 20, cost_usd: 0.002 }),
      item({ lang: "si", correct: null, error: "timeout", latency_ms: 40 }),
    ];
    const agg = aggregateRun(meta, items);
    expect(agg.items).toBe(4);
    expect(agg.correct).toBe(2);
    expect(agg.accuracy).toBeCloseTo(2 / 3);
    expect(agg.cost_usd).toBeCloseTo(0.004);
    expect(agg.errors).toBe(1);
    expect(agg.p50_ms).toBe(20); // sorted [10,20,30,40], nearest-rank: ceil(.5*4)=2 → index 1
  });

  test("unscored run yields null accuracy", () => {
    const agg = aggregateRun(meta, [item({ correct: null })]);
    expect(agg.accuracy).toBeNull();
    expect(agg.correct).toBeNull();
  });
});
