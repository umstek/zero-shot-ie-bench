// Run schemas — append-only artifacts under runs/<track>/<yyyy-mm>/.
// Docs: docs/plan/04-data-model.md. Track-specific prediction payloads stay
// open (typed per track by the scorer); the fields below are the contract.
import { z } from "zod";

export const RunStatus = z.enum(["ok", "partial", "failed", "deferred"]);

export const RunWhere = z.discriminatedUnion("kind", [
  z.looseObject({
    kind: z.literal("hosted"),
    provider: z.string(),
  }),
  z.looseObject({
    kind: z.literal("local"),
    machine: z.string(), // catalog/machines/<id>.yaml
    runtime: z.unknown(),
  }),
]);

export const RunMeta = z.looseObject({
  run_id: z.string().min(1),
  status: RunStatus,
  deferred_reason: z.string().nullable(),
  deployment: z.string(),
  model_version: z.string(),
  served_version: z.string().nullable(),
  where: RunWhere,
  suite: z.string(),
  suite_version: z.number(),
  suite_hash: z.string(),
  params: z.unknown(),
  params_hash: z.string().nullable(),
  bench: z.looseObject({
    version: z.string(),
    git_sha: z.string(),
  }),
  started_at: z.string(),
  finished_at: z.string().nullable(),
  totals: z.looseObject({
    items: z.number(),
    correct: z.number().nullable(),
    cost_usd: z.number(),
    p50_ms: z.number().nullable(),
    p95_ms: z.number().nullable(),
    errors: z.number(),
  }),
  source: z.string(), // bench | legacy-v0 | import:<name>
});

export const RunItem = z.looseObject({
  item_id: z.string().min(1),
  lang: z.string(),
  prediction: z.unknown(),
  gold: z.unknown(),
  correct: z.boolean().nullable(),
  score: z.number().nullable(),
  prob: z.number().nullable(),
  latency_ms: z.number().nullable(),
  compute_ms: z.number().nullable(), // measured inside the runner, not over HTTP
  cost_usd: z.number(),
  usage: z
    .looseObject({
      in: z.number().nullable(),
      out: z.number().nullable(),
    })
    .nullable(),
  error: z.string().nullable(),
});
export type RunMetaOut = z.output<typeof RunMeta>;
export type RunItemOut = z.output<typeof RunItem>;
