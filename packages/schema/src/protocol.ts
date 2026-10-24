// Runner protocol schemas — the HTTP contract between bench (TS) and
// runners/<island> (Python FastAPI). Docs: docs/plan/02-architecture.md.
// Kept deliberately loose in phase 1; runners/_protocol generates matching
// Pydantic models from catalog/schema/protocol.json in phase 3.
import { z } from "zod";

// GET /info
export const RunnerInfo = z.looseObject({
  runner: z.string(), // island name, e.g. "gliner"
  protocol: z.string().default("v1"),
  models: z.array(
    z.looseObject({
      model_id: z.string(),
      version: z.string().nullable(),
      runtime: z.unknown(), // {engine, dtype, quant, device, batch}
    }),
  ),
  host: z.looseObject({
    hostname: z.string(),
    cpu: z.string(),
    ram_gb: z.number(),
    gpus: z
      .array(
        z.looseObject({
          model: z.string(),
          vram_gb: z.number(),
        }),
      )
      .default([]),
    os: z.string(),
  }),
});

// POST /v1/<track> — request side
export const TrackRequest = z.looseObject({
  model_id: z.string(),
  items: z.array(
    z.looseObject({
      item_id: z.string(),
      input: z.unknown(), // track-typed
      question: z.unknown(), // track-typed
      params: z.unknown().optional(),
    }),
  ),
});

// POST /v1/<track> — response side. compute_ms is measured inside the server
// so HTTP overhead never pollutes latency numbers (docs/plan/02).
export const TrackResponse = z.looseObject({
  model_id: z.string(),
  results: z.array(
    z.looseObject({
      item_id: z.string(),
      prediction: z.unknown(), // track-typed
      prob: z.number().nullable(),
      error: z.string().nullable(),
    }),
  ),
  compute_ms: z.number(),
  usage: z.unknown().nullable(),
});
export type RunnerInfoOut = z.output<typeof RunnerInfo>;
export type TrackRequestOut = z.output<typeof TrackRequest>;
export type TrackResponseOut = z.output<typeof TrackResponse>;
