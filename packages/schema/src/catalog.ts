// Catalog entity schemas — the source of truth for catalog/** data files.
// Docs: docs/plan/04-data-model.md. Objects are deliberately loose
// (passthrough) so hand-curated extras and transitional fields don't fail
// validation; required structure below is what CI enforces.
import { z } from "zod";

export const Modality = z.enum(["text", "image", "audio", "video"]);

export const ModelStatus = z.enum(["active", "culled", "retired"]);

const dateish = z.string(); // ISO date or date-time; free enough for hand edits

export const Claims = z.looseObject({
  tracks: z.array(z.string()).min(1),
  modalities: z.array(Modality).min(1),
  languages: z.union([z.string(), z.array(z.string())]).nullable(),
});

export const ModelVersion = z.looseObject({
  id: z.string().min(1), // stable key used by runs, e.g. "elephant-1@2026-10"
  hf_revision: z.string().nullable(), // pinned SHA when the engine pins one
  api_slug: z.string().nullable(), // dated API slug for hosted versions
  released: dateish.nullable(),
});

// unknown runtime/requirement facts may be absent or null
const Runtime = z.looseObject({
  engine: z.string().nullable().optional(), // torch | onnx | llama.cpp | vllm | …
  dtype: z.string().nullable().optional(),
  quant: z.string().nullable().optional(),
  device: z.string().nullable().optional(),
  batch: z.number().nullable().optional(),
});

const Requires = z.looseObject({
  vram_gb: z.number().nullable().optional(),
  ram_gb: z.number().nullable().optional(),
  accel: z.string().nullable().optional(), // cuda | cpu | mps | …
});

export const HostedDeployment = z.looseObject({
  id: z.string().min(1),
  kind: z.literal("hosted"),
  version: z.string(), // ModelVersion.id
  provider: z.string(), // catalog/providers/<slug>.yaml
  endpoint: z.string(), // exact endpoint/model id the adapter calls
  adapter: z.string(), // bench/src/providers module, e.g. "openrouter.systemone"
  display: z.string().optional(), // v0 README/result row name
  notes: z.string().optional(),
});

export const LocalDeployment = z.looseObject({
  id: z.string().min(1),
  kind: z.literal("local"),
  version: z.string(),
  runner: z.string(), // runners/<island>
  runtime: Runtime,
  requires: Requires.nullable(),
  display: z.string().optional(), // v0 README/result row name
  notes: z.string().optional(),
});

export const Deployment = z.discriminatedUnion("kind", [HostedDeployment, LocalDeployment]);

export const Model = z.looseObject({
  id: z.string().min(1), // "<org-slug>/<model-slug>"
  family: z.string().min(1),
  org: z.string(),
  license: z.string().nullable(),
  open_weights: z.boolean(),
  params_b: z.number().nullable(),
  claims: Claims,
  status: ModelStatus,
  links: z
    .looseObject({
      card: z.string().nullable(),
      paper: z.string().nullable(),
    })
    .nullable(),
  versions: z.array(ModelVersion).min(1),
  deployments: z.array(Deployment),
  notes: z.string().optional(),
  // transitional v0 bookkeeping (docs/plan/05 phase 1 acceptance); dropped
  // once runs/ carries benched-ness
  v0: z
    .looseObject({
      benched: z.boolean().optional(),
    })
    .optional(),
});

export const ProviderContract = z.looseObject({
  name: z.string(),
  kind: z.string().default("systemone"), // systemone | chat | rerank | …
  endpoint: z.string().optional(),
  notes: z.string().optional(),
});

export const Provider = z.looseObject({
  id: z.string().min(1),
  name: z.string(),
  base_url: z.string().nullable(),
  auth: z.array(z.string()).nullable(), // env var names the adapter reads
  docs: z.string().nullable(),
  zdr: z.string().nullable(), // what we know about zero-data-retention
  contracts: z.array(ProviderContract).default([]),
  notes: z.string().optional(),
});

export const Gpu = z.looseObject({
  model: z.string(),
  vram_gb: z.number(),
});

export const Machine = z.looseObject({
  id: z.string().min(1),
  kind: z.enum(["owned", "rented", "ci"]),
  cpu: z.string(),
  ram_gb: z.number(),
  gpus: z.array(Gpu).default([]),
  os: z.string(),
  // D5: owned machines carry an amortised rate; null until the owner sets it.
  cost_per_hour_usd: z.number().nullable(),
  cost_basis: z
    .looseObject({
      hardware_usd: z.number().nullable(),
      life_hours: z.number().nullable(),
      power_usd_per_hour: z.number().nullable(),
    })
    .nullable(),
  fingerprint_match: z.looseObject({
    hostname: z.string().nullable(),
  }),
});

export const TrackMetrics = z.looseObject({
  primary: z.array(z.string()).min(1),
  secondary: z.array(z.string()).default([]),
});

export const Track = z.looseObject({
  id: z.string().min(1),
  description: z.string(),
  metrics: TrackMetrics,
  live: z.boolean().default(false), // exercised by legacy runs today
});

export const Language = z.looseObject({
  code: z.string().min(1),
  name: z.string(),
  script: z.string().nullable(),
  tier: z.enum(["popular", "medium", "rare"]).nullable(),
});
export type ModelOut = z.output<typeof Model>;
export type ProviderOut = z.output<typeof Provider>;
export type MachineOut = z.output<typeof Machine>;
export type TrackOut = z.output<typeof Track>;
export type LanguageOut = z.output<typeof Language>;
