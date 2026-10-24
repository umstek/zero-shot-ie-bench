# Methodology

How tracks are scored, what "deferred" means, and the cost model.
Source of truth: `docs/plan/02-architecture.md`; schemas in
`packages/schema`, enforced by `bun run validate`.

## Tracks and metrics

A track is a task × input modality (`catalog/tracks.yaml`); each has its
own suite collection (`suites/<track>/…`), run collection
(`runs/<track>/…`) and lake output (`build/<track>.runs.json`). Metrics
are never averaged across tracks.

| Track | Primary | Secondary |
|---|---|---|
| decision / classify | accuracy | Brier, ECE (10-bin, Guo 2017), p50/p95 latency |
| ner / relation | span micro-F1 (strict) | partial F1 (overlap > 0, greedy IoU), per-label F1 |
| typed-extract / schema-extract | field-value accuracy | p_null honesty, offset grounding |
| rerank | nDCG@k, MRR | — |

Cost per item, latency and failure rate apply to every track. Local
latency is the runner's server-measured `compute_ms`; hosted latency is
client-side around the adapter call, with the provider's own usage block
recorded per item.

## Suites

Suites are versioned pools of parallel items (`same item_id in every
language file`), hashed by `sha256(suite.yaml + items/*)`; the hash is
recorded on every run, so editing a suite never rewrites history. The
canary suites (~24 items, all languages) are the cheap scheduled
nerf-detector; full pools run on arrival or on request.

## Runs

`runs/` is append-only (CI-enforced by `scripts/check-append-only.ts`).
A run's identity is the tuple
`model · version · provider|machine+runtime · suite@hash · params ·
bench version · git sha · started_at` — rebenching adds history points,
never overwrites. Aggregates are build outputs (`bun run lake`), not
data.

### Deferred

A deployment declares `requires: {vram_gb, ram_gb, accel}`. When the
target machine does not meet it, or the runner is unreachable, the bench
writes a run with `status: deferred` and the reason — "what unlocks when
the GPU box exists" is a query over deferred runs, not a rework.

## Cost

Hosted: the provider's own usage accounting from the run, never
reconstructed from list prices (the one exception, four providers that
report tokens but no $, is recorded in the run meta as
`v0_cost_derived` legacy bookkeeping). Local: `machine.$/h × wall-time
÷ items`, with owned machines carrying an amortised rate (hardware ÷
expected life + power) — local cost is never shown as $0.
