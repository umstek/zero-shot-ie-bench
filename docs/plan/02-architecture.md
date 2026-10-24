# 02 — Architecture

*2026-10-24. How the pieces fit together. Requirements are cited as R1–R12
([01](01-requirements.md)). Decisions D1–D6 are settled in [00](00-index.md).*

## The one-paragraph version

**Data is the spine; code is the plumbing.** A curated **catalog** (models, versions,
providers, deployments, machines) plus versioned **suites** (questions per track and
language) are the inputs. A single TypeScript **bench** runs a suite against a
**deployment** and appends an immutable **run** to the lake. A deployment is either a
hosted API (called in-process by a TS adapter) or a local model (a Python **runner**
serving one HTTP protocol). A build step turns the lake into Parquet per track. The
static **board** queries that Parquet in the browser. Everything people read (README
tables, counts, charts) is generated from it.

```
            catalog/  (models, providers, deployments, machines — YAML, curated)
            suites/   (track × language items — JSONL, versioned)
                │
                ▼
   ┌──────────── bench (TS + Bun) ─────────────┐
   │ provider adapters ──► hosted APIs          │      runners/<island>  (Python, uv)
   │ http runner client ─────────────────────────────►  FastAPI /v1/<track> + Gradio UI
   │ scorers per track, cost, timing            │      one venv per dependency island
   └──────────────┬────────────────────────────┘
                  ▼ append-only
            runs/<track>/…  (meta JSON + item JSONL)   observations/ (dated price/ZDR)
                  │
                  ▼  lake build (DuckDB)
            build/<track>.parquet  ──►  apps/board (DuckDB-WASM + ECharts, static)
                                   ──►  generated README sections, charts
```

## The three dimensions (R7, R8)

| Dimension | Values | Where it lives |
|---|---|---|
| **Track** = task × input modality | `decision`, `score`, `classify`, `ner`, `relation`, `typed-extract` (Clairic-class), `schema-extract` (NuExtract-class), `rerank`, later `event`; modality `text`, `image`, `audio`, `video` | `catalog/tracks.yaml`; a track has its own **metric set** and its own **collection** (`runs/<track>/`, `build/<track>.parquet`) |
| **Language** | the selected set (today: 9 non-English + English) | a **column** inside every track; suites ship **parallel items** (same `item_id` in every language), so differences across languages are real and not caused by different questions |
| **Deployment** | model-version × (provider \| local runtime on a machine) | `catalog/models/…` + `catalog/machines/…`; recorded on every run |

**Claimed vs. measured.** A model's catalog entry *claims* tracks and modalities, for
example "Clef-omni: audio + video". The board shows measured cells where they exist
and greys out claimed-but-unmeasured ones. That gap is itself a backlog.

A model used *secondarily* as something else, such as a reranker used as a classifier,
gets a deployment entry with `used_as: classify`. Its native track and its borrowed
track stay separate rows.

Metrics differ by track, so a cross-track score is never averaged:

| Track | Primary | Secondary |
|---|---|---|
| decision / classify | accuracy | Brier, ECE (calibration — the AA "every model is overconfident" finding), p50/p95 latency |
| ner / relation | span-level micro-F1 (strict + partial) | per-label F1 |
| typed-extract / schema-extract | field-value accuracy | null-precision (`p_null` honesty), offset-grounding rate |
| rerank | nDCG@k, MRR | — |

Cost per item, latency, and failure rate apply to every track.

## Run identity and history (R3, R10)

A run is identified by the tuple below. Changing **any** field produces a new series
point, never an overwrite.

`model_id · model_version · provider | (machine_id + runtime) · suite_id@suite_hash ·
params_hash · runner_version · bench_git_sha · started_at`

- **model_version**: the HF revision SHA for weights, or the dated API slug
  (`pplx-decider-v1.2-20261016`) / `X-Model-Version` header for APIs. When a provider
  exposes none, record `unversioned` plus a response fingerprint. A score step that
  lines up with a version boundary is labelled; one that doesn't is flagged as a
  **silent change** (a probable nerf).
- **runtime** (local): `{engine: torch|onnx|llama.cpp|vllm, dtype, quant, device,
  batch}`, as reported by the runner's `/info`.
- **Canary suite** (~24 items per track, all languages) re-run on a schedule. It costs
  cents and gives the history line its density. Full pools only run on arrival or on
  request.

## Local runners (R1, R2, R4)

- `runners/<island>/` — **one independent uv project per dependency island** (own
  `pyproject.toml`, `uv.lock`, `.python-version`, `.venv`). Never a uv workspace
  (see [01](01-requirements.md), disagreement 1).
- Every runner imports `runners/_protocol` (a path dependency) and exposes:
  - `GET /info` → models served, revisions, runtime, a host fingerprint (CPU, RAM, GPU,
    VRAM, OS) that the bench matches to a `machine_id`
  - `POST /v1/<track>` → a track-typed request/response. The response carries
    **`compute_ms`** measured inside the server, so HTTP overhead never pollutes the
    latency numbers
  - `GET /healthz`, plus a **Gradio UI** mounted on the same app
    (`gr.mount_gradio_app`). One process serves both the API and the self-host UI.
    Dockerfile per runner for cloud hosting.
- **Feasibility is data (R2).** Each deployment declares `requires: {vram_gb, ram_gb,
  accel}`. When the bench targets a machine that doesn't meet it, or the runner OOMs,
  the bench writes a run with `status: deferred` and a reason. "What can I run now that
  the GPU box exists?" is then a query over `status = deferred` × new machine specs.
  No separate registry is needed.
- Local cost per item = `machine.$_per_hour × wall_time / items`. A machine you own
  uses an amortised rate (hardware ÷ expected life + power) set once in its machine
  file; rented machines use list price. Local cost is never $0 (D5).
- **No Python bench driver (D1).** Runners only serve models. A machine that benches
  needs Bun installed next to uv.

## Hosted side (R5, R6)

- **`bench/`** (TS + Bun): CLI (`bench run`, `bench canary`, `bench sync-metadata`),
  provider adapters (one module per *contract*, not per model:
  `systemone` (TypeSafe/Clef/Kev/Celeris…), `openrouter`, `workers-ai`, `fastino`,
  `scaledown`, `typellm`…), the HTTP client for local runners, scorers per track, and
  measured cost from provider usage fields.
- `bench sync-metadata` pulls provider catalogs (e.g. the OpenRouter models and
  endpoints APIs) and **appends dated observations** (price, context, ZDR/retention,
  quantization). It never edits the catalog. Price history and "this endpoint turned
  ZDR-unsafe on date X" then come for free (R11).
- **`apps/playground`**: a Bun full-stack server (`Bun.serve` + React via HTML
  imports). It imports the same adapters from `bench`, so a provider is wired once.
  Keys stay server-side. It also lists live local runners, so one UI can try both.

## Board (R9)

Static site, no backend: **Vite + React + TanStack Router/Table + DuckDB-WASM over
Parquet + Apache ECharts** (Observable Plot for the odd small-multiples view). Host on
Cloudflare Pages or GitHub Pages; it runs equally well offline from `build/`.

Views, in priority order:

1. **Picks**: generated answers to concrete questions (cheapest ≥90 %, fastest local
   on CPU, best rare-language, best open-weights, ZDR-safe only…)
2. **Explorer**: faceted table (track, modality, language, local/hosted, GPU needed,
   ZDR, license, price, size) with columns picked per track
3. **Model page**: every deployment of one model, its history line with version
   markers, and its language heatmap
4. **Self-host vs API**: cost per 1k items against volume; the break-even point for
   your machines
5. **Provider picker**: same model across providers (price, ZDR, latency,
   stability)
6. **GPU planner**: which deferred runs a candidate machine would unlock

## Stack decisions

| Concern | Pick | Why / rejected |
|---|---|---|
| TS runtime, package manager, tests | **Bun 1.4.x**: workspaces + `catalog:` versions + isolated linker; `bun test` | one tool; Vitest not needed |
| Schema SSOT | **Zod 4** in `packages/schema` → `z.toJSONSchema()` → JSON Schema in `catalog/schema/` → Pydantic via `datamodel-codegen` for runners | one definition, both languages; CI fails on drift |
| Lint/format | Biome (TS), Ruff (Py) | |
| Lake | per-run **JSONL** (git-diffable) → **DuckDB** builds Parquet per track | `@duckdb/node-api` under Bun is believed to work (Bun fixed the DuckDB N-API crash in 1.2.2), **not yet tested on this box → spike first**. Fallback: DuckDB CLI in the `just` recipe |
| Charts | **ECharts 6** (+ Observable Plot) | ECharts covers scatter, heatmaps, time series and is well maintained; uPlot dropped (01, disagreement 3); Recharts-class libraries are too weak for heatmaps |
| Board build | Vite + React + TanStack (D2) | Evidence.dev (SQL + Markdown → static, DuckDB-WASM) rejected: the custom decision views (picks, break-even, planner) outgrow it |
| Suite media | Git LFS (D3) | an HF dataset pointer only once media passes ~1 GB or video suites arrive |
| Local UI | **Gradio 6**, mounted on the runner's FastAPI app | your pick (R4); one process per runner |
| Task runner | **`just`** at the root, shelling to `bun run …` and `uv run --project runners/<x> …` | moon's Python toolchain is still marked unstable; Pants doesn't run on Windows; add Turborepo only once CI gets slow |
| Interop | optional exporter to **Every Eval Ever** (MIT schema, Jun 2026) | not the native format (it doesn't model deployments × machines × cost), but an export is cheap and makes results citable |

## Invariants CI enforces

1. `just render && git diff --exit-code`: generated README sections, counts and charts
   match the lake.
2. Every catalog, suite and run file validates against `catalog/schema/`.
3. `runs/` is append-only: CI rejects a PR that modifies or deletes an existing run
   file.
4. TS packages import only `packages/*`. Runners never import each other, only
   `runners/_protocol`.
