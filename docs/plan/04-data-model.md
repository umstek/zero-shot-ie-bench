# 04 — Data model

*2026-10-24. Shapes of the catalog, suite, run and observation files. These are
sketches; the source of truth will be Zod in `packages/schema`. Field names are
proposals.*

## Entities and how they relate

```
provider 1─* deployment *─1 model_version *─1 model
                 │                                  ╲ claims: tracks[], modalities[], languages[]
machine 1─* (local deployment run)
suite@hash ─┐
deployment ─┼─► run 1─* run_item
machine?  ──┘
provider 1─* observation   (dated: price, zdr, context, quant, retention)
```

- **model**: identity of the weights/service family. Stable fields only.
- **model_version**: HF revision or dated API slug. Never deleted.
- **deployment**: *how* you reach a model_version: `hosted` (provider + endpoint id +
  adapter) or `local` (runner island + runtime config + requirements). The bench
  always targets a deployment.
- **run**: one deployment × one suite × one machine/provider × one moment.

## `catalog/models/<org>/<model>.yaml`

```yaml
id: base-labs/elephant-1
family: elephant
org: Base Labs
license: MIT
open_weights: true
params_b: 9
claims:                       # what the vendor says; the board contrasts with measured
  tracks: [decision, score]
  modalities: [text, image]
  languages: multilingual
status: active                # active | culled | retired  (culled = still shown in archive views)
links: { card: https://huggingface.co/baselabs/elephant-1, paper: null }
versions:
  - id: elephant-1@2026-10    # stable key used by runs
    hf_revision: null
    api_slug: baseten/elephant-1
    released: 2026-10-20
deployments:
  - id: elephant-1.openrouter-free
    kind: hosted
    version: elephant-1@2026-10
    provider: openrouter
    endpoint: baseten/elephant-1:free
    adapter: openrouter.systemone
    notes: plain slug has no endpoints; 20 req/min
  - id: elephant-1.local-bf16
    kind: local
    version: elephant-1@2026-10
    runner: small-decisions
    runtime: { engine: torch, dtype: bf16, quant: null }
    requires: { vram_gb: 20, ram_gb: 32, accel: cuda }   # → deferred on seoul
```

## `catalog/machines/<id>.yaml`

```yaml
id: seoul
kind: owned                    # owned | rented | ci
cpu: "…"
ram_gb: 32
gpus: []                       # [{model: RTX 4090, vram_gb: 24}]
os: Windows 11
cost_per_hour_usd: 0.05        # D5: owned = (hardware ÷ expected life) + power; rented = list price
cost_basis: { hardware_usd: 0, life_hours: 0, power_usd_per_hour: 0 }   # owned only; cost_per_hour_usd is derived from it
fingerprint_match: { hostname: "…" }   # how /info reports are matched to this file
```

## `suites/<track>/<suite-id>/suite.yaml` + items

```yaml
id: decision/mixed-pool
version: 3
track: decision
languages: [en, si, ta, hi, ja, ar, sw, …]
scorer: exact-choice           # module in bench/src/tracks/decision/
items_license: CC-BY-4.0
```

`items/<lang>.jsonl`, one parallel item per line:

```json
{"item_id":"mp-017","lang":"si","tier":"rare","input":{"text":"…","images":[]},"question":{"type":"choice","options":["…"]},"gold":"…"}
```

`suite_hash` = hash of `suite.yaml` + all item files, computed by the bench and
recorded on each run. Editing a suite bumps `version`. Old runs keep pointing at the
old hash.

## `runs/<track>/<yyyy-mm>/<run-id>.meta.json`

```json
{
  "run_id": "01JBZ…",
  "status": "ok",                         // ok | partial | failed | deferred
  "deferred_reason": null,                // e.g. "requires vram_gb 20 > 0 on seoul"
  "deployment": "elephant-1.openrouter-free",
  "model_version": "elephant-1@2026-10",
  "served_version": "baseten/elephant-1:free#2026-10-24",   // what the API/runner reported
  "where": {"kind": "hosted", "provider": "openrouter"},    // or {"kind":"local","machine":"seoul","runtime":{…}}
  "suite": "decision/mixed-pool", "suite_version": 3, "suite_hash": "sha256:…",
  "params": {"threshold": null, "temperature": 0}, "params_hash": "…",
  "bench": {"version": "0.1.0", "git_sha": "20e4825"},
  "started_at": "2026-10-24T13:02:11+05:30", "finished_at": "…",
  "totals": {"items": 48, "correct": 43, "cost_usd": 0.0, "p50_ms": 18, "p95_ms": 41, "errors": 0},
  "source": "bench"                       // bench | legacy-v0 | import:<name>
}
```

`<run-id>.items.jsonl`, one per item:

```json
{"item_id":"mp-017","lang":"si","prediction":"…","gold":"…","correct":true,"score":null,"prob":0.93,"latency_ms":17,"compute_ms":null,"cost_usd":0,"usage":{"in":412,"out":0},"error":null}
```

Track-specific fields (span lists for `ner`, field maps with `p_null`/offsets for
`typed-extract`, rankings for `rerank`) go in a typed `prediction`. The scorer adds
per-item metric fields.

## `observations/<provider>/<yyyy-mm>.jsonl`

```json
{"at":"2026-10-24T08:00:00Z","provider":"openrouter","endpoint":"openai/gpt-6-luna-decisions","price_in_per_m":0.10,"price_out_per_m":0,"context":1050000,"zdr":false,"retains_prompts":true,"quantization":null,"source":"api:/models/{id}/endpoints"}
```

Only appended when something changed since the last observation, so the file stays
small and every line marks a change.

## Collections after the build (`build/`, JSON written by `packages/lake`)

| File | Grain | Used by |
|---|---|---|
| `<track>.items/<run-id>.json` | one row per item of one run, joined with catalog dims; **loaded lazily** per run | language heatmaps, error drill-down |
| `<track>.runs.json` | one row per run with metrics per language and overall | history lines, explorer |
| `deployments.json` | catalog flattened + latest observation + feasibility per machine | facets, picks, planner |
| `observations.json` | all observations | price/ZDR history |

One collection per track, as you asked (R7), so a NER F1 and a decision accuracy are
never in the same column. The board joins them only through `deployments`.
