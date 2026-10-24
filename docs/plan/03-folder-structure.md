# 03 — Folder structure

*2026-10-24. Target layout of this repo as a Bun + uv monorepo. Why each piece exists
is in [02](02-architecture.md). Every top-level folder has exactly one kind of
content: data, TS code, Python code, or docs.*

```
zero-shot-ie-bench/
├── package.json                # THE entry point: Bun workspaces + `catalog:` versions + scripts
│                               #   (bun run gen / validate / typecheck / test / lint / lake / render / bench / runner / board)
├── scripts/                    # bodies of the multi-step scripts, e.g. runner.ts (picks an island, calls uv), render.ts
├── bunfig.toml                 # [install] linker = "isolated"
├── tsconfig.base.json
├── biome.json
├── .gitattributes              # LFS patterns for suites/**/assets/**
├── ruff.toml                   # shared Python lint config (each runner extends it)
├── README.md                   # ~200 lines; tables between <!-- gen:… --> markers are generated
├── CHANGELOG.md                # one entry per release / arrival round (replaces README narrative bullets)
│
├── catalog/                    # ── DATA: curated, hand-edited, schema-validated ──
│   ├── schema/                 #   GENERATED JSON Schema (from packages/schema) — do not edit
│   ├── tracks.yaml             #   task × modality definitions + metric set per track
│   ├── languages.yaml          #   the selected language set, script, tier (rare/common)
│   ├── providers/              #   one file per provider: contract, base URL, ZDR policy, docs link
│   │   ├── openrouter.yaml
│   │   ├── cloudflare-workers-ai.yaml
│   │   └── typesafe.yaml …
│   ├── models/                 #   one file per model; versions + deployments live inside
│   │   ├── fastino/gliner2.5-multi.yaml
│   │   ├── typesafe/jev.yaml
│   │   └── base-labs/elephant-1.yaml …
│   └── machines/               #   one file per machine: CPU, RAM, GPU, VRAM, OS, $/h, owned|rented
│       ├── seoul.yaml          #   (this desktop)
│       └── gpu-box-1.yaml …
│
├── suites/                     # ── DATA: versioned question pools ──
│   └── <track>/<suite-id>/     #   e.g. decision/mixed-pool, decision/canary, ner/transcripts-v1
│       ├── suite.yaml          #   id, version, track, languages, scorer, license of the items
│       ├── items/<lang>.jsonl  #   parallel items: same item_id in every language
│       └── assets/             #   images/audio/video — Git LFS (D3; .gitattributes at root)
│
├── runs/                       # ── DATA: append-only, never edited (CI-enforced) ──
│   └── <track>/<yyyy-mm>/
│       ├── <run-id>.meta.json  #   identity tuple, machine/provider, runtime, status, totals
│       └── <run-id>.items.jsonl#   one row per item: prediction, gold, correct, latency, cost
│
├── observations/               # ── DATA: scraped, dated, append-only (price, ZDR, context, quant) ──
│   └── <provider>/<yyyy-mm>.jsonl
│
├── build/                      # GITIGNORED: <track>.runs.json, <track>.items/, deployments.json, chart PNGs (board input)
│
├── packages/                   # ── TS libraries (imported, never run directly) ──
│   ├── schema/                 #   Zod 4 source of truth for catalog / suite / run / runner-protocol shapes
│   │   └── src/{catalog,suite,run,protocol}.ts  →  `bun run gen` writes catalog/schema/*.json
│   └── lake/                   #   load catalog+runs, validate, aggregate metrics, write JSON per track (plain TS)
│
├── bench/                      # ── TS project: ALL benchmark execution (hosted in-process, local over HTTP) ──
│   ├── package.json
│   └── src/
│       ├── cli.ts              #   run | canary | sync-metadata | render | doctor
│       ├── providers/          #   one adapter per API *contract*:
│       │   ├── systemone.ts    #     TypeSafe /v1/systemone (Jev, Clef via contract, Kev, Celeris, Semif…)
│       │   ├── openrouter.ts   #     chat + decisions catalog + endpoints metadata
│       │   ├── workers-ai.ts   #     Clef, Clairic
│       │   ├── fastino.ts  scaledown.ts  typellm.ts …
│       │   └── index.ts        #     registry: catalog deployment.adapter → module
│       ├── local/runner-client.ts   # HTTP client for runners/* (uses packages/schema protocol types)
│       ├── tracks/<track>/     #   request builder + scorer per track (decision, ner, typed-extract, rerank…)
│       ├── cost.ts  timing.ts  machine.ts
│       └── render/             #   generated README sections + chart images
│   └── test/
│
├── apps/
│   ├── board/                  # static decision board — Vite 8 + React + TanStack + ECharts, in-memory filters
│   │   └── src/{routes/{picks,explore,model.$id,crossover,providers,planner},charts,queries}
│   └── playground/             # try hosted (and live local) models — Bun.serve + React, keys server-side
│       └── src/{server.ts, ui/}
│
├── runners/                    # ── Python: one INDEPENDENT uv project per dependency island ──
│   ├── _protocol/              #   tiny shared lib (path dependency): FastAPI factory, /info, /healthz,
│   │                           #   /v1/<track> models (generated Pydantic), host fingerprint, Gradio mount
│   ├── gliner/                 #   GLiNER2/2.5, GLiFormer, GLiREL, Relex, GLiClass, Laya — transformers 4.57.x
│   ├── von/                    #   transformers 5.x, py3.13
│   ├── relik/                  #   py3.11, torch 2.3.1, Windows csv shim
│   ├── systemone-local/        #   Kev / Decider / OpenThai servers (today's ports 8009/8018/8029)
│   ├── small-decisions/        #   d1, decision2, julia, k2type, certo, mojev, rlcd… (split further if pins clash)
│   ├── nanodiff/               #   diffusion decision engine + its training scripts
│   └── <island>/               #   each one:
│       ├── pyproject.toml      #     own deps; [tool.uv.sources] runner-protocol = { path = "../_protocol", editable = true }; [tool.pyrefly]
│       ├── uv.lock  .python-version
│       ├── src/<island>/{server.py, models/*.py, ui.py}
│       ├── tests/
│       ├── Dockerfile          #     for cloud self-hosting
│       └── README.md           #     what it serves, hardware needs, how to start
│
├── docs/
│   ├── research/               # dated research notes (unchanged convention)
│   ├── plan/                   # these documents
│   ├── methodology.md          # how tracks are scored, what "deferred" means, cost model
│   └── adding-a-model.md       # the 3-step recipe (catalog entry → adapter/runner → bench run)
│
├── legacy/                     # TEMPORARY during migration: today's engines/, demos/, bench_*.py, app.py,
│                               # _trials/ — deleted when phase 5 of 05-migration lands
└── .github/workflows/
    ├── ci.yml                  # schema validate · append-only check · bun test · per-runner pytest (changed only) · render diff
    └── board.yml               # build lake → build board → deploy Pages
```

## Rules that keep this tree from regrowing into the current one

1. **Adding a hosted model = one YAML file.** It touches code only when it brings a new
   API *contract*, and then one adapter file.
2. **Adding a local model = one YAML file + one model module in an existing island.**
   A new island only when its pins conflict with every existing one.
3. **Nothing under `catalog/`, `suites/`, `runs/`, `observations/` is code**, and
   nothing under `bench/`, `apps/`, `packages/`, `runners/` is data.
4. **No hand-written number in README or docs** that also exists in the lake. Cite the
   generated section instead.
5. **Runners never import each other**; TS code imports only `packages/*`; `apps/*`
   may import `bench`'s provider registry (`@zsib/bench/providers`) and nothing else
   from it.
6. **One-off experiments** (today's `_trials/trial_N.py`) become a dated suite + a
   catalog entry + a run. The run *is* the trial record. Scratch code that never
   produced a run doesn't get committed.

## Where today's files go

| Today | Goes to |
|---|---|
| `bench_spectrum.py`, `bench_multilingual.py`, `bench.py`, `bench_graded.py` | question pools → `suites/`; drivers → replaced by `bench/` |
| registries (`OPENROUTER_SYSTEMONE`, `CLEF_MODELS`, `ALL_SYSTEMS`, `CULLED`…) | `catalog/models/*.yaml` (`status: culled` is a field) |
| `engines/{openrouter,jev,clef,fastino,typellm,agentjev,ollaya}_client.py` | `bench/src/providers/*.ts` (rewritten) |
| `engines/{d1,decision2,julia,k2type,relik,von,glinerx,intern_decision,d3}_client.py`, `engines/*_engine/` | `runners/<island>/src/…/models/` (island grouping confirmed by each one's pins) |
| `demos/*` | each runner's `ui.py` (Gradio) + `apps/playground` for hosted |
| `app.py` (3,730 lines) | split: local tabs → runner UIs; hosted tabs → playground; results tabs → board |
| `results/*.json` | converted once into `runs/` (marked `source: legacy-v0`) |
| `README.md` tables, `docs/charts/` | generated by `bench render` |
| `TRIALS.md`, `_trials/SUMMARY.md` | `CHANGELOG.md` + run records |
| `requirements*.txt`, `overrides.txt`, `.venv*` | each island's `pyproject.toml` + `uv.lock` |
| `tests/test_*wiring*`, `test_derived_cost` (they police duplication) | deleted; replaced by schema validation + the render diff |
