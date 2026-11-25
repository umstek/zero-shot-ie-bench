# AGENTS.md

Guide for coding agents (and humans) working on **zsib** (`zero-shot-ie-bench`).
Read [docs/methodology.md](docs/methodology.md) for scoring, and
[docs/plan/03-folder-structure.md](docs/plan/03-folder-structure.md) for the full
layout rules. This file is the checklist of *what to touch* for common tasks.

## Hard rules

1. **`runs/` is append-only.** Never edit or delete a run; rebench to add a
   new one (CI: `scripts/check-append-only.ts`).
2. **No hand-written numbers that exist in the lake.** README tables live
   between `<!-- gen:… -->` markers and come from `bun run render`. The board
   computes counts from `build/*.json`. Never type a model/run/family count
   into README, docs, or JSX.
3. **Data vs code.** `catalog/`, `suites/`, `runs/` are data only;
   `bench/`, `apps/`, `packages/`, `runners/` are code only.
4. **Suites are versioned by content hash.** Editing a suite item changes its
   hash; add a new suite version instead of rewriting a live one.
5. **Runners never import each other.** TS imports only `packages/*`;
   `apps/*` may import `@zsib/bench/providers` and nothing else from `bench`.
6. Run JS/TS tests with `NODE_ENV=test` (the dev shell may export
   `NODE_ENV=production`).

## Adding a model

Full recipe: [docs/adding-a-model.md](docs/adding-a-model.md). Checklist:

| Step | Hosted model | Local model |
|---|---|---|
| Catalog | `catalog/models/<org>/<model>.yaml`: `claims.tracks`, `claims.modalities`, `claims.capabilities`, `versions`, `deployments` (`kind: hosted`, `provider`, `endpoint`, `adapter`, `zdr_safe`, `display`) | Same YAML with `kind: local`, `runner: <island>`, `runtime`, `requires` (RAM/VRAM) |
| Provider | New provider? Add `catalog/providers/<provider>.yaml` | — |
| Code | Only for a **new API contract**: one adapter in `bench/src/providers/<provider>.ts` + a line in `bench/src/providers/index.ts`. The adapter's env key (e.g. `OPENROUTER_API_KEY`) is read by the bench | One model module in an existing island: `runners/<island>/src/<pkg>/models/<engine>.py`, registered in that package's `models/__init__.py` (`MODELS` + predict dispatch). New island only if its pins conflict with every existing one |
| Runner tests | — | `cd runners/<island> && uv run pytest` (weight tests skip offline) |
| Validate | `bun run validate` | `bun run validate` |
| Trial gate | `bun run --cwd bench dev run --deployment <id> --suite decision/gate-12` | Start runner (`uv run uvicorn <pkg>.server:app --port …`), set `ZSIB_RUNNER_<ISLAND>=http://127.0.0.1:<port>`, then same command |
| Bench | `--suite decision/mixed-pool`, then `decision/multilingual-pool`; NER models: `--suite ner/graded-pool` | Same. No runner URL → a `deferred` run is written (that's correct, not a failure) |
| Regenerate | `bun run render` (lake + README tables), commit `runs/` + README together | Same |

Then check the **hand-maintained prose** that names model families (these are
not generated, so update them if the new model adds a family or a new group):

- `apps/board/src/routes/overview.tsx`: `SYSTEM_TYPES[].families` (and the
  descriptions if the model introduces a new kind of system).
- `README.md`: the **What's covered** table.
- `CHANGELOG.md`: an entry under the next version.

Counts (models, deployments, per-group and per-modality numbers) are computed
from the catalog and need no edits.

## Updating an existing model

- **New version / API slug:** add an entry under `versions` and point the
  deployment at it. Never rewrite the old version: past runs reference it.
- **New way to run it** (another provider or local runtime): add a deployment
  under `deployments`; no new model file.
- **Retired:** set `status` accordingly. Keep the file and its runs.
- **Rebench / drift check:** `bun run --cwd bench dev canary --deployment <id>`,
  or a full `run`. It adds history; never delete the old run.

## Adding a task type (track) or modality

- Track definition and metrics: `catalog/tracks.yaml` (+ schema in
  `packages/schema/src`, then `bun run gen` to regenerate `catalog/schema/*.json`
  and the runner Pydantic models via `runners/_protocol/tools/`).
- Scorer: `bench/src/tracks/<track>/scorer.ts`, wired in `bench/src/tracks/index.ts`.
- Suite: `suites/<track>/<suite>/` (suite.yaml + items).
- Lake / board: `packages/lake/src`, `apps/board/src/types.ts` (`Track`),
  and a route in `apps/board/src/routes/` + `router.tsx` + `scripts/prerender.ts`.
- Docs: the track table in `docs/methodology.md`, the **What's covered**
  section in README, and `SYSTEM_TYPES` / `MODALITIES` in `overview.tsx`.
- New modality: suites are text-only today. Image/audio/video claims are shown
  as "claimed, not yet benchmarked" until a suite for that modality exists.
  Update that wording in `overview.tsx` and README when one lands.

## Where things live

| Path | What |
|---|---|
| `catalog/` | models, providers, machines, languages, tracks (+ generated JSON schemas) |
| `suites/` | versioned item pools (`decision/{gate-12,mixed-pool,multilingual-pool,canary}`, `ner/{graded-pool,canary}`) |
| `runs/` | append-only run records |
| `bench/` | Bun CLI: `run`, `canary`, `doctor`; provider adapters; scorers; README renderer |
| `runners/` | Python/uv dependency islands speaking one HTTP protocol (`runners/_protocol`) |
| `packages/schema`, `packages/lake` | Zod schemas (source of truth) and lake builder (`build/*.json`) |
| `apps/board` | static board (Overview, Picks, Explorer, Charts, NER) |
| `apps/cloud` | `/try/` BYOK tryer; lists hosted deployments from the catalog automatically |
| `apps/local` | local playground |

## Gates before a PR

```sh
bun install
NODE_ENV=test bun run check      # typecheck + tests + biome
bun run validate                 # catalog shape
bun run render && git diff --exit-code README.md   # generated tables current
bun run build:site               # board + tryer + prerender
```

CI also runs schema-diff, the phase acceptance scripts, the append-only check,
and each runner's offline `uv` tests. `main` deploys to staging; a GitHub
release deploys production (see [docs/hosting.md](docs/hosting.md)).
