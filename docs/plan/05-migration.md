# 05 — Migration plan

*2026-10-24. From today's flat repo to [03](03-folder-structure.md), one PR per phase
(direct pushes to `main` are blocked anyway). The current bench stays usable until
phase 4's acceptance passes.*

## Phase 0 — freeze v0 (½ hour)

- Tag `v0-final` on `main`. Every number in today's README stays citable at that tag.- `git mv` engines/, demos/, bench_*.py, app.py, make_chart_images.py, _trials-tracked
  files, results/, tests/ → `legacy/`. Fix the relative paths so `legacy/` still runs.

## Phase 1 — spine: schema + catalog + machines (1–2 days)

- Root `package.json` (Bun workspaces, `catalog:` versions), `bunfig.toml`,
  `biome.json`, root `scripts` (no `justfile`). **`typescript` in the catalog is `^7`** (not 5 or 6), with a
  `typecheck` script running `tsc --noEmit` in each workspace. Every other catalog
  entry is floored at its current major (see 02, "Version floors").
- `packages/schema` (Zod 4) → `catalog/schema/*.json`.
- **No DuckDB** (see 02, "Lake"). If `@duckdb/node-api` is already in the root catalog,
  remove it.
- Extract `catalog/models/*.yaml` from the Python registries + README tables (a
  one-off script under `legacy/`). Write `machines/seoul.yaml`.
- **Acceptance:** every system in today's README has a catalog file, a generated
  count (81 / 78 / 38) matches the README at `v0-final`, and `bun run typecheck` passes
  on TypeScript 7.

## Phase 2 — suites + legacy runs into the lake (1 day)

- Move the mixed pool, multilingual pool, NER questions and the gate set into
  `suites/` as parallel-item JSONL. Add a `canary` suite per track.
- Convert `legacy/results/*.json` into `runs/` with `source: legacy-v0` (D4).
- Add `.gitattributes` LFS patterns before the first media asset lands (D3).
- `packages/lake` builds the per-track JSON; `bench render` generates the README tables.
- **Acceptance:** the generated tables match the `v0-final` README cell for cell.
  This is the regression test for the whole data model.

## Phase 3 — runners (2–4 days, one island per PR)

- Before the first island: `uv self update` on every machine that builds runners (uv ≥ 0.13).
- `runners/_protocol` first, then islands in order of how much they're used:
  `gliner` → `systemone-local` → `small-decisions` → `von` → `relik` → `nanodiff`.
- Each island: own `pyproject.toml` + `uv.lock` from today's pins, `/info` + `/v1/<track>`
  + Gradio mount, Dockerfile, its existing tests moved in, Pyrefly passing.
- **Acceptance per island:** the runner answers the same items as its `legacy/` engine
  with identical predictions on `seoul`.

## Phase 4 — bench (TS) (3–5 days)

- Adapters in order of how many systems they unlock: `openrouter` → `systemone` →
  `workers-ai` → `fastino` → the rest. Scorers for `decision` and `ner` first.
- `bench run`, `bench canary`, `bench sync-metadata`, the `status: deferred` path.
- **Acceptance:** re-run the decision mixed pool for the top ~10 hosted + top ~5
  local deployments. Scores land within noise of `v0` (hosted models are allowed to
  have drifted; that drift is the first real nerf-detection data point, so record it).

## Phase 5 — apps + delete legacy (4–6 days)

- `apps/board` on **Vite 8**. First check that TanStack Router's route generator works
  with TypeScript 7 (nothing official says either way); if it doesn't, pin that one
  workspace to `@typescript/typescript6` until TS 7.1. Picks + explorer + model page first; crossover, provider picker and
  planner after.
- `apps/playground`.
- Delete `legacy/`. README cut to ~200 lines + generated sections.
- CI: schema validation, append-only check, render diff, per-island tests, board
  deploy.

## After the migration (new capability, not migration)

- New tracks, each a suite + a scorer: `typed-extract` (Clairic, ScaleDown Extract,
  Schemer), `schema-extract` (NuExtract3), `relation`, and image/audio variants of
  `decision`.
- Canary cadence (a scheduled CI job or local cron).
- Every Eval Ever exporter.

## Rough total

About 2–3 weeks of focused work, front-loaded on phases 1–2, which deliver most of the
value: no more hand-edited counts, and an append-only history.
