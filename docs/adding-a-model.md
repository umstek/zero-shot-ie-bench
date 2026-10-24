# Adding a model

The three-step recipe (docs/plan/03 rule 1–2). Code changes only when a
new API *contract* or a new dependency island arrives.

## 1. Catalog entry

One YAML file: `catalog/models/<org>/<model>.yaml` — id, family, org,
license, claims, versions, deployments (hosted: provider + endpoint +
adapter; local: runner island + runtime + requires).
`bun run validate` gates the shape. Adding a hosted model touches code
only if its provider speaks a new contract — then one adapter file in
`bench/src/providers/` and a registry line. A local model is one module
in an existing island; a new island only when its pins conflict with
every existing one.

## 2. Trial gate

Screen on `decision/gate-12` (the first-12 fixed mini-pool, one warm-up
call): `bun run --cwd bench dev canary --deployment <id>` — wait, that
runs the canary; the gate is `run --suite decision/gate-12`. 12/12 (or a
user-directed override, recorded) graduates to the benchmark. The run IS
the trial record — nothing else is committed.

## 3. Bench

`bun run --cwd bench dev run --deployment <id> --suite decision/mixed-pool`
(then the multilingual pool). The run lands in `runs/`, the lake picks
it up, `bun run render` regenerates the README tables, and CI's
`render && git diff --exit-code` keeps every human-facing number
generated.
