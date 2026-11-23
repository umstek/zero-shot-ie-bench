# Board roadmap — planned views

The three views below are planned but **not built**. Their stub pages were
removed from the site (empty pages don't belong on a published board); add the
route when the view is implemented. Intent copied from the former stubs and
[docs/plan/02-architecture.md](plan/02-architecture.md) §Board.

## Self-host vs API (was `/crossover`)

Planned (per docs/plan/02-architecture.md §Board): cost per 1k items against
volume, and the break-even point where a local deployment on a machine in
`catalog/machines/` becomes cheaper than the hosted providers — using measured
run cost and the amortised machine rates.

## Provider picker (was `/providers`)

Planned (per docs/plan/02-architecture.md §Board): the same model across its
providers — price, ZDR status, latency and stability side by side. The data is
already in `deployments.json` (one row per deployment); this page groups it by
`model_id`.

## GPU planner (was `/planner`)

Planned (per docs/plan/02-architecture.md §Board + R2 "feasibility is data"):
pick a candidate machine spec and see which deferred runs it would unlock — a
query over `status = deferred` runs × `deployment.requires` × the machine's
VRAM/RAM/accel.
