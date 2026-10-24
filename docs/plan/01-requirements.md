# 01 — Requirements

*2026-10-24. Distilled from the ZCode/GLM chat "Research new decision and extraction AI
models" (this repo, 13:31–14:13) plus the repo as it stands at `20e4825`. Numbers are
stable — later docs cite them as R1…R12.*

## The problem

The repo outgrew its shape. README 1,467 lines, `app.py` 3,730, two ~1,300-line bench
drivers, 17 engine clients, 30 demos, 29 tests, 4 venvs, 81 systems / 38 families. One
new model touches ~4 code files, ~4 README tables, TRIALS, charts and a set of counts.
The README is a hand-edited database, and the tests parse markdown to catch drift
in it. It is too big to make decisions from.

## What you asked for

| # | Requirement | Source |
|---|---|---|
| R1 | **Local models in one place**: a folder inside this repo (not a separate repo), split into several projects by dependency set or implementation language | you, 13:31 + 14:02 correction |
| R2 | **Run local models even when a hosted version exists.** When one can't run for resource reasons, **record that** so it can be retried later | 13:31 |
| R3 | **Machine registry**: the specs of every machine used, listed in one place, with each benchmark run naming its machine. A GPU box runs some models differently *and costs differently* | 13:31 |
| R4 | **Local UI** so someone can self-host or cloud-host the local models and try them. Gradio is fine | 13:31 |
| R5 | **Hosted runs = one project, TypeScript + Bun**, latest stack, no Python, since these are just API calls | 13:31 |
| R6 | **A separate TS app to try hosted models** | 13:31 |
| R7 | **Capability dimension**: task type (NER/IE, classifier, decision, reranker used as a classifier, extraction…) and modality (text, image, audio, video). Benchmark each separately, and **store each benchmark type in its own collection** (CSV is fine) | 13:31 |
| R8 | **Language dimension**: a fixed question set in selected languages, measured **per task type** | 13:31 |
| R9 | **Decision app** (not Python): results with types and filters for choosing self-host vs hosted, which model, where to host, whether it needs a GPU, and which provider. Use good JS charting | 13:31 |
| R10 | **Run identity** = model + version + provider or machine + anything else that matters. **Rebenching keeps history**, so a nerf or an improvement shows up | 13:31 |
| R11 | **Track all metadata** you can find: ZDR, price, and so on | 13:31 |
| R12 | **Root = this repo, set up as a proper monorepo** | 14:02, 14:11 |

Standing house rules this plan keeps: measured cost (not derived), the trial gate, dated
`docs/research/` notes, semantic commits through PRs (direct pushes to `main` are blocked).

## What I'm keeping from GLM's brainstorm

- One source of truth for system metadata, with **every human-facing table generated**
  and a CI check `render && git diff --exit-code`. This removes the whole "76 → 78 in
  8 places" class of bug.
- Runs as **append-only artifacts**; aggregates are build outputs.
- A **canary suite** for cheap nerf detection, and a **local twin** as a control group:
  if the hosted score moves while the local copy of the same weights doesn't, the change
  is on the serving side.
- Local cost is **not $0**: with machine $/h recorded it becomes comparable to API prices.
- A short README, with history moved to a changelog and the board tiered so headline
  picks are separate from the full table.

## Where I disagree with GLM (verified)

1. **"uv workspace: one lockfile, separate venv per member" is false.** uv workspaces
   resolve **one** dependency set with **one** `requires-python` for the whole
   workspace. uv's docs say not to use them when members conflict or need separate
   environments ([uv workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/)).
   This repo is exactly that case:
   - the main env pins `transformers==4.57.6`
   - von needs `transformers==5.17.0`
   - relik needs Python 3.11 with `torch==2.3.1`

   **Fix:** each local runner is an *independent* uv project with its own `uv.lock` and
   `.venv`. Shared Python code is a path dependency (`tool.uv.sources`), which is what
   uv recommends for this case.
2. **Two bench drivers (Python + TS) running the same suites is unnecessary.** Make
   local runners *model servers only*, behind one HTTP protocol, and have the single
   TS bench drive both local and hosted (see 02). That gives one scorer, one run
   format and one place for metrics. Python keeps what it is actually needed for:
   loading the weights.
3. **uPlot is not needed.** A model's rebench history is tens of points, not 100k.
   ECharts covers everything. Fewer libraries.
4. **Generic "CSV for everything" doesn't fit model metadata.** It is nested (versions ×
   providers × prices over time). Tabular CSV/JSONL stays for **run rows**; the
   hand-curated catalog is YAML, one file per model.
