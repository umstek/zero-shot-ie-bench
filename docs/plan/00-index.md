# Restructure plan — index

*2026-10-24. Planning only: no code changed, nothing committed. Written by Claude
(Opus 5.5) from the GLM chat "Research new decision and extraction AI models",
independent research, and the repo at `20e4825`.*

| Doc | What it answers |
|---|---|
| [01-requirements.md](01-requirements.md) | What you asked for (R1–R12), what's kept from GLM, and the 4 places GLM was wrong |
| [02-architecture.md](02-architecture.md) | Dimensions (track × modality × language × deployment), run identity and history, runners, bench, board, stack table |
| [03-folder-structure.md](03-folder-structure.md) | **The tree**, rules that stop it regrowing, where every current file goes |
| [04-data-model.md](04-data-model.md) | YAML/JSONL shapes for catalog, machines, suites, runs, observations; Parquet collections |
| [05-migration.md](05-migration.md) | 5 phases, one PR each, with an acceptance test per phase |

## Decisions (settled 2026-10-24; numbers stable)

1. **Runners are model servers; the TS bench drives them. Decided.** One scorer, one
   run format. Benching a GPU box means installing Bun there (a single binary) next to
   uv. Rejected: a Python bench driver inside each runner.
2. **Board = custom Vite + React + TanStack + DuckDB-WASM + ECharts. Decided.**
   Rejected: Evidence.dev (faster v1, but it can't carry the picks / break-even /
   planner views).
3. **Suite media in Git LFS. Decided.** Keeps "one clone = everything runnable".
   Revisit and move to a Hugging Face dataset pointer if media passes ~1 GB or video
   suites arrive.
4. **Import `v0` results as runs (`source: legacy-v0`). Decided.** History starts on
   day one, and phase 2's acceptance test (generated tables = `v0-final` README)
   depends on it.
5. **Owned hardware is costed at an amortised $/h set once per machine
   (hardware ÷ expected life + power). Decided.** Rented machines use list price.
   Local cost is never shown as $0.
6. **`screenapp-ie-replacement-report.md` moved to `D:\src\screenapp`. Done
   2026-10-24.** It is a ScreenApp analysis, not bench content.

## Sources used for the stack choices

- [uv workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/): single
  `requires-python`, not for conflicting members
- [Bun catalogs](https://bun.com/docs/pm/catalogs), [Bun isolated installs](https://bun.com/docs/pm/isolated-installs),
  [Bun 1.2.2 DuckDB N-API fix](https://bun.com/blog/bun-v1.2.2)
- [Every Eval Ever paper](https://arxiv.org/pdf/2606.14516), [repo](https://github.com/evaleval/every_eval_ever)
- [moon v2 RC (Python toolchain unstable)](https://moonrepo.dev/blog/moon-v2-rc)
- [Gradio 6 migration guide](https://www.gradio.app/guides/gradio-6-migration-guide)
- [TanStack Start SPA mode](https://tanstack.com/start/latest/docs/framework/react/guide/spa-mode),
  [TanStack Charts comparison (ECharts / Plot bundle sizes)](https://tanstack.com/charts/latest/docs/comparison),
  [ApexCharts 100k-point benchmark](https://apexcharts.com/blog/javascript-chart-library-benchmark/)
- [Evidence (MotherDuck glossary)](https://motherduck.com/glossary/evidence/)
