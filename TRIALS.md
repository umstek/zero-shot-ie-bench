# Candidate trials

Every new system passes through here before it touches the benchmark
spectra. The trial is a fixed 12-question mini-pool drawn from this
repo's graded questions — 8 sentiment easy + 4 sentiment medium, the
first 12 of `bench_spectrum.py`'s mixed classification pool — run with
one untimed warm-up call and per-question timing, through the same
engine clients and question phrasing the benchmark itself uses (hosted
models run their pool over their API; for cents, not a full-suite bill).
**12/12 graduates to the full benchmark** (both suites, systems-table
row, charts, tests); everything else — failed, unrunnable, or gated —
gets a row below. The data is kept; the main README's tables stay
benched-systems-only.

Since 2026-10-10 the gate is universal: hosted and provider models go
through the same pool as community repos, before any benchmark
integration. Systems already benched either predate the gate or entered
directly during the 2026-10-10 hosted sweep — the gate applies to
everything new from here on.

History: the first catalog sweep (2026-09-25, the community "All about
Jev" catalog — 1,619 entries → 9 plausible candidates, four of them
unrunnable or gated) ran two pools per candidate — 12 easy-tier and
16 hard-tier classification, plus the first 3 NER texts where the
interface supported spans. The recent trials use the 12-question pool
only. A second sweep (2026-10-10, the
[decision-model sweep](docs/research/2026-10-10-decision-model-sweep.md)
over the catalog's new arrivals) trialed three more; none beat the 12/12
graduates. The strict 12/12 bar dates from that second sweep — the first
benched two sub-bar candidates anyway, as noted below.

## Graduates (now benchmark rows)

| System | Trial | Note |
|---|---|---|
| `fastino/GLiNER2.5-Decide` | 12/12 cls, 3/3 NER, 0.346 s/q | the only candidate NER was run on (span interface) |
| `alibiserikbay/JevK5-Lite` | 12/12, 0.328 s/q | |
| `notnotsamuel/LFM2.5-350M-RLCD` | 10/12, 0.654 s/q | first-sweep exception — benched despite the score, before the 12/12 bar was tightened |
| `pngwn/nanodiff-350m-typed-decisions` | 6/12, 10.5 s/q | first-sweep exception — benched despite the score, before the 12/12 bar was tightened |

## Not promoted

| System | Trial result |
|---|---|
| `Quazim0t0/Byrne-Jev-79M` | 9/12 easy, 7/16 hard — dominated by the graduates above |
| Dohnuts 0.8B (iACE, from-scratch) | unrunnable — its released runtime hardcodes CUDA (flash-linear-attention[rocm], `.to("cuda")`); no CPU path |
| `tasksource/modernbert-tasksource-jev` | unrunnable — its `modernjev` package is unpublished, source links 404, card says "preview, not ready to use" |
| `shreyanbr/system-one-gold` | unrunnable — requires a `systemone` engine package and calibration file that are not published |
| `idlabs/jev-typed-decisions-causal-0.6b` | gated on HF (401) |
| `anthonym21/qwen3-0.6b-rlcd-decision` (eve-rlcd's RLCD recipe on Qwen3-0.6B, 2026-10-10) | 11/12 easy at 2.2 s/q — beats its integrated LFM2.5-RLCD sibling's 10/12 trial score, below the 12/12 graduates |
| `shgao/rsi-jev-v6.1-vl-4b` (RSI-Jev, 2026-10-10; `rsi-jev serve` speaks the Jev wire API so the repo's JevClient runs it) | 10/12 easy at 3.3 s/q — Decision Index 0.3 public 50.98 on its card, but the mini-pool ties LFM2.5-RLCD's trial at 5× the latency |
| `Manavarya09/verdict` "Verdict-MM" (verdictml, 118M multilingual e5, 2026-10-10) | 9/12 easy at 0.090 s/q — the fastest trial by far, weakest of the sweep; conformal abstention noted on the card |
| `nandakishorm/vega-08b-public-intents` (frozen Qwen3.5-0.8B feeding a 57 MB particle-settling "physics engine", 2026-10-10; no relation to Decision 2.0's Vega-27B) | 9/12 easy at 0.45 s/q CPU fp32 — its conformal abstain flagged every miss (17/17 on the non-abstained answers) but accuracy stays below the graduates; the shipped adapters never gated on and the repo carries no license |

## Raw artifacts

Trial scripts (`_trials/trial_*.py`), per-question verdict JSONs,
fetched-card notes (`_trials/card-*.md`) and the running summary live in
`_trials/` — gitignored, local only. This file is the committed record;
a row is added here whenever a trial runs.
