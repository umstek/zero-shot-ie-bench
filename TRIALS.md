# Candidate trials

Every new system (hosted or community) is screened on a fixed
12-question mini-pool before benchmark integration: the first 12 of
`bench_spectrum.py`'s mixed classification pool, one warm-up call,
per-question timing. 12/12 graduates to the benchmark. Everyone else is
here — the good ones are in the README, the rest is this file, data
kept. Raw artifacts in `_trials/` (gitignored).

## Culled from the benchmark

Benched as first-sweep exceptions before the 12/12 bar was tightened,
later pulled back out. Their engines stay vendored in the repo
(`engines/rlcd_engine/`, `engines/nanodiff_engine/`), runnable via
`demos/lfm_rlcd_demo.py` and `demos/nanodiff_demo.py`.

| System | What | Params | License | Trial |
|---|---|---|---|---|
| [LFM2.5-RLCD](https://huggingface.co/notnotsamuel/LFM2.5-350M-RLCD) | RLCD-trained LFM2.5, constrained decoding over a JSON schema | 350M | engine MIT; weights LFM Open License v1.0 | 10/12 |
| [nanodiff 350M](https://huggingface.co/pngwn/nanodiff-350m-typed-decisions-lam1) | bidirectional diffusion LM, option-letter softmax | 350M | MIT | 6/12 |
| [nanodiff 350M v2](https://huggingface.co/pngwn/nanodiff-350m-typed-decisions-v2) | v2 retrain of the same recipe | 350M | MIT | — (v1's recipe) |

Mixed-pool spectrum (48 cls questions, 2026-09-25 / 2026-10-05 runs):

| System | Cls | s/q | NER |
|---|---|---|---|
| LFM2.5-RLCD 350M | 56.2% | 0.469 | n/a (schema subset can't express spans) |
| nanodiff 350M | 25.0% | 11.56 | n/a (single-letter choice interface) |
| nanodiff 350M v2 | 18.8% | 10.778 | n/a (same as v1) |

Multilingual suite (9 languages; popular / medium / rare / overall):

| System | popular | medium | rare | overall | s/q |
|---|---|---|---|---|---|
| LFM2.5-RLCD 350M | 78% | 67% | 11% | 52% | 0.507 |
| nanodiff 350M | 28% | 39% | 39% | 35% | 12.223 |
| nanodiff 350M v2 | 39% | 28% | 33% | 33% | 11.689 |

Per-language (LFM2.5-RLCD 350M): Chinese 50% · French 100% · Icelandic
17% · Sinhala 0% · Spanish 83% · Turkish 50% · Ukrainian 67% · Vietnamese
83% · Welsh 17%. (nanodiff 350M): 33% · 17% · 50% · 33% · 33% · 50% ·
33% · 33% · 33%. (nanodiff 350M v2): 33% · 50% · 33% · 33% · 33% · 17%
· 33% · 33% · 33%.

Why out: all three failed the 12/12 gate — nanodiff v1/v2 are chance-level
(near-uniform option probabilities) and the two slowest systems ever
benched here; LFM2.5-RLCD was the 10/12 first-sweep exception —
mid-table at 56.2% / 52% with no NER path. Full recorded rows
(per-question predictions included) in `_trials/culled/`.

## Never benched (trial failures)

| System | Trial result |
|---|---|
| `Quazim0t0/Byrne-Jev-79M` | 9/12 easy, 7/16 hard — dominated by the benchmarked systems |
| Dohnuts 0.8B (iACE, from-scratch) | unrunnable — its released runtime hardcodes CUDA (flash-linear-attention[rocm], `.to("cuda")`); no CPU path |
| `tasksource/modernbert-tasksource-jev` | unrunnable — its `modernjev` package is unpublished, source links 404, card says "preview, not ready to use" |
| `shreyanbr/system-one-gold` | unrunnable — requires a `systemone` engine package and calibration file that are not published |
| `idlabs/jev-typed-decisions-causal-0.6b` | gated on HF (401) |
| `anthonym21/qwen3-0.6b-rlcd-decision` (eve-rlcd's RLCD recipe on Qwen3-0.6B, 2026-10-10) | 11/12 easy at 2.2 s/q — beats its LFM2.5-RLCD sibling's 10/12 trial score above, still below the 12/12 bar |
| `shgao/rsi-jev-v6.1-vl-4b` (RSI-Jev, 2026-10-10; `rsi-jev serve` speaks the Jev wire API so the repo's JevClient runs it) | 10/12 easy at 3.3 s/q — Decision Index 0.3 public 50.98 on its card, but the mini-pool ties LFM2.5-RLCD's trial at 5× the latency |
| `Manavarya09/verdict` "Verdict-MM" (verdictml, 118M multilingual e5, 2026-10-10) | 9/12 easy at 0.090 s/q — the fastest trial by far, weakest of the sweep; conformal abstention noted on the card |
| `nandakishorm/vega-08b-public-intents` (frozen Qwen3.5-0.8B feeding a 57 MB particle-settling "physics engine", 2026-10-10; no relation to Decision 2.0's Vega-27B) | 9/12 easy at 0.45 s/q CPU fp32 — its conformal abstain flagged every miss (17/17 on the non-abstained answers) but accuracy stays below the graduates; the shipped adapters never gated on and the repo carries no license |

The Sept-25 sweep also ran a 16-hard-tier pool (+ NER where supported);
later trials use the 12-question pool only.
