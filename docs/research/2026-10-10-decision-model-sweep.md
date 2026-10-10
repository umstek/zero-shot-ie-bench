# New decision-model sweep — 2026-10-10

Follow-up to the 2026-10-05 sweep, triggered by the Oct 6–9 arrivals: Cloudflare's
clef-omni, Microsoft's Decision-1, Nace.AI's Drex v1.5, OpenAI's GPT-6 Luna
Decisions, and LiquidAI's open d1 weights. Sources: OpenRouter's decisions-modality
catalog + per-model endpoints API (authenticated), Cloudflare Workers AI docs,
models.dev database, the nace-ai and vllm-sr HF orgs, LiquidAI's HF org, the
Ollaya catalog, and the community "awesome-jev" list (kraayenjon/awesome-jev,
updated 2026-09-27 — the "All about Jev" catalog; this repo's Sept-25 trial sweep
predates its latest 16 PRs). Research + routing probes only; weights untouched.

## Findings

### Hosted — rides an existing engine (immediate)

1. **Clef Omni** — `@cf/cloudflare/clef-omni` on Workers AI (also mirrored as
   `cloudflare/clef-omni` on OpenRouter at the same $0.15/M) · released 2026-10-09 ·
   fine-tune of Qwen3-Omni-30B-A3B (30B total / 3B active MoE) · 64k ctx ·
   $0.15/M input, output free · weights open on HF (Cloudflare/clef-omni).
   Same `{model, state, questions}` contract as the benched Clef pair; extends
   the Clef images[] extension with audio[] (max 4 × 8 MiB / 300 s) and videos[]
   (max 2, 2 fps sampling) state inputs. Text path identical for the benches;
   rides `engines/clef_client.py` (two dict lines). Local twin: 30B MoE ≈ 60 GB
   bf16 — out of reach on this machine (bigger than even Clef-Flash's 19 GB).
2. **Microsoft-Decision-1** — `microsoft/microsoft-decision-1` via OpenRouter
   /systemone (Azure provider, the only endpoint) · added 2026-10-09 ·
   Qwen3.5-9B post-trained for single-pass scoring · 32,768 ctx · $0.042/M input,
   output free · closed weights, "updated continually while the API shape stays
   the same" (announcement: commandline.microsoft.com/microsoft-decision-1-model-foundry).
   Routing probe passes under this account's enforced ZDR (Azure endpoint
   satisfies the data policy) → no †. p50 330 ms provider-side.
3. **Drex v1.5** — `nace-ai/drex-v1.5` via OpenRouter /systemone (DeepInfra
   provider) · card dated 2026-09-28, endpoint snapshot 2026-10-09 · 9B
   (MiMo-V2.6-Distill-Qwen-9B backbone) · 131,072 ctx · $0.04/M input, output
   free · **open weights** (nace-ai/drex-v1.5, Nace.AI Open RAIL-M) + **open
   runtime** (github.com/nace-ai/drex-decision-models, Apache-2.0, adapted from
   Kev; serve.py speaks POST /v1/systemone). Card claims rank 1 on Decision
   Index 0.2.1 (58.28). Routing probe passes ZDR. Local: 9B fp32 ≈ 36 GB —
   over this machine's 31.4 GB; the GGUF path needs Nace's llama.cpp/Ollaya
   forks (CAPABILITY decision is not in stock llama.cpp) — note as local-twin
   caveat, bench hosted. Sibling: Drex DLM (diffusion decision model, 8B,
   CC BY-NC 4.0, GGUFs) — same fork constraint.
4. **GPT-6 Luna Decisions** — `openai/gpt-6-luna-decisions` · added 2026-10-06 ·
   GPT-6 Luna behind OpenAI's Decisions API · 1.05M ctx · $0.10/M input, output
   free · up to 200 questions/request · vision-capable state. **Re-verified
   2026-10-10: still refused under this account's enforced ZDR** ("0 endpoints
   out of 1 requested are available matching your guardrail restrictions and
   data policy"). Not benchmarkable without relaxing account privacy or a
   direct OpenAI key; keep the documented exclusion, refresh its date.

### Local — official open weights

5. **LiquidAI d1-3B** — `LiquidAI/d1-3B` (+ GGUF, + w8a8) · released 2026-10-05 ·
   3.12B on LFM2.5-VL-3B · 32,768 ctx · LFM Open License v1.0 (lfm1.0, same
   family as the benched LFM2.5-RLCD weights) · 17 languages (ar zh en fr de hi
   id it ja ko pl pt ru es th vi) · multimodal (text + images in one state) ·
   card claims best-under-10B on Decision Index 0.2.1 (48.57). Loads
   in-process with `trust_remote_code=True` (auto_map AutoModel →
   modeling_d1.D1Model; backbone model_type lfm2_vl) under transformers ≥5.14 —
   .venv-von's 5.17 qualifies; fp32 CPU ≈ 12.5 GB fits the 31.4 GB machine.
   API: `model.system_one(state, questions)` / `system_one_batch` — noul/choice/
   score schema, one forward, zero output tokens. The hosted `liquid/d1` row
   (benched 2026-10-07) is the same line's API twin — a rare local-vs-hosted
   same-family pair.
6. **LiquidAI d1-omni-600M** — `LiquidAI/d1-omni-600M` (+ GGUF) · 2026-10-05 ·
   587M (381M trunk on LFM2.5-Encoder-350M + 94M SigLIP2 vision + 112M
   FastConformer audio) · 16,384 ctx · lfm1.0 · 16 languages · text + images +
   ≤30 s speech in one state, zero output tokens. Encoder-trunk cousin: the
   omni sibling at 1/5 the size.

### Local — Ollaya catalog arrivals (16 → 20 since the 10-05 sweep)

7. **decima** — A. M. Madani · 122M/321M mmBERT-base/multilingual-e5-small
   encoders + late-interaction option scorer (order-invariant) + ordinal score
   head · multilingual, "fastest on a CPU" per the catalog · updated 2026-10-05.
   Rides `engines/ollaya_client.py` (registry lines + daemon pull).
8. **snap** — logitlab's snap1-2b · MiniCPM5-2B fine-tune, Q8_0 GGUF on
   llama.cpp, CPU-capable · 0.648 typed-decisions on Ollaya's suite.
9. **arbiter** — Codekins' Arbiter (Zyot Lab) · Gemma 3 4B IT + LoRA, fixed
   24-slot head · noul + choice ≤16 options + score of exactly 6 levels ·
   multilingual · 4.3B Q8.
10. **credence** — Txoka's Credence v1 · MiCA refinements of Winnow-E4B, two
    Q8_0 checkpoints trading accuracy vs calibration (+ vision tag) · 7.5B —
    neither beats original Winnow on every metric; lowest priority of the four.

### Local — "awesome-jev" catalog additions since the Sept-25 trials

11. **eve-rlcd 0.6B** — github.com/anthony-maio/eve-rlcd · RL-trained (reward =
    outcome − stated probability; RLVR ablation) · weights
    anthonym21/qwen3-0.6b-rlcd-decision · choice/score/noul in one forward.
    Trial-pool first, like the Sept-25 sweep.
12. **RSI-Jev 0.8B / 2B** — github.com/Shanghua-Gao/RSI-Jev · open weights, POST
    /v1/systemone schema, one forward, self-measured confidence bins.
13. **Verdict (Manavarya09) 118M** — github.com/Manavarya09/verdict · Apache-2.0
    multilingual bi-encoder, temperature scaling + conformal abstain set, CPU or
    browser via ONNX, fits on your own labels in seconds; card says it loses to
    Laya on typed decisions. Name collision with the benched Verdict 151M
    (heman10x) — would need a distinct display name.

### Not taken

- **Decision 2.0 Nox-4B / Lux-9B / Vega-27B** (vllm-sr, snapshots 2026-09-29 +
  GGUFs 2026-10-08) — family extension of the benched Kai/Eos/Sol. Nox-4B fp32
  ≈ 16 GB would fit, Lux/Vega do not; GGUFs need Ollaya support that isn't
  there yet. The size ladder is already covered 0.6→2B; revisit if Ollaya adds
  them or Nox turns out multimodal (a Nox-Omni-4B third-party remix exists).
- **Vela-2.0** 0.3B/0.8B/4B/9B (vllm-sr, 2026-10-02/03) — the 1.0 line was
  dismissed as fixed-taxonomy; unverified whether 2.0 is schema-driven. Check
  before the next sweep.
- **Microsoft-Decision-1 local** — weights closed.
- **jevos** (1B 17-layer MiniCPM, Noul-only GGUF over llama.cpp CPU) — noul-only
  contract needs the Span-01 per-label mapping; niche next to the systems here.
- **jevlike / PocketJev / jev-visual / jevmlx / JEVfire** — demos or
  Apple/CUDA-only runtimes, not schema-driven engines for this bench.
- **Cloudflare clef-omni local twin / Drex local twin** — RAM/fork constraints
  above.

## Integration order

Hosted first (cheap, registry-only), then official local weights, then the
Ollaya quartet, then catalog trials:

1. **Clef Omni** — engines/clef_client.py MODELS + INPUT_USD_PER_MTOK; CLEF_MODELS
   rows in bench_spectrum.py / bench_multilingual.py / app.py; both benches; README.
2. **Microsoft-Decision-1** — OPENROUTER_SYSTEMONE rows in the three registries;
   both benches; README.
3. **Drex v1.5** — same; license note (Open RAIL-M weights, Apache-2.0 runtime)
   + local-twin caveat; both benches; README.
4. **d1-3B** — engines/d1_client.py (trust_remote_code loader, D1_HOME override)
   + demos/d1_demo.py (text tour + image stop) + benches in .venv-von + README.
5. **d1-omni-600M** — engine extension + demos/d1_omni_demo.py (image stop; audio
   if a generated clip is feasible) + benches + README.
6. **Ollaya: decima → snap → arbiter → credence** — ollaya_client registry +
   daemon pulls + benches (multilingual suite matters for decima/arbiter).
7. **Trials: eve-rlcd / RSI-Jev / Verdict-MM** — mini-pool trials per the _trials
   harness; graduate only if they beat the integrated field.
8. Close out: Luna re-verification date, wiring tests, feature-comparison column,
   cost table rows, charts, repo layout.

## Outcome (2026-10-10, same day)

All eight steps executed: the spectrum bench and the multilingual
suite each cover 85 systems (83 before arbiter and credence landed),
and the repo header counts 88 systems / 85 benchmarked.

| System | Spectrum (48q) | Multilingual (54 texts) | Notes |
|---|---|---|---|
| Clef-omni (Workers AI) | 89.6% · 0.040 s/q | 98.1% (100/100/94) | derived $1.4e-5/q (4.5k/8.8k input tokens × $0.15/M) |
| Microsoft Decision-1 (OpenRouter) | 91.7% · 0.065 s/q | 100% (100/100/100) | $2.1e-6/q; Azure endpoint passed the ZDR routing probe |
| Drex v1.5 (OpenRouter) | 93.8% · 0.075 s/q | 98.1% (100/100/94) | $1.4e-6/q — leanest prompts of the hosted field |
| d1-3B (local) | 83.3% · 5.818 s/q | 90.7% (100/100/72) | image stop 0.987 in the demo; engines/d1_client.py |
| d1-omni-600M (local) | 83.3% · 0.227 s/q | 51.8% (83/39/33) | ties its 25× larger sibling on English; English-centric card reads true; SAPI 16 kHz speech stop in demos/d1_demo.py |
| decima 321M (Ollaya) | 83.3% · 0.256 s/q | 92.6% (100/100/78) | best rare tier of any local system under 4B |
| decima small 122M (Ollaya) | 83.3% · 0.103 s/q | 68.5% (72/67/67) | fastest row on the board |
| snap 2B (Ollaya) | 81.2% · 2.995 s/q | 90.7% (100/100/72) | beyond its English+Italian card |
| arbiter 4B (Ollaya) | 89.6% · 4.878 s/q | 96% (100/100/89) | new best local classifier (past winnow e4b 87.5%); rare tier 89%, behind winnow/credence's 94% |
| credence 7.5B (Ollaya) | 85.4% · 6.775 s/q | 98.1% (100/100/94) | card's caveat held: below winnow on the pool (87.5 vs 85.4), multilingual level (98.1 vs 98, rare 94 = 94) |

Trials (fixed 12-question mini-pool; none beat the 12/12 graduates
GLiNER2.5-Decide / JevK5-Lite, so none promoted):

- anthonym21/qwen3-0.6b-rlcd-decision (eve-rlcd): 11/12 at 2.224 s/q —
  beats its integrated LFM2.5-RLCD sibling's 10/12.
- shgao/rsi-jev-v6.1-vl-4b: 10/12 at 3.267 s/q (rsi-jev serve on
  :11436; the repo JevClient speaks to it unchanged).
- Manavarya09/verdict "Verdict-MM": 9/12 at 0.090 s/q (verdictml 0.1.0).

Gotchas hit during integration (for the next sweep): Ollaya 0.7.2 was
too old for the quartet (decima needs ≥0.11, snap ≥0.10; upgraded to
0.12.1); arbiter's first pull died with a registry "error decoding
response body" and succeeded on retry; the RSI-Jev server serves under
the full name `rsi-jev-v6.1-vl-4b` (short alias rejected 422);
eve-rlcd's Decider.load defaults to CUDA (device="cpu" required); the
_trials/rlcd stub from the LFM2.5 trial shadowed the site-packages
`rlcd` (moved to _trials/lfm25_rlcd_pkg/).
