# zero-shot-ie-bench

Zero-shot information-extraction and classification benchmark: a curated
catalog of decision, classification, NER and reranking systems — hosted
APIs and local runner islands — benched against versioned, multilingual
suites, with an append-only run history and generated result tables.

- **Catalog** (`catalog/`): one YAML per system; providers, tracks,
  languages, machines. Schema-validated (`bun run validate`).
- **Suites** (`suites/`): versioned parallel-item pools with content
  hashes. Canary suites are the cheap scheduled nerf-detector.
- **Runs** (`runs/`): append-only; every table below is generated from
  them (`bun run lake && bun run render`). Rebenching adds history; a
  nerf or improvement shows up as drift.
- **Bench** (`bench/`): TypeScript + Bun — provider adapters (one per
  API contract), scorers per track, `run | canary | doctor`.
- **Runners** (`runners/`): one independent Python/uv project per
  dependency island, all speaking one HTTP protocol; a machine that
  benches needs Bun next to uv, nothing else.
- **Board / Playground** (`apps/`): static decision board and a
  try-it UI for hosted + live-local models.

Quickstart: `bun install` · `bun run check` (typecheck + tests + lint) ·
`bun run render` (lake + README tables) · `bun run --cwd bench dev
doctor`. Deeper docs: [methodology](docs/methodology.md) (scoring,
deferred runs, cost model) · [adding a model](docs/adding-a-model.md)
(the 3-step recipe) · [the restructure plan](docs/plan/00-index.md).

The v0 era (2026-09-20 → 2026-10-24, 81 systems / 38 families) is
preserved verbatim at tag `v0-final`; history since lives in
[CHANGELOG.md](CHANGELOG.md) and the run records themselves.

## Mixed-pool spectrum benchmark (all 78 systems)

Every system answers the same 48 mixed classification questions
(sentiment + topic, measured difficulty); extractors also answer 18 NER
questions. † = hosted via a non-ZDR endpoint.

<!-- gen:mixed-pool -->
| System | Classification | s/question | NER exact | s/question |
|---|---|---|---|---|
| Solar Decide (OpenRouter) † | **97.9%** | 0.654 | n/a |  |
| GPT-6 Luna Decisions (OpenRouter) † | 95.8% | 0.419 | n/a |  |
| D1 (OpenRouter) † | 93.8% | 0.031 | n/a |  |
| Solar Decide Flash (OpenRouter) † | 93.8% | 0.069 | n/a |  |
| Decider V1 27B (OpenRouter) | 93.8% | 0.075 | n/a |  |
| Decider V1.1 27B (OpenRouter) | 93.8% | 0.080 | n/a |  |
| Microsoft Decision-1 (OpenRouter) | 91.7% | 0.065 | n/a |  |
| TypeLLM thinking (hosted) † | 91.7% | 0.338 | n/a |  |
| Decision 3.0 Nano 2B (local) | 91.7% | 9.376 | n/a |  |
| Drex v1.5 (OpenRouter) | 91.7% | 0.537 | n/a |  |
| Decider V1.2 27B (OpenRouter) | 91.7% | 0.539 | n/a |  |
| D1-ultra (OpenRouter) † | 91.7% | 0.602 | n/a |  |
| Mercury Decide (OpenRouter) † | 89.6% | 0.051 | n/a |  |
| Clef-omni (Workers AI) | 89.6% | 0.040 | n/a |  |
| arbiter 4B (Ollaya) | 89.6% | 4.878 | n/a |  |
| Jev (cloud, jev-1.14.0) | 89.6% | 0.246 | n/a |  |
| GLiDE (Fastino) | 89.6% | 0.656 | n/a |  |
| Elephant-1 9B (OpenRouter) † | 89.6% | 3.361 | n/a |  |
| winnow e4b (Ollaya) | **87.5%** | 8.138 | n/a |  |
| qwen3-reranker-8b (OpenRouter) † | 87.5% | 0.564 | n/a |  |
| TypeLLM (hosted) † | 87.5% | 0.411 | n/a |  |
| GLiNER2.5-Decide | 85.4% | 0.412 | 61% | 0.476 |
| GLiNER2.5-Decide (Fastino) | 85.4% | 0.853 | 56% | 0.832 |
| Decider Flash V1.2 (OpenRouter) | 85.4% | 0.066 | n/a |  |
| Kev 4B (OpenRouter) † | 85.4% | 0.079 | n/a |  |
| Span-01 † | 85.4% | 0.440 | n/a |  |
| jevk5 4B (Ollaya) | 85.4% | 7.299 | n/a |  |
| credence 7.5B (Ollaya) | 85.4% | 6.775 | n/a |  |
| Intern-Decision 4B | 85.4% | 8.0 | n/a |  |
| GLiNER2.5-base | 83.3% | 0.111 | 61% | 0.135 |
| GLiNER2.5-base (Fastino) | 83.3% | 0.834 | 56% | 0.856 |
| Clef-flash (Workers AI) | 83.3% | 0.083 | n/a |  |
| decider 0.8B (local) | 83.3% | 2.144 | n/a |  |
| cohere-rerank-4-pro (OpenRouter) † | 83.3% | 0.418 | n/a |  |
| nli deberta-v3-large (Ollaya) | 83.3% | 0.369 | n/a |  |
| Decision 2.0 Kai 0.6B (local) | 83.3% | 0.640 | n/a |  |
| Decision 2.0 Sol 2B (local) | 83.3% | 2.578 | n/a |  |
| Decision 3.0 Edge 0.6B (local) | 83.3% | 2.745 | n/a |  |
| Decision 3.0 Lite 0.85B (local) | 83.3% | 5.354 | n/a |  |
| D1 3B (local) | 83.3% | 5.818 | n/a |  |
| D1-omni 600M (local) | 83.3% | 0.227 | n/a |  |
| decima 321M (Ollaya) | 83.3% | 0.256 | n/a |  |
| decima small 122M (Ollaya) | 83.3% | 0.103 | n/a |  |
| mxbai-rerank-large-v2 | 83.3% | 4.255 | n/a |  |
| gliclass-base | 81.2% | 0.110 | n/a |  |
| decision 0.75B (Ollaya) | 81.2% | 0.817 | n/a |  |
| gliclass-large | 81.2% | 0.411 | n/a |  |
| JevK5-Lite | 81.2% | 0.392 | n/a |  |
| snap 2B (Ollaya) | 81.2% | 2.995 | n/a |  |
| Intern-Decision 2B | 81.2% | 3.166 | n/a |  |
| GLiNER2.5-multi | 79.2% | 0.152 | **67%** | 0.180 |
| Tev1 4B (OpenRouter) | 79.2% | 0.081 | n/a |  |
| GLiNER2.5-multi (Fastino) | 79.2% | 0.831 | 61% | 0.849 |
| GLiFormer-large | 79.2% | 0.412 | 61% | 0.411 |
| AgentJev 0.6B (local) | 79.2% | 0.646 | n/a |  |
| Span-01 Lite † | 79.2% | 0.440 | n/a |  |
| voyage-rerank-2.5 (OpenRouter) † | 79.2% | 0.373 | n/a |  |
| Decision 2.0 Eos 0.8B (local) | 79.2% | 0.981 | n/a |  |
| Intern-Decision 0.8B | 77.1% | 1.585 | n/a |  |
| nemotron-rerank-1b-v2 | 75.0% | 0.838 | n/a |  |
| von-1.0 (option-marker) | 75.0% | 0.13 | n/a |  |
| Kev 0.8B (local) | 72.9% | 0.587 | n/a |  |
| bge-reranker-v2-m3 | 72.9% | 0.339 | n/a |  |
| MoJev 0.85B | 72.9% | 3.375 | n/a |  |
| GLiNER2.5-small | 70.8% | 0.043 | 33% | 0.053 |
| OpenThai 0.8B (local) | 70.8% | 0.504 | n/a |  |
| Laya typed-decisions | 70.8% | 1.243 | n/a |  |
| mxbai-rerank-base-v2 | 70.8% | 1.338 | n/a |  |
| cohere-rerank-4-fast (OpenRouter) † | 70.8% | 0.371 | n/a |  |
| Laya (local) | 68.8% | 0.143 | n/a |  |
| Lumma-fev 4B | 68.8% | 2.921 | n/a |  |
| cohere-rerank-v3.5 (OpenRouter) † | 66.7% | 0.500 | n/a |  |
| nli modernbert-large (Ollaya) | 66.7% | 0.281 | n/a |  |
| GLiFormer-base | 66.7% | 0.139 | 56% | 0.145 |
| gliclass-modern-base | 66.7% | 0.051 | n/a |  |
| K2-Type 0.9B (local) | 66.7% | 0.601 | n/a |  |
| voyage-rerank-2.5-lite (OpenRouter) † | 64.6% | 0.358 | n/a |  |
| GTE-rerank-ModernBERT-base | 60.4% | 0.183 | n/a |  |
| Lumma-fev 0.15B | 56.2% | 0.256 | n/a |  |
| Lumma-fev 0.6B | 54.2% | 0.587 | n/a |  |
| Clef (Workers AI) | 0.0% | n/a | n/a |  |
<!-- /gen:mixed-pool -->

### Measured cost per hosted run

The provider's own usage accounting per run — never list-price
reconstruction (the four tokens-but-no-$ providers carry v0's derived
values, flagged in the run meta).

<!-- gen:cost -->
| System | 48-q run | 54-text run | $ / question |
|---|---|---|---|
| Span-01 Lite † | $0 | $0 | $0 (free tier) |
| Mercury Decide (OpenRouter) † | $0 | $0 | $0 (free tier) |
| GLiNER2.5-base / -multi / -Decide (Fastino) | $0.000026 | $0.000031 | $0.00000039 |
| Span-01 † | $0.000043 | $0.000040 | $0.0000009 |
| Decider Flash V1.2 (OpenRouter) | $0.000051 | $0.000062 | $0.0000011 |
| voyage-rerank-2.5-lite † | $0.000068 | $0.000115 | $0.0000014 |
| Kev 4B (OpenRouter) | $0.000072 | $0.000106 | $0.0000015 |
| Microsoft Decision-1 (OpenRouter) | $0.000103 | $0.000149 | $0.0000021 |
| Decider V1.1 27B (OpenRouter) | $0.000105 | $0.000127 | $0.0000022 |
| D1 (OpenRouter) † | $0.000130 | $0.000163 | $0.0000030 |
| voyage-rerank-2.5 † | $0.000170 | $0.000287 | $0.0000035 |
| Decider V1 27B (OpenRouter) | $0.000211 | $0.000254 | $0.0000044 |
| Tev1 4B (OpenRouter) | $0.000229 | $0.000269 | $0.0000050 |
| Clef-flash (Workers AI) | $0.00041 | $0.00078 | $0.0000086 |
| Clef-omni (Workers AI) | $0.00068 | $0.0013 | $0.000014 |
| Solar Decide Flash (OpenRouter) † | $0.00087 | $0.0010 | $0.0000181 |
| TypeLLM thinking (hosted) † | $0.0029 | $0.0027 | $0.000061 |
| qwen3-reranker-8b † | $0.0032 | $0.0036 | $0.0000667 |
| cohere-rerank-v3.5 † | $0.048 | $0.054 | $0.0010 |
| cohere-rerank-4-fast † | $0.096 | $0.108 | $0.0020 |
| cohere-rerank-4-pro † | $0.120 | $0.135 | $0.0025 |
| Clef (Workers AI) | $0 | $0.0021 | $0 |
| D1-ultra (OpenRouter) † | $0 | $0.000171 | $0 |
| Decider V1.2 27B (OpenRouter) | $0 | $0.000124 | $0 |
| Drex v1.5 (OpenRouter) | $0 | $0.000101 | $0 |
| Elephant-1 9B (OpenRouter) † | $0 | $0 | $0 (free tier) |
| GLiDE (Fastino) | $0 | $0.00071 | $0 |
| GPT-6 Luna Decisions (OpenRouter) † | $0 | $0.000759 | $0 |
| Jev (cloud, jev-1.14.0) | $0 | $0.00027 | $0 |
| Solar Decide (OpenRouter) † | $0 | $0.001 | $0 |
| TypeLLM (hosted) † | $0 | $0.00017 | $0 |
<!-- /gen:cost -->

## Multilingual benchmark (9 languages, no English)

The same 54 parallel texts per system; tiers are the language grouping
(popular / medium / rare scripts).

<!-- gen:multilingual -->
| System | Popular | Medium | Rare | All |
|---|---|---|---|---|
| Jev (cloud, jev-1.14.0) | **100%** | **100%** | **100%** | **100%** |
| jevk5 4B (Ollaya) | 100% | 100% | 100% | **100%** |
| Solar Decide (OpenRouter) † | 100% | 100% | 100% | **100%** |
| Solar Decide Flash (OpenRouter) † | 100% | 100% | 100% | **100%** |
| Clef (Workers AI) | 100% | 100% | 100% | **100%** |
| Clef-flash (Workers AI) | 100% | 100% | 100% | **100%** |
| GLiDE (Fastino) | 100% | 100% | 100% | **100%** |
| TypeLLM (hosted) † | 100% | 100% | 100% | **100%** |
| TypeLLM thinking (hosted) † | 100% | 100% | 100% | **100%** |
| Decider V1 27B (OpenRouter) | 100% | 100% | 100% | **100%** |
| Decider V1.1 27B (OpenRouter) | 100% | 100% | 100% | **100%** |
| D1 (OpenRouter) † | 100% | 100% | 100% | **100%** |
| Tev1 4B (OpenRouter) | 100% | 100% | 100% | **100%** |
| Mercury Decide (OpenRouter) † | 100% | 100% | 100% | **100%** |
| Microsoft Decision-1 (OpenRouter) | 100% | 100% | 100% | **100%** |
| Decider V1.2 27B (OpenRouter) | 100% | 100% | 100% | **100%** |
| D1-ultra (OpenRouter) † | 100% | 100% | 100% | **100%** |
| Elephant-1 9B (OpenRouter) † | 100% | 100% | 100% | **100%** |
| GPT-6 Luna Decisions (OpenRouter) † | 100% | 100% | 100% | **100%** |
| winnow e4b (Ollaya) | 100% | 100% | 94% | 98% |
| Span-01 † | 100% | 100% | 94% | 98% |
| Kev 4B (OpenRouter) † | 100% | 100% | 94% | 98% |
| Intern-Decision 4B | 100% | 100% | 94% | 98% |
| credence 7.5B (Ollaya) | 100% | 100% | 94% | 98% |
| Clef-omni (Workers AI) | 100% | 100% | 94% | 98% |
| Drex v1.5 (OpenRouter) | 100% | 100% | 94% | 98% |
| Decider Flash V1.2 (OpenRouter) | 100% | 100% | 94% | 98% |
| qwen3-reranker-8b (OpenRouter) † | 100% | 100% | 89% | 96% |
| Lumma-fev 4B | 100% | 100% | 89% | 96% |
| arbiter 4B (Ollaya) | 100% | 100% | 89% | 96% |
| Decision 3.0 Nano 2B (local) | 100% | 100% | 78% | 93% |
| decima 321M (Ollaya) | 100% | 100% | 78% | 93% |
| Intern-Decision 2B | 100% | 100% | 72% | 91% |
| D1 3B (local) | 100% | 100% | 72% | 91% |
| snap 2B (Ollaya) | 100% | 100% | 72% | 91% |
| GLiNER2.5-multi (mDeBERTa) | 100% | 100% | 67% | 89% |
| GLiNER2.5-multi (Fastino) | 100% | 100% | 67% | 89% |
| decision 0.75B (Ollaya) | 100% | 94% | 72% | 89% |
| Intern-Decision 0.8B | 94% | 94% | 78% | 89% |
| Span-01 Lite † | 100% | 100% | 61% | 87% |
| Decision 3.0 Lite 0.85B (local) | 100% | 100% | 61% | 87% |
| decider 0.8B (local) | 100% | 100% | 50% | 83% |
| OpenThai 0.8B (local) | 100% | 100% | 50% | 83% |
| MoJev 0.85B | 100% | 100% | 50% | 83% |
| Decision 2.0 Sol 2B (local) | 100% | 100% | 50% | 83% |
| mxbai-rerank-large-v2 | 100% | 100% | 50% | 83% |
| K2-Type 0.9B (local) | 100% | 94% | 56% | 83% |
| cohere-rerank-4-pro (OpenRouter) † | 100% | 89% | 61% | 83% |
| gliclass-large | 100% | 89% | 56% | 81% |
| Decision 2.0 Eos 0.8B (local) | 89% | 89% | 61% | 80% |
| Kev 0.8B (local) | 94% | 89% | 50% | 78% |
| JevK5-Lite | 100% | 89% | 44% | 78% |
| Laya Router (mmBERT) | 89% | 94% | 44% | 76% |
| Decision 2.0 Kai 0.6B (local) | 94% | 83% | 44% | 74% |
| voyage-rerank-2.5 (OpenRouter) † | 83% | 78% | 61% | 74% |
| nemotron-rerank-1b-v2 | 89% | 72% | 61% | 74% |
| nli deberta-v3-large (Ollaya) | 89% | 78% | 39% | 69% |
| GLiNER2.5-Decide | 89% | 83% | 33% | 69% |
| GLiNER2.5-Decide (Fastino) | 89% | 83% | 33% | 69% |
| decima small 122M (Ollaya) | 72% | 67% | 67% | 69% |
| gliclass-base | 94% | 67% | 39% | 67% |
| bge-reranker-v2-m3 | 67% | 67% | 67% | 67% |
| cohere-rerank-v3.5 (OpenRouter) † | 67% | 67% | 67% | 65% |
| voyage-rerank-2.5-lite (OpenRouter) † | 94% | 61% | 39% | 65% |
| Decision 3.0 Edge 0.6B (local) | 94% | 72% | 28% | 65% |
| GLiFormer-large | 94% | 61% | 33% | 63% |
| AgentJev 0.6B (local) | 83% | 83% | 22% | 63% |
| Laya typed-decisions | 100% | 44% | 33% | 59% |
| GLiNER2.5-base | 89% | 50% | 28% | 56% |
| GLiNER2.5-base (Fastino) | 89% | 50% | 28% | 56% |
| cohere-rerank-4-fast (OpenRouter) † | 72% | 61% | 33% | 56% |
| GLiFormer-base | 83% | 44% | 33% | 54% |
| D1-omni 600M (local) | 83% | 39% | 33% | 52% |
| von-1.0 (option-marker) | 72% | 33% | 39% | 48% |
| mxbai-rerank-base-v2 | 56% | 50% | 39% | 48% |
| nli modernbert-large (Ollaya) | 61% | 39% | 39% | 46% |
| GTE-rerank-ModernBERT-base | 61% | 33% | 39% | 44% |
| gliclass-modern-base | 44% | 33% | 39% | 39% |
| Lumma-fev 0.6B | 28% | 39% | 44% | 37% |
| GLiNER2.5-small | 56% | 22% | 28% | 35% |
| Lumma-fev 0.15B | 39% | 33% | 33% | 35% |
<!-- /gen:multilingual -->

## Layout

```
catalog/    curated data: models, providers, tracks, languages, machines
suites/     versioned question pools (parallel items, hashed)
runs/       append-only run records (CI-enforced)
bench/      TS benchmark: adapters, scorers, CLI
runners/    Python model servers, one uv project per dependency island
packages/   schema (Zod SSOT) + lake (aggregation to build/)
apps/       board (static) + playground (try it)
docs/       methodology, adding-a-model, the restructure plan
legacy/     the v0 code (frozen; deleted at the end of the restructure)
```

Machine: `seoul` (this desktop — Ultra 9 185H, 32 GB, no discrete GPU;
amortised $/h recorded per machine file). Secrets stay in the
environment; the bench never writes them to runs.
