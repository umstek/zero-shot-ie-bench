# zero-shot-ie-bench

Thirty-nine zero-shot systems (thirty-eight of them benchmarked) across
twenty-three information-extraction and classification families —
extractor encoders, a purpose-built classifier, cross-encoder rerankers,
and typed-decision engines (local and cloud, the hosted ones behind
OpenRouter's decision and rerank endpoints) — demoed, benchmarked, and
cross-compared in one repo with a web UI.

| System | Kind | Size | License | Cost |
|---|---|---|---|---|
| [GLiNER 2.5](https://github.com/fastino-ai/GLiNER2) (`fastino/gliner2.5-*`) | local extractor encoder (boundary arch) | 74M / 194M / 287M | Apache 2.0 | $0 · local |
| [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) (`fastino/GLiNER2.5-Decide`) | local decision-tuned classifier in the GLiNER 2.5 family (spans + label heads, one pass) | 340M | Apache 2.0 | $0 · local |
| [GLiFormer](https://github.com/Knowledgator/GLiFormer) (`knowledgator/gliformer-*`) | local extractor encoder (layout-aware DeBERTa) | ~190M / 575.6M | Apache 2.0 | $0 · local |
| [GLiREL](https://github.com/jackboyla/GLiREL) (`jackboyla/glirel-large-v0`) | local zero-shot relation extractor (label-prompted encoder over entity pairs) | ~467M | CC BY-NC-SA 4.0 | $0 · local |
| [GLiClass](https://github.com/knowledgator/gliclass) (`knowledgator/gliclass-*-v3.0`) | local zero-shot classifier (all labels, one pass) | 33M / 151M / 187M / 439M | Apache 2.0 | $0 · local |
| [mxbai-rerank-base-v2](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v2) · [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) · [GTE-rerank-ModernBERT-base](https://huggingface.co/Alibaba-NLP/gte-rerank-modernbert-base) | local cross-encoder rerankers (score text+label pairs, argmax = decision) | 494M / 568M / 150M | Apache 2.0 | $0 · local |
| [Laya](https://huggingface.co/convaiinnovations/laya) (`laya`) | local typed-decision engine (choice/score/noul) | 421M (322M multilingual) | Apache 2.0 | $0 · local |
| [von-1.0](https://huggingface.co/wfzyx/von-1.0) (`von-sdk`) | local typed-decision engine (System One protocol) | 396M | Apache 2.0 | $0 · local |
| [open-alternative-jev](https://github.com/ikermoel/open-alternative-jev) (`so1`) | local decision harness over any ChatML LLM (logprobs) | BYO LLM (tested Qwen2.5-0.5B) | MIT | $0 · local |
| [Kev](https://github.com/jaredpalmer/kev) (`jaredpalmer/kev-0.8b`) | local decision engine (open-weight Jev lookalike, System One contract) | 0.8B (9.3M trained) | Apache 2.0 | $0 · local |
| [AgentJev](https://github.com/malevrigns/agent-jev) | local decision engine (candidate head over Qwen3-0.6B, own API) | 0.6B | Apache 2.0 | $0 · local |
| [decider](https://huggingface.co/Mapika/decider-0.8b) (`decider-ai`) | local decision engine (System One contract) | 0.8B | Apache 2.0 | $0 · local |
| [OpenThai-SystemOne](https://huggingface.co/iapp-technology/OpenThai-SystemOne) (`openthai-systemone`) | local decision engine (Thai/English, System One contract) | 0.8B | Apache 2.0 package; weights gated on HF | $0 · local |
| [Verdict](https://huggingface.co/heman10x/rlcd-modernbert-151m) | local decision encoder (ModernBERT + abstention head) | 151M | Apache 2.0 | $0 · local |
| [JevK5-Lite](https://huggingface.co/alibiserikbay/JevK5-Lite) (`jevk5` runtime) | local decision classifier (label-head encoder, one pass) | 437M | Apache 2.0 | $0 · local |
| [LFM2.5-RLCD](https://huggingface.co/notnotsamuel/LFM2.5-350M-RLCD) (`engines/rlcd_engine/`, vendored) | local decision engine (constrained decoding over LFM2.5) | 350M | engine MIT; weights LFM Open License v1.0 | $0 · local |
| [Certo 421M](https://huggingface.co/altslate/certo-decision-model) (`engines/certo_engine/`, vendored) | local decision model (calibrated per-option score head over ModernBERT-large, no generation) | 421M | MIT (engine + weights) | $0 · local |
| [MoJev](https://huggingface.co/MoLeMo-Lab/mojev) 0.85B (`engines/mojev_engine/`, adapted) | local decision engine (packed one-pass candidate scoring, Qwen3.5 + fla) | 0.85B | engine MIT; checkpoint card MIT (Qwen base-model license on the encoder weights) | $0 · local |
| [nanodiff 350M](https://huggingface.co/pngwn/nanodiff-350m-typed-decisions-lam1) (`engines/nanodiff_engine/`, vendored) | local decision model (bidirectional diffusion LM — the only non-autoregressive system here) | 350M | MIT | $0 · local |
| [Jev](https://www.typesafe.ai/) (`jev-latest` via System One API) | cloud typed-decision engine (choice/score/noul) | closed | proprietary API | metered · n/r ‡ |
| [Kev 4B](https://openrouter.ai/jaredpalmer/kev-4b) (`jaredpalmer/kev-4b` via OpenRouter) | cloud decision engine (same System One contract as local Kev, hosted) | 4B | proprietary API | metered · $1.5e-6/q † |
| [Span-01](https://openrouter.ai/respan/span-01) / [Span-01 Lite](https://openrouter.ai/respan/span-01-lite) (`respan/span-01*`) | cloud behavior scorer (one noul probability per label, argmax = decision) | closed | proprietary API | metered · $9.0e-7/q · Lite free † |
| [Qwen3-Reranker 8B](https://openrouter.ai/qwen/qwen3-reranker-8b) · [Voyage rerank-2.5](https://openrouter.ai/voyageai/rerank-2.5) (+lite) · [Nemotron Rerank VL 1B](https://openrouter.ai/nvidia/llama-nemotron-rerank-vl-1b-v2:free) · [Cohere Rerank](https://openrouter.ai/cohere/rerank-4-pro) (4 Pro / 4 Fast / v3.5), all via OpenRouter | cloud rerankers (same (instruction, label) argmax mapping as the local ones) | 8B / closed / 1.7B / closed | proprietary API | metered · $1.4e-6–$2.5e-3/q · Nemotron free † |

† Non-ZDR endpoints: these providers may retain request data (OpenRouter's
account privacy settings gate this — the account used here allows them).
Every local system in the repo keeps text on the machine. The Cost column
marks who bills per request: `$0 · local` systems cost no API money
(only CPU time), `metered` systems bill per token/search with the
**measured** $ per mixed-pool question shown (see [Measured cost per
hosted run](#measured-cost-per-hosted-run)); ‡ Jev's provider reports
tokens but no cost, so it has no measured $ figure.

Four mechanism families are represented — **extractors** (GLiNER 2.5,
GLiFormer: spans/entities/relations/records, plus GLiREL: zero-shot
relations over entity pairs it is handed), **classifiers** (GLiClass:
all labels in one forward pass), **cross-encoder rerankers** as decision
engines (score one (instruction, label) pair per label, argmax — local
trio + seven hosted), and **typed-decision engines** (ask `choice` /
`score` / `noul` questions over a JSON state, all answered in one
call — Laya, von, so1, the local open-weight Jev lookalikes Kev /
AgentJev / decider / OpenThai / JevK5-Lite, Verdict, LFM2.5-RLCD, Certo,
MoJev, nanodiff, and cloud Jev / Kev 4B / Span-01). The GLiNER lineage
forked between Fastino (GLiNER 2.5, original author Urchade Zaratiana)
and Knowledgator (classic `gliner`, GLiFormer, GLiClass); GLiREL
(jackboyla) grows out of the same architecture into relation extraction,
and Laya's card positions itself as the open local counterpart of cloud
Jev.

## Benchmark results

Measured on CPU, zero-shot, identical label sets and descriptions,
out-of-the-box defaults; every case runs 5 times. Full detail in
[`results/bench_results.json`](results/bench_results.json); methodology
in `bench.py`.

Classification accuracy (mean latency ± std per text):

| System | Sentiment (24) | Topic (12) | s/text (sentiment+topic mean) |
|---|---|---|---|
| GLiNER2.5-small | 83.3% | 100% | 0.053±0.019 |
| GLiNER2.5-base | 100% | 100% | 0.105±0.007 |
| GLiNER2.5-multi | 100% | 100% | 0.126±0.014 |
| GLiFormer-base | 100% | 75.0% | 0.118±0.013 |
| GLiFormer-large | 100% | 83.3% | 0.399±0.035 |
| Laya (local) | 95.8% | 83.3% | 0.164±0.009 (batched ÷ n) |
| Jev (cloud) | 100% | 100% | 0.052±0.002 (batched ÷ n) |

NER, strict document+span+label match over 30 gold entities (decision
engines have no span output):

| System | Precision | Recall | F1 | s/text |
|---|---|---|---|---|
| GLiNER2.5-small | 0.78 | 0.97 | 0.87 | 0.051±0.015 |
| GLiNER2.5-base | 0.94 | 1.00 | 0.97 | 0.111±0.007 |
| GLiNER2.5-multi | 0.97 | 1.00 | 0.98 | 0.127±0.037 |
| GLiFormer-base | 1.00 | 1.00 | **1.00** | 0.129±0.023 |
| GLiFormer-large | 0.97 | 1.00 | 0.98 | 0.447±0.066 |

**Every system was fully deterministic** — identical predictions on all
5 runs including the cloud API; only latency varies (σ ≈ 0.002–0.07 s).

Caveats worth knowing:

- The sentiment set is easy (everything ≥83%); GLiFormer topic misses
  lean "business" (base misses one more than large — but beats large on
  NER).
- Jev and Laya ran as one batched request/forward per task; per-text
  latency is batch ÷ n.
- **Laya prompt-format gotcha:** dict-shaped `instructions` (fine on Jev)
  collapse Laya onto one label (58.3%); benchmarked with strings, where
  it scores 95.8%.

## Mixed-pool spectrum benchmark (all 38 systems)

`bench_spectrum.py` — the headline comparison. Every system answers the
same **one mixed pool** of 48 classification questions (sentiment + topic,
varying difficulty); the six extractors also answer 18 NER questions.
Question difficulty is **measured** as the fraction of answering systems
that got it wrong; the web UI plots accuracy along that spectrum, and
per-question predictions live in `results/bench_spectrum_results.json`.

Classification accuracy on the mixed pool (48 questions), NER scored as
exact span-set match (18 questions). † = hosted via OpenRouter (non-ZDR):

| System | Classification | s/question | NER exact | s/question |
|---|---|---|---|---|
| Jev (cloud) | **93.8%** | 0.018 | n/a | |
| qwen3-reranker-8b (OpenRouter) † | 87.5% | 0.564 | n/a | |
| GLiNER2.5-Decide | 85.4% | 0.412 | 61% | 0.476 |
| Kev 4B (OpenRouter) † | 85.4% | 0.079 | n/a | |
| Span-01 † | 85.4% | 0.440 | n/a | |
| GLiNER2.5-base | 83.3% | 0.111 | 61% | 0.135 |
| decider 0.8B (local) | 83.3% | 2.144 | n/a | |
| cohere-rerank-4-pro (OpenRouter) † | 83.3% | 0.418 | n/a | |
| gliclass-base | 81.2% | 0.110 | n/a | |
| gliclass-large | 81.2% | 0.411 | n/a | |
| JevK5-Lite | 81.2% | 0.392 | n/a | |
| GLiNER2.5-multi | 79.2% | 0.152 | **67%** | 0.180 |
| GLiFormer-large | 79.2% | 0.412 | 61% | 0.411 |
| von-1.0 (option-marker) | 79.2% | 0.226 | n/a | |
| AgentJev 0.6B (local) | 79.2% | 0.646 | n/a | |
| Span-01 Lite † | 79.2% | 0.440 | n/a | |
| voyage-rerank-2.5 (OpenRouter) † | 79.2% | 0.373 | n/a | |
| Kev 0.8B (local) | 72.9% | 0.587 | n/a | |
| bge-reranker-v2-m3 | 72.9% | 0.339 | n/a | |
| MoJev 0.85B | 72.9% | 3.375 | n/a | |
| GLiNER2.5-small | 70.8% | 0.043 | 33% | 0.053 |
| OpenThai 0.8B (local) | 70.8% | 0.504 | n/a | |
| Laya typed-decisions | 70.8% | 1.243 | n/a | |
| mxbai-rerank-base-v2 | 70.8% | 1.338 | n/a | |
| cohere-rerank-4-fast (OpenRouter) † | 70.8% | 0.371 | n/a | |
| Laya (local) | 68.8% | 0.143 | n/a | |
| GLiFormer-base | 66.7% | 0.139 | 56% | 0.145 |
| gliclass-edge | 66.7% | **0.016** | n/a | |
| gliclass-modern-base | 66.7% | 0.051 | n/a | |
| cohere-rerank-v3.5 (OpenRouter) † | 66.7% | 0.500 | n/a | |
| voyage-rerank-2.5-lite (OpenRouter) † | 64.6% | 0.358 | n/a | |
| GTE-rerank-ModernBERT-base | 60.4% | 0.183 | n/a | |
| LFM2.5-RLCD 350M | 56.2% | 0.469 | n/a | |
| so1 + Qwen2.5-0.5B | 43.8% | 0.237 | n/a | |
| Verdict 151M (local) | 39.6% | 0.165 | n/a | |
| nemotron-rerank-vl-1b (OpenRouter) † | 35.4% | 4.521 | n/a | |
| nanodiff 350M | 25.0% | 11.56 | n/a | |
| Certo 421M | 22.9% | 1.014 | n/a | |

Takeaways:

- **qwen3-reranker-8b at 87.5% is the best non-Jev system of all 38** —
  one more datapoint for rerankers doubling as decision engines (bge
  locally reaches 72.9%, Kev/OpenThai level, where the same rerankers
  sit near zero on JevBench's leaderboard).
- Best local classifier: GLiNER2.5-Decide 85.4% (still NER-capable at
  61%); best local sarcasm reader: gliclass-large; speed king:
  gliclass-edge at 16 ms/question; most robust extractor:
  GLiNER2.5-multi.
- Hosted kev-4b ties Decide at 85.4% and is 2nd overall multilingual
  (98%, one Welsh text short of perfect); Span-01 matches it on both
  (85.4% / 98%); the cheaper hosted
  tiers fade fast (voyage-2.5 79%/74%, Cohere 4 Fast 71%/56%, v3.5
  67%/65%), and Nemotron VL is the one clear miss — a vision reranker
  scored on text-only pairs (35%/41%).
- von's 79.2% replaces a withdrawn 62.5% row (the PyPI SDK's default
  NLI loader had initialized untrained classifier weights).
- Verdict is the fastest local decision engine per question but
  abstains on 22/48 (abstention scores wrong): 73% on what it answers,
  39.6% overall. so1 works mechanically, but a 0.5B base LLM is not
  enough brain — swap in a bigger one.
- Chance-level negatives kept as census rows: nanodiff 350M (diffusion
  LM, 25.0%, slowest at 11.6 s/question) and Certo 421M (22.9%,
  near-uniform option probabilities).
- Latencies exclude model download/loading, include the first forward
  pass — except the local System One servers (Kev, decider, OpenThai),
  which answer one untimed warm-up question first (lazy weight
  loading). gliclass runs fine on pinned transformers 4.57.6 despite
  ≥5 metadata; von genuinely needs 5 (`.venv-von`).

### Measured cost per hosted run

Each hosted system's results entry carries the provider's own usage
accounting — the `usage` block billed per API response, never
reconstructed from list prices. Measured on these exact runs (one pass;
48 mixed-pool questions / 54 multilingual texts):

| System | 48-q run | 54-text run | $ / question |
|---|---|---|---|
| Span-01 Lite † | $0 | $0 | $0 (free tier) |
| nemotron-rerank-vl-1b † | $0 | $0 | $0 (free tier) |
| Span-01 † | $0.000043 | $0.000040 | $0.0000009 |
| voyage-rerank-2.5-lite † | $0.000068 | $0.000115 | $0.0000014 |
| Kev 4B (OpenRouter) | $0.000072 | $0.000106 | $0.0000015 |
| voyage-rerank-2.5 † | $0.000170 | $0.000287 | $0.0000035 |
| qwen3-reranker-8b † | $0.0032 | $0.0036 | $0.0000667 |
| cohere-rerank-v3.5 † | $0.048 | $0.054 | $0.0010 |
| cohere-rerank-4-fast † | $0.096 | $0.108 | $0.0020 |
| cohere-rerank-4-pro † | $0.120 | $0.135 | $0.0025 |
| Jev (cloud) | — | — | not reported ‡ |

The cost charts plot the **metered** systems only — free tiers bill $0
(Span-01 Lite's plain id is priced $0.0, same as its `:free` twin;
Nemotron runs on `:free`) and Jev reports no cost (‡ tokens only:
6,094 / 6,430 input across its 2 + 1 batched requests). What the
measurements say that the price lists don't: Cohere bills ~2.5 search
units per rerank request (a 48-q run on 4-pro costs $0.12, not the
naive 48 × $0.001); kev-4b and Span-01 sit near $1e-6/question — kev
batches each 24-question task into one request, Span's payloads are
tiny. Not benched on OpenRouter on purpose: `typesafe/jev-1.13`
(RBAC-gated; Jev measured via TypeSafe directly) and
`typesafe/jev-router` (a chat router, not a typed-decision endpoint).

The web UI renders these as interactive charts; the same charts, as
images (regenerate with `python make_chart_images.py`):

**Classification (48-question mixed pool)**

![Classification accuracy, mixed pool](docs/charts/cls_accuracy.png)

![Speed vs accuracy — up and left is better](docs/charts/cls_tradeoff.png)

![Cost vs accuracy, metered hosted systems — up and left is better](docs/charts/cls_cost.png)

![Measured cost per question, metered hosted systems (ranked)](docs/charts/cls_cost_bars.png)

![Accuracy vs question difficulty](docs/charts/cls_spectrum.png)

![Mean latency per question](docs/charts/cls_latency.png)

**NER exact-match (18 questions, extractors only)**

![NER exact-match rate, mixed pool](docs/charts/ner_exact.png)

![Exact match vs question difficulty](docs/charts/ner_spectrum.png)

## Multilingual benchmark (9 languages, no English)

`bench_multilingual.py` — the same six sentence meanings per language
(2 positive / 2 negative / 2 neutral sentiment), labels in English.
Popular: Spanish, French, Chinese · Medium: Vietnamese, Turkish,
Ukrainian · Rare: **Sinhala**, Icelandic, Welsh. Sentences were verified
by blind back-translation through an independent model instance (it
caught 8 errors, including a sentiment-flipping Sinhala word). All 38
systems answer the same 54 texts.

| System | Popular | Medium | Rare | All |
|---|---|---|---|---|
| Jev (cloud) | **100%** | **100%** | **100%** | **100%** |
| Span-01 † | 100% | 100% | 94% | 98% |
| Kev 4B (OpenRouter) † | 100% | 100% | 94% | 98% |
| qwen3-reranker-8b (OpenRouter) † | 100% | 100% | 89% | 96% |
| GLiNER2.5-multi (mDeBERTa) | 100% | 100% | 67% | 89% |
| Span-01 Lite † | 100% | 100% | 61% | 87% |
| decider 0.8B (local) | 100% | 100% | 50% | 83% |
| OpenThai 0.8B (local) | 100% | 100% | 50% | 83% |
| MoJev 0.85B | 100% | 100% | 50% | 83% |
| cohere-rerank-4-pro (OpenRouter) † | 100% | 89% | 61% | 83% |
| gliclass-large | 100% | 89% | 56% | 81% |
| Kev 0.8B (local) | 94% | 89% | 50% | 78% |
| JevK5-Lite | 100% | 89% | 44% | 78% |
| Laya Router (mmBERT) | 89% | 94% | 44% | 76% |
| voyage-rerank-2.5 (OpenRouter) † | 83% | 78% | 61% | 74% |
| gliclass-base | 94% | 67% | 39% | 67% |
| bge-reranker-v2-m3 | 67% | 67% | 67% | 67% |
| cohere-rerank-v3.5 (OpenRouter) † | 67% | 67% | 67% | 65% |
| voyage-rerank-2.5-lite (OpenRouter) † | 94% | 61% | 39% | 65% |
| GLiNER2.5-Decide | 89% | 83% | 33% | 69% |
| GLiFormer-large | 94% | 61% | 33% | 63% |
| AgentJev 0.6B (local) | 83% | 83% | 22% | 63% |
| Laya typed-decisions | 100% | 44% | 33% | 59% |
| GLiNER2.5-base | 89% | 50% | 28% | 56% |
| cohere-rerank-4-fast (OpenRouter) † | 72% | 61% | 33% | 56% |
| GLiFormer-base | 83% | 44% | 33% | 54% |
| LFM2.5-RLCD 350M | 78% | 67% | 11% | 52% |
| von-1.0 (option-marker) | 72% | 33% | 39% | 48% |
| mxbai-rerank-base-v2 | 56% | 50% | 39% | 48% |
| GTE-rerank-ModernBERT-base | 61% | 33% | 39% | 44% |
| gliclass-modern-base | 44% | 33% | 39% | 39% |
| nemotron-rerank-vl-1b (OpenRouter) † | 50% | 33% | 39% | 41% |
| so1 + Qwen2.5-0.5B | 39% | 39% | 33% | 37% |
| GLiNER2.5-small | 56% | 22% | 28% | 35% |
| nanodiff 350M | 28% | 39% | 39% | 35% |
| gliclass-edge | 50% | 22% | 22% | 31% |
| Certo 421M | 28% | 28% | 33% | 30% |
| Verdict 151M (local) | 50% | 11% | 6% | 22% |

![Accuracy by system and language](docs/charts/ml_heatmap.png)

![Overall multilingual accuracy](docs/charts/ml_overall.png)

Highlights:

- Multilingual reach tracks the backbone's pretraining breadth: the
  Qwen3.5-0.8B trio (decider, OpenThai, MoJev) repeats 100/100/50
  exactly; AgentJev's smaller Qwen3-0.6B holds 83% through six
  languages then collapses on rare scripts (22%).
- GLiNER2.5-multi is perfect through Ukrainian but drops on Welsh/Sinhala
  (89% overall — best local); gliclass-large transfers surprisingly
  well for an English-family release (81%, 100% Chinese, joint-best
  local Sinhala 67%); Jev is the only 100%-on-Sinhala system.
- bge-reranker-v2-m3's flat 67/67/67 is structural: it never predicts
  neutral (35 neg / 19 pos across all 54 texts), so every language
  scores exactly 4/6 — right on the four subjective texts, wrong on
  both neutrals. Reproduced bit-for-bit with score dumps.
- Laya's English-only typed-decisions specialist holds 100% popular but
  collapses on non-Latin scripts (59% overall); the Router row (76%) is
  the family's best. English-only encoders (Verdict 22%) degrade with
  distance from English as expected.
- von's complete option-marker checkpoint scores 48% (old 33–48% runs
  used randomly initialized weights and are invalid; the rerun pins
  model + SDK revisions, recorded in the results file).

## Feature comparison

| Capability | GLiNER 2.5 | GLiFormer | GLiClass | Rerankers | Laya | von | so1 | Jev | Kev | AgentJev | decider | OpenThai | Verdict | JevK5-Lite | LFM2.5-RLCD | Certo | MoJev | nanodiff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ability group | Extractor | Extractor | Classifier | Cross-encoder rerankers (decision via argmax) | Decision engine | Decision engine | Decision engine (BYO LLM) | Decision engine (cloud) | Decision engine (local, open weights) | Decision engine (local, open weights) | Decision engine (local, open weights) | Decision engine (local, open weights) | Decision engine (local, encoder head) | Decision engine (local, label-head encoder) | Decision engine (local, constrained decoding) | Decision engine (local, per-option score head) | Decision engine (local, packed one-pass scoring) | Decision engine (local, diffusion LM) |
| Zero-shot NER spans | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Text classification | ✅ | ✅ | ✅ | ✅ via argmax | ✅ choice | ✅ choice | ✅ choice | ✅ choice | ✅ choice | ✅ choice | ✅ choice | ✅ choice | ✅ choice | ✅ choice | ✅ choice | ✅ choice | ✅ choice | ✅ choice |
| All labels scored in one pass | ✅ | ✅ | ✅ (its core design) | ❌ one pair per label | ✅ | ✅ | ✅ packed | ✅ one request | ✅ one request | ✅ one request | ✅ one request | ✅ one request | ✅ per query | ✅ one pass | ✅ per field | ✅ one pass | ✅ packed | ✅ one forward |
| Relations | ✅ + JointIE graph | ✅ joint head | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Span attributes (per-entity sentiment) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Structured records | ✅ flat, anchor-based | ✅ nested Pydantic | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ flat closed schema | ❌ | ❌ | ❌ |
| Ordinal score rubrics | ✅ via Decide (untested here) | ❌ | ❌ | ❌ | ✅ score | ✅ rate | ✅ | ✅ score | ✅ score | ✅ score | ✅ score | ✅ score | ✅ score (untested here) | ❌ (lite is classification-only) | ❌ | ❌ | ❌ | ❌ |
| Yes/no judgments | ❌ | ❌ | ❌ | ❌ | ✅ noul | ✅ judge | ✅ yes_no | ✅ noul | ✅ noul | ✅ boolean | ✅ noul | ✅ noul | ✅ noul (untested here) | ❌ | ✅ boolean | ❌ | ❌ | ❌ |
| Text embeddings | ❌ | ✅ 1024-d | ❌ (reranker-capable) | ❌ (cross-encoders only) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Multilingual | ✅ multi ckpt (89% over 9 langs here) | ❌ English (63%) | ✅ large 81% over 9 langs | bge-v2-m3 67% over 9 langs; mxbai 48% / GTE 44% | ✅ Router, 100+ langs (76%) | option-marker: 48% over 9 langs | = base LLM's languages (37%) | ✅ 100% incl. Sinhala | ✅ 78% over 9 langs | 63% over 9 langs | 83% over 9 langs | 83% over 9 langs | 22% over 9 langs | ✅ 78% over 9 langs | 52% over 9 langs | 30% over 9 langs | ✅ 83% over 9 langs | 35% over 9 langs |
| Runs offline / data local | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Cost | free | free | free | free | free | free | free | $0.042/1M input | free (CPU time) | free (CPU time) | free (CPU time) | free (CPU time) | free (CPU time) | free (CPU time) | free (CPU time) | free (CPU time) | free (CPU time) | free (CPU time) |
| License | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 | MIT (lib) | proprietary API | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 (package); weights gated | Apache 2.0 | Apache 2.0 | MIT (engine); LFM Open License v1.0 (weights) | MIT (engine + weights) | MIT (engine; Qwen base-model license on encoder weights) | MIT |
| Batch shape | per text | per text (batch_size) | per text, all labels | per text, one pair per label | all questions, one pass | per text | one packed prompt | all questions, one request | all questions, one request | all questions, one request | all questions, one request | all questions, one request | per text, all options | per text, all heads + labels | per text, all field candidates | per text, all option descriptions | per text, all packed candidates | per text, one masked forward |

## Setup

Python 3.10+ for the main environment; 3.12+ for von (tested on 3.13,
Windows, CPU-only). Shell commands assume a POSIX shell — Git Bash on
Windows works; the `NAME=value` server launches in particular will not
parse in PowerShell/cmd.

```bash
uv venv .venv
uv pip install --python .venv -r requirements.txt --overrides overrides.txt
# so1 is not on PyPI (needed by the so1 tab + benchmarks):
uv pip install --python .venv "open-alternative-jev @ git+https://github.com/ikermoel/open-alternative-jev"
# the three cross-encoder rerankers run in-process in the main venv;
# the reranker-as-decision-engine path needs sentence-transformers:
uv pip install --python .venv sentence-transformers==5.7.0
# von needs transformers 5.x, so it lives in its own venv:
uv venv --python 3.13 .venv-von
uv pip install --python .venv-von -r requirements-von.txt
# Kev (Jev's open-weights lookalike) serves the same System One contract
# locally; it needs transformers >=5.17, so it runs from its own clone:
git clone https://github.com/jaredpalmer/kev.git ../kev
cd ../kev && uv sync --extra serve && cd -
# start it before benchmarking "Kev 0.8B (local)":
uv run --extra serve --project ../kev python -m kev.serve \
    --run jaredpalmer/kev-0.8b --port 8009
# AgentJev (own /api/evaluate contract) also runs from its own clone;
# weights convert to .pt once (see its model card):
git clone https://github.com/malevrigns/agent-jev.git ../agent-jev
# decider + OpenThai-SystemOne also serve the System One contract; both
# live in a shared transformers-5 CPU venv (here C:/venvs/agent-jev):
uv venv C:/venvs/agent-jev
uv pip install --python C:/venvs/agent-jev/Scripts/python.exe \
    "decider-ai[serve]" "openthai-systemone[server]" onnxruntime
# start them (separate terminals) before benchmarking; both lazy-load
# weights on the first request, so the benchmarks fire one untimed
# warm-up question first:
DECIDER_MODEL=Mapika/decider-0.8b DECIDER_DEVICE=cpu \
    C:/venvs/agent-jev/Scripts/python -m uvicorn decider.serve:app --port 8018
OPENTHAI_SYSTEMONE_MODEL=iapp/OpenThai-SystemOne \
    C:/venvs/agent-jev/Scripts/python -m uvicorn \
    openthai_systemone.server:app --port 8029
# Verdict runs in-process (no server) from the Verdict-open-jev checkout;
# set VERDICT_HOME if it is not at C:\src\verdict (its artifacts download
# from heman10x/rlcd-modernbert-151m), and run the benchmarks with the
# agent-jev venv python.
# JevK5-Lite, LFM2.5-RLCD 350M and MoJev 0.85B also run in .venv-von
# (transformers 5.17 is already pinned there): the jevk5 lite runtime plus
# jsonschema. The LFM engine itself is vendored in engines/rlcd_engine/ (engine
# code MIT; the LiquidAI/LFM2.5-350M weights it loads are under the LFM
# Open License v1.0). MoJev is a 0.85B Qwen3.5 + fla packed one-pass scorer
# whose scorer class loads from the checkpoint via trust_remote_code; its
# request path is adapted in engines/mojev_engine/ (MIT). Certo (engines/certo_engine/,
# vendored from the certo repo, MIT) and nanodiff (engines/nanodiff_engine/,
# vendored from BY571/nanoDiff + the pngwn release, MIT) run in the main
# venv — nothing extra to install beyond sentence-transformers above:
uv pip install --python .venv-von/Scripts/python.exe "jevk5[lite]==0.3.1" jsonschema
```

Use uv for this install: `overrides.txt` deliberately overrides GLiClass
0.1.20's transformers ≥5 metadata with the validated 4.57.6 version required
by GLiNER2. This is an explicit compatibility exception for that pinned
GLiClass release; plain pip dependency resolution cannot install this mix.
Altair is included explicitly for the benchmark charts. `protobuf` and
`sentencepiece` are included explicitly because `gliner2[local]` does not
pull them in and the DeBERTa tokenizer needs them. Model checkpoints
(~0.3–2.3 GB each) download from Hugging Face on first run. Cloud keys
live in a repo-root `.env` (gitignored): `TYPESAFE_API_KEY` for Jev
(paid API, see `engines/jev_client.py`) and `OPENROUTER_API_KEY` for the
hosted OpenRouter systems (see `engines/openrouter_client.py`).

## Run

```bash
.venv/Scripts/python demos/demo.py       # GLiNER 2.5 tour: entities,
                                         # classification, relations, JointIE,
                                         # span attributes, records — ends
                                         # with the decision-tuned sibling
                                         # GLiNER2.5-Decide (--model decide
                                         # tours only that checkpoint)
.venv/Scripts/python demos/demo_gliformer.py   # GLiFormer tour + embeddings
                                          # (--model base or large)
.venv/Scripts/python demos/demo_glirel.py      # GLiREL zero-shot relations
                                         # (GLiNER 2.5 base supplies the
                                         # entity spans it scores pairs of)
.venv/Scripts/python demos/demo_laya.py        # Laya tour + multilingual Router
.venv/Scripts/python demos/demo_jev.py         # Jev tour (2 paid API requests)
.venv-von/Scripts/python demos/jevk5_demo.py   # JevK5-Lite tour (label-head
                                         # encoder; transformers-5 venv)
.venv-von/Scripts/python demos/lfm_rlcd_demo.py # LFM2.5-RLCD constrained-decoding
                                          # tour (vendored engines/rlcd_engine/)
.venv/Scripts/python demos/reranker_demo.py    # three cross-encoder rerankers as
                                         # decision engines: pair scores +
                                         # argmax (sentence-transformers)
.venv/Scripts/python demos/certo_demo.py       # Certo 421M calibrated decide()
                                         # tour (vendored engines/certo_engine/)
.venv/Scripts/python demos/nanodiff_demo.py    # nanodiff 350M diffusion-LM tour
                                         # (~10 s/question; vendored
                                         # engines/nanodiff_engine/)
.venv-von/Scripts/python demos/mojev_demo.py   # MoJev 0.85B packed one-pass
                                         # scoring (transformers-5 venv)

.venv/Scripts/python bench.py            # flat suite, 5 runs per case
                                         # (--repeats N) → results/bench_results.json
.venv/Scripts/python bench_spectrum.py --system <name>   # one mixed pool per
                                         # system → results/bench_spectrum_results.json
                                         # (38 systems; von, JevK5-Lite,
                                         # LFM2.5-RLCD 350M and MoJev 0.85B:
                                         # same command under
                                         # .venv-von/Scripts/python)
# decider/OpenThai speak plain HTTP, so any interpreter works — but Verdict
# imports rlcd in-process and needs the transformers-5 agent-jev interpreter:
C:/venvs/agent-jev/Scripts/python bench_spectrum.py --system "Verdict 151M (local)"
.venv/Scripts/python bench_multilingual.py --system <name>
                                         # 9 languages, all 38 systems →
                                         # results/bench_multilingual_results.json
                                         # (same interpreter rules)
.venv/Scripts/python app.py              # web UI at http://127.0.0.1:7860
```

## Web UI

`app.py` serves a live demo tab per family — GLiNER 2.5 (checkpoint
selector includes GLiNER2.5-Decide), GLiFormer, GLiREL, GLiClass,
Rerankers, Laya, von, JevK5-Lite, LFM2.5-RLCD, Certo, MoJev, nanodiff,
so1, Jev (cloud), and OpenRouter (all ten hosted systems: Kev 4B,
Span-01/Lite, seven rerankers — one metered API request per click, needs
`OPENROUTER_API_KEY`) — plus benchmark tabs (classification / extraction
/ multilingual: tables and charts from `results/bench_*_results.json`,
including the cost-vs-accuracy charts) and a Compare tab (feature
matrix). The remaining local engines (Kev, AgentJev, decider, OpenThai,
Verdict) run as separate servers or venvs and are covered by the
benchmark and compare tabs.

The `.venv-von` systems (von, JevK5-Lite, LFM2.5-RLCD, MoJev) spawn
one-shot `demos/*_demo.py --serve` runners, each loading its model once
per click in a single `.venv-von` process; nanodiff does the same under
the main venv. von's shared loader (`engines/von_client.py`) pins the
upstream SDK and model revision and requires the complete
`option_marker.pt` state dict — missing or incompatible weights fail
before inference; there is no random-head fallback (first use downloads
~3 GB).

Regression tests run without model downloads or cloud credentials:

```bash
.venv/Scripts/python -m unittest discover -s tests -v
```

## Family maps

Version coverage per family (every listed size is benchmarked, except the
extra Laya checkpoints noted below):

| Checkpoint | Params | Notes |
|---|---|---|
| `fastino/gliner2.5-small-v1` | 74M | DeBERTa-v3-xsmall, fast CPU — benchmarked |
| `fastino/gliner2.5-base-v1` | 194M | default English — benchmarked |
| `fastino/gliner2.5-multi-v1` | 287M | mDeBERTa, multilingual — benchmarked; largest 2.5, no `large` exists |
| `fastino/GLiNER2.5-Decide` | 340M | decision-tuned sibling — benchmarked; best local cls on the mixed pool |
| `knowledgator/gliformer-base-v1` | ~190M | benchmarked; best NER F1 here |
| `knowledgator/gliformer-large-v1` | 575.6M | benchmarked; family is base+large only, no small |
| `convaiinnovations/laya` (+multilingual, typed-decisions) | 421M / 322M | English root benchmarked; subfolders exist for the other two |
| `jaredpalmer/kev-0.8b` | 0.8B (9.3M trained) | Qwen3.5 base + LoRA/head; 4B/9B siblings exist but are impractical on CPU — 0.8B benchmarked |
| `aimeigaoshou/agent-jev` | 0.6B | Qwen3 base, LM head swapped for a candidate head; single checkpoint — benchmarked |
| `Mapika/decider-0.8b` | 0.8B (752M) | Qwen3.5-0.8B-Base fine-tune, PyPI `decider-ai` System One server — benchmarked |
| `iapp/OpenThai-SystemOne` | 0.8B | Qwen3.5 Gated DeltaNet hybrid + 256-way slot head, Thai/English — benchmarked (weights gated on HF) |
| `heman10x/rlcd-modernbert-151m` | 151M | ModernBERT-base decision head with trained abstention ("Verdict") — benchmarked |
| `alibiserikbay/JevK5-Lite` | 437M | lite build of JevK5 (JevBench #3): label-head DeBERTa-v3-large encoder, `jevk5` package — benchmarked |
| `notnotsamuel/LFM2.5-350M-RLCD` | 350M | LFM2.5 backbone + RLCD training; vendored constrained-decoding engine (`engines/rlcd_engine/`) — benchmarked |
| `mixedbread-ai/mxbai-rerank-base-v2` | 494M | cross-encoder reranker, (instruction, label) pair scoring — benchmarked as a decision engine |
| `BAAI/bge-reranker-v2-m3` | 568M | XLM-RoBERTa-large cross-encoder, multilingual — benchmarked; best reranker here (72.9%) |
| `Alibaba-NLP/gte-rerank-modernbert-base` | 150M | ModernBERT-base cross-encoder — benchmarked |
| `altslate/certo-decision-model` | 421M | ModernBERT-large + calibrated per-option score head; vendored engine (`engines/certo_engine/`) — benchmarked |
| `MoLeMo-Lab/mojev` | 0.85B | Qwen3.5 + fla packed one-pass scorer, loads via trust_remote_code; adapted engine (`engines/mojev_engine/`) — benchmarked |
| `pngwn/nanodiff-350m-typed-decisions-lam1` | 350M | bidirectional diffusion LM (BY571/nanoDiff architecture); vendored engine (`engines/nanodiff_engine/`) — benchmarked |
| `akhilaaa3/Jev-Omni` | 12B | multimodal (text/image/audio/video) Gemma 4 fine-tune, own API — not run: needs a CUDA GPU and ~50 GB fp32 weights |
| `fastino/gliner2-{base,large,multi}-v1` | — | older span-architecture line, different loader — not run |
| `gliner-community/gliner_*-v2.5` | — | classic `gliner` package line — not run |

## Noted, not benchmarked

Systems we looked at but never ran on this CPU-only bench. Every number
in this section is **borrowed** (JevBench or the project's own report),
not measured here — the tables above only carry numbers from our pools.

Blocked by hardware or runtime:

| System | Why not run here |
|---|---|
| `akhilaaa3/Jev-Omni` | 12B multimodal Gemma 4 fine-tune; needs CUDA and ~50 GB fp32 weights |
| kev 4B / 9B (local) | `kev-0.8b`'s larger siblings; impractical on CPU (4B benched via OpenRouter) |
| `Mapika/decider-2b`, `decider-35b-a3b` | larger decider siblings of the benchmarked `decider-0.8b`; GPU-priced |
| `fastino/gliner2-{base,large,multi}-v1` | older span-architecture GLiNER 2 line, different loader |
| `gliner-community/gliner_*-v2.5` | classic `gliner` line (LUKE-descended); a one-off CPU trial of `gliner_large-v2.5` scored 56.2% classification / 67% NER at ~0.4 s/question — dominated by GLiNER2.5-base, so not added |
| [TypeLLM](https://github.com/TypeLLM/TypeLLM) | type-safe generation harness; documented Qwen3.8-27B setup uses a Linux SGLang GPU server. Self-reports 195/231 JevBench items (228/231 with thinking mode) |

Seen only on the JevBench v1.4.2 leaderboard (its composite is a harmonic
mean of Intelligence/Calibration/Speed/Cost — a different metric from the
accuracy-only scores in this README):

| System | JevBench v1.4.2 (borrowed) |
|---|---|
| decider-4b v2 | #1 at 64.13 — but Jev 1.13.0 (#2, 63.29) keeps the Intelligence (53.1 vs 49.4) and Calibration (76.3 vs 75.0) leads; decider wins Speed (92.9 vs 83.3) and Cost. We benchmarked `decider-0.8b` only |
| JevK5 v0.2.0 | #3 at 62.04 — its lite build (`alibiserikbay/JevK5-Lite`) is benchmarked above |
| Cygnet | #4 at 61.76 |
| Hopper | #5 at 59.43 |

### Trial-pool runs, not promoted

A sweep of the community "All about Jev" catalog (1,619 entries → 9
plausible candidates, four of them unrunnable or gated) tried each
runnable one on two fixed mini-pools from this repo's graded questions —
12 easy-tier and 16 hard-tier classification, plus NER where supported.
Measured here, but on trial pools — not the 48-question spectrum. Four
graduated to the full benchmark (GLiNER2.5-Decide, JevK5-Lite,
LFM2.5-RLCD 350M, nanodiff 350M); the rest:

| System | Trial result |
|---|---|
| `Quazim0t0/Byrne-Jev-79M` | 9/12 easy, 7/16 hard — dominated by the systems above |
| Dohnuts 0.8B (iACE, from-scratch) | unrunnable — its released runtime hardcodes CUDA (flash-linear-attention[rocm], `.to("cuda")`); no CPU path |
| `tasksource/modernbert-tasksource-jev` | unrunnable — its `modernjev` package is unpublished, source links 404, card says "preview, not ready to use" |
| `shreyanbr/system-one-gold` | unrunnable — requires a `systemone` engine package and calibration file that are not published |
| `idlabs/jev-typed-decisions-causal-0.6b` | gated on HF (401) |

## Repo layout

| File | What it is |
|---|---|
| `demos/demo.py` / `demos/demo_gliformer.py` / `demos/demo_glirel.py` / `demos/demo_laya.py` / `demos/demo_jev.py` | scripted tours, one per system, shared sample texts |
| `demos/jevk5_demo.py` | JevK5-Lite tour + one-shot runner (`--serve`) inside `.venv-von`, spawned by its web-UI tab |
| `demos/lfm_rlcd_demo.py` | LFM2.5-RLCD tour + one-shot runner (`--serve`) inside `.venv-von`, spawned by its web-UI tab |
| `demos/reranker_demo.py` | three cross-encoder rerankers as decision engines: (instruction, label) pair scores + argmax (main venv, sentence-transformers) |
| `demos/certo_demo.py` | Certo 421M calibrated `decide()` tour (vendored `engines/certo_engine/`, main venv) |
| `demos/mojev_demo.py` | MoJev 0.85B tour + one-shot runner (`--serve`) inside `.venv-von`, spawned by its web-UI tab |
| `demos/nanodiff_demo.py` | nanodiff 350M diffusion-LM tour + one-shot runner (`--serve`) in the main venv, spawned by its web-UI tab |
| `app.py` | Gradio web UI: live tab per family + benchmark + compare |
| `demos/von_demo.py` | one-shot von runner inside `.venv-von`, spawned by the von tab |
| `engines/von_client.py` | pinned, complete option-marker checkpoint loader shared by demo and benchmarks |
| `engines/jev_client.py` | dependency-free Python client for the TypeSafe System One API (also used against the local Kev server) |
| `engines/agentjev_client.py` | dependency-free client for the local AgentJev loopback API |
| `engines/openrouter_client.py` | dependency-free client for OpenRouter's `/systemone` and `/rerank` endpoints (hosted Kev 4B, Span-01, seven rerankers; `OPENROUTER_API_KEY` in repo-root `.env`) |
| `engines/{rlcd,certo,mojev,nanodiff}_engine/` | vendored inference packages for the local decision systems (attribution headers with source repo, revision and license inside each) |
| `bench.py` | flat-suite benchmark driver (classification + NER, 5× determinism) |
| `bench_spectrum.py` | the headline mixed-pool benchmark, one run per `--system` |
| `bench_graded.py` | question pools for the mixed-pool benchmark (source for `bench_spectrum.py`) |
| `bench_multilingual.py` | 9-language zero-shot suite, all 38 systems (Sinhala/Icelandic/Welsh in the rare tier) |
| `make_chart_images.py` | renders the benchmark charts to `docs/charts/*.png` for this README |
| `results/bench_*_results.json` | latest results, rendered by the web UI |

## License

MIT — see [LICENSE](LICENSE). Model licenses belong to their authors
(Apache 2.0 for the open model families; so1's library and the vendored
`engines/*_engine/` packages are MIT; the LiquidAI/LFM2.5-350M weights
behind LFM2.5-RLCD are under the LFM Open License v1.0); Jev access is
subject to TypeSafe AI's terms.
