# zero-shot-ie-bench

Seven zero-shot information-extraction and classification families —
extractor encoders, a purpose-built classifier, and typed-decision engines
(local and cloud) — demoed, benchmarked, and cross-compared in one repo
with a web UI.

| System | Kind | Size | License / cost |
|---|---|---|---|
| [GLiNER 2.5](https://github.com/fastino-ai/GLiNER2) (`fastino/gliner2.5-*`) | local extractor encoder (boundary arch) | 74M / 194M / 287M | Apache 2.0, free |
| [GLiFormer](https://github.com/Knowledgator/GLiFormer) (`knowledgator/gliformer-*`) | local extractor encoder (layout-aware DeBERTa) | ~190M / 575.6M | Apache 2.0, free |
| [GLiClass](https://github.com/knowledgator/gliclass) (`knowledgator/gliclass-*-v3.0`) | local zero-shot classifier (all labels, one pass) | 33M / 151M / 187M / 439M | Apache 2.0, free |
| [Laya](https://huggingface.co/convaiinnovations/laya) (`laya`) | local typed-decision engine (choice/score/noul) | 421M (322M multilingual) | Apache 2.0, free |
| [von-1.0](https://huggingface.co/wfzyx/von-1.0) (`von-sdk`) | local typed-decision engine (System One protocol) | 395M | Apache 2.0, free |
| [open-alternative-jev](https://github.com/ikermoel/open-alternative-jev) (`so1`) | local decision harness over any ChatML LLM (logprobs) | BYO LLM (tested Qwen2.5-0.5B) | MIT, free |
| [Jev](https://www.typesafe.ai/) (`jev-latest` via System One API) | cloud typed-decision engine (choice/score/noul) | closed | paid API |

The GLiNER lineage forked: Urchade Zaratiana (original GLiNER author,
ex-Knowledgator) is on the GLiNER2 paper with Fastino; Knowledgator kept the
classic `gliner` package, built GLiFormer on it and also maintains GLiClass.
Laya's card positions it explicitly as the open local counterpart of cloud
Jev; von speaks the same System One protocol locally, and so1 is a library
that turns any open LLM into a Jev-style decider. Three different animals:

- **Extractors** (GLiNER 2.5, GLiFormer): spans, entities, relations,
  records — per-text calls, run offline.
- **Classifiers** (GLiClass): zero-shot text→label scores with every label
  answered in one forward pass.
- **Decision engines** (Laya, von, so1, Jev): you ask typed questions
  (`choice`, `score`, `noul`) over a JSON state; all questions in one call
  are answered together (one forward pass / one request).

## Benchmark results

Measured on CPU, zero-shot, identical label sets and descriptions,
out-of-the-box defaults. Full detail and per-item misses in
[`bench_results.json`](bench_results.json); methodology notes in `bench.py`.

Classification accuracy:

Every case runs 5 times; accuracy below is from the first run, and the
determinism section shows whether repeats changed anything (spoiler: no).

Classification accuracy (mean latency ± std per text):

| System | Sentiment (24) | Topic (12) | s/text |
|---|---|---|---|
| GLiNER2.5-small | 83.3% | 100% | 0.046±0.010 |
| GLiNER2.5-base | 100% | 100% | 0.103±0.008 |
| GLiNER2.5-multi | 100% | 100% | 0.121±0.014 |
| GLiFormer-base | 100% | 75.0% | 0.118±0.013 |
| GLiFormer-large | 100% | 83.3% | 0.400±0.035 |
| Laya (local) | 95.8% | 83.3% | 0.164±0.009 (batched ÷ n) |
| Jev (cloud) | 100% | 100% | 0.052±0.002 (batched ÷ n) |

NER, strict span+label match (decision engines have no span output):

| System | Precision | Recall | F1 | s/text |
|---|---|---|---|---|
| GLiNER2.5-small | 0.77 | 0.96 | 0.86 | 0.058±0.011 |
| GLiNER2.5-base | 0.93 | 1.00 | 0.97 | 0.117±0.007 |
| GLiNER2.5-multi | 0.97 | 1.00 | 0.98 | 0.131±0.010 |
| GLiFormer-base | 1.00 | 1.00 | **1.00** | 0.134±0.011 |
| GLiFormer-large | 0.97 | 1.00 | 0.98 | 0.428±0.050 |

### Determinism (5 runs per case)

Output stability — share of cases whose prediction was identical across all
5 runs (label for classification, full span set for NER):

| System | Sentiment | Topic | NER span sets |
|---|---|---|---|
| GLiNER2.5 small/base/multi | 100% | 100% | 100% |
| GLiFormer base/large | 100% | 100% | 100% |
| Laya (local) | 100% | 100% | n/a |
| Jev (cloud) | 100% | 100% | n/a |

**Every system was fully deterministic** — identical predictions on every
repeat, including the cloud API. The only measured variance is latency
(σ ≈ 0.002–0.05 s; largest for GLiFormer-large). Zero-shot outputs are
therefore reproducible run-to-run on identical inputs; what varies between
machines is speed, not answers.

Honest caveats:

- The sentiment set is easy (everything ≥83%); misses for the small GLiNER
  model are neutrals drifting to pos/neg; both GLiFormer topic misses lean
  "business" (base misses one more than large — but beats large on NER).
- Jev ran as one batched request per task per repeat; Laya as one batched
  forward pass per task per repeat — per-text latency is batch ÷ n for both.
- **Laya prompt-format gotcha found here:** dict-shaped `instructions`
  (which Jev handles fine) collapse Laya onto one label (58.3% sentiment);
  with string instructions it scores 95.8%. Benchmarked with strings.
- Laya's own card is candid: base checkpoints are classification-shaped,
  near-chance zero-shot on nuanced decision workflows, probabilities ship
  over-confident before temperature fitting, and >20-option `choice`
  questions are weak without raising `head_max_len`.
- GLiFormer demo quirks: the large checkpoint missed the second purchase in
  flat-record mode while base caught both; base returned empty results in
  the combined multi-task call where large succeeded; base embeddings are
  768-d vs large 1024-d.

## Mixed-pool spectrum benchmark (all 13 systems)

`bench_spectrum.py` — the headline comparison. Every system answers the
same **one mixed pool** of 48 classification questions (sentiment + topic,
varying difficulty) and the five extractors also answer 18 NER questions.
No difficulty tiers and no separate per-tier scores: after all systems
have run, each question's difficulty is **measured** as the fraction of
answering systems that got it wrong (continuous 0–1). The web UI plots
accuracy along that spectrum; per-question predictions live in
`bench_spectrum_results.json`.

Classification accuracy on the mixed pool (48 questions), NER scored as
exact span-set match (18 questions):

| System | Classification | s/question | NER exact | s/question |
|---|---|---|---|---|
| Jev (cloud) | **93.8%** | 0.037 | n/a | |
| GLiNER2.5-base | 83.3% | 0.111 | 61% | 0.135 |
| gliclass-base | 81.2% | 0.110 | n/a | |
| gliclass-large | 81.2% | 0.411 | n/a | |
| GLiNER2.5-multi | 79.2% | 0.152 | **67%** | 0.180 |
| GLiFormer-large | 79.2% | 0.412 | 61% | 0.411 |
| GLiNER2.5-small | 70.8% | 0.043 | 33% | 0.053 |
| Laya (local) | 68.8% | 0.143 | n/a | |
| GLiFormer-base | 66.7% | 0.139 | 56% | 0.145 |
| gliclass-edge | 66.7% | **0.016** | n/a | |
| gliclass-modern-base | 66.7% | 0.051 | n/a | |
| von-1.0 | 62.5% | 0.684 | n/a | |
| so1 + Qwen2.5-0.5B | 43.8% | 0.237 | n/a | |

Takeaways: the pool is deliberately mixed, so absolute numbers are lower
than on the easy flat suite above — that is the point. **gliclass-large is
the best local sarcasm reader found** (the hardest sentiment questions;
second only to cloud Jev) and gliclass-edge is the speed king at 16
ms/question. GLiNER2.5-multi is the most robust extractor. von-1.0's
93.5% card benchmark is on its own adversarial decision suite; on general
sentiment/topic it underperforms — its strength is its trained domain,
not open-ended classification. The so1 technique works mechanically on
any ChatML LLM, but a 0.5B base model is not enough brain for it — the
harness is the contribution, swap in a bigger LLM. Note: gliclass PyPI
metadata asks for transformers ≥5 but runs fine on the pinned 4.57.6; von
genuinely needs transformers 5, hence the separate `.venv-von`.

## Graded benchmark (easy / medium / hard)

`bench_graded.py` — exam-style tiers across subjects: sentiment and topic
(8 texts per tier each) plus NER (6 texts per tier). Easy = one strong
signal; medium = mixed signals and cross-domain vocabulary; hard = sarcasm,
negation flips, lowercase brands, and context-dependent ambiguity (Apple the
company vs apple the fruit) — gold labels stay unambiguous to a careful
human. Headline results (accuracy, single run; determinism established in
the main benchmark):

| System | Sentiment E/M/H | Topic E/M/H | NER F1 E/M/H |
|---|---|---|---|
| GLiNER2.5-small | 88 / 63 / 50% | 100 / 63 / 63% | 0.93 / 0.73 / 0.84 |
| GLiNER2.5-base | 100 / 88 / 75% | 100 / 63 / 75% | 1.00 / 0.90 / 0.87 |
| GLiNER2.5-multi | 100 / 75 / 63% | 100 / 63 / 75% | 1.00 / 0.90 / 0.94 |
| GLiFormer-base | 100 / 75 / 38% | 75 / 50 / 63% | 1.00 / 0.85 / 0.91 |
| GLiFormer-large | 100 / 100 / 38% | 100 / 63 / 75% | 1.00 / 0.87 / 0.91 |
| Laya (local) | 100 / 63 / 63% | 88 / 75 / 25% | n/a |
| Jev (cloud) | 100 / 88 / **100%** | 100 / 88 / **88%** | n/a |

Takeaways: easy tier saturates for everyone; the hard tier is where the
cloud LLM-style model pulls away (Jev reads sarcasm at 100% where both
GLiFormers drop to 38%); lowercase brands are the hardest NER condition
(medium tier); GLiNER2.5-multi is the most robust hard-tier NER (0.94).

## Multilingual benchmark (9 languages, no English)

`bench_multilingual.py` — the same six sentence meanings per language
(2 positive / 2 negative / 2 neutral sentiment), labels in English.
Popular: Spanish, French, Chinese · Medium: Vietnamese, Turkish, Ukrainian ·
Rare: **Sinhala**, Icelandic, Welsh. Sentences were verified by blind
back-translation through an independent model instance (it caught 8 errors,
including a sentiment-flipping Sinhala word) and the full table is in the
PR for native-speaker review.

| System | Popular | Medium | Rare |
|---|---|---|---|
| GLiNER2.5-multi (mDeBERTa) | 100% | 100% | 66.7% |
| GLiFormer-large (English-only control) | 94.4% | 61.1% | 33.3% |
| Laya Router (mmBERT) | 88.9% | 94.4% | 44.4% |
| Jev (cloud) | **100%** | **100%** | **100%** |

Per-language highlights: GLiNER2.5-multi is perfect through Turkish and
Ukrainian, then drops on Welsh (67%) and Sinhala (33%) while keeping
Icelandic (100%). Jev is the only system that handles Sinhala at 100%. The
English-only control degrades exactly as expected with distance from
English.

## Feature comparison

| Capability | GLiNER 2.5 | GLiFormer | Laya | Jev |
|---|---|---|---|---|
| Zero-shot NER, custom labels | ✅ | ✅ | ❌ | ❌ |
| Text classification | ✅ | ✅ | ✅ choice | ✅ choice |
| Relations | ✅ + JointIE graph | ✅ joint head | ❌ | ❌ |
| Span attributes (per-entity sentiment) | ✅ | ❌ | ❌ | ❌ |
| Structured records | ✅ flat | ✅ nested Pydantic | ❌ | ❌ |
| Ordinal scoring rubrics | ❌ | ❌ | ✅ score | ✅ score |
| Yes/no judgments | ❌ | ❌ | ✅ noul | ✅ noul |
| Text embeddings | ❌ | ✅ 1024-d | ❌ | ❌ |
| Multilingual | ✅ multi ckpt | ❌ English evals | ✅ Router, 100+ langs | model-dependent |
| Offline / data stays local | ✅ | ✅ | ✅ | ❌ |
| Cost | free | free | free | paid per token |
| Batch shape | per text | per text (batch_size) | all questions, one pass | all questions, one request |

## Setup

Python 3.10+ (tested on 3.13, Windows, CPU-only).

```bash
uv venv .venv
uv pip install --python .venv -r requirements.txt
# or: python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
```

Notes: `protobuf` and `sentencepiece` are pinned explicitly because
`gliner2[local]` does not pull them in and the DeBERTa tokenizer needs them.
Model checkpoints (~0.3–2.3 GB each) download from Hugging Face on first
run. For Jev only: set `TYPESAFE_API_KEY` in the environment or a `.env`
file in the repo root (gitignored) — see `jev_client.py`; Jev is a paid API.

## Run

```bash
.venv/Scripts/python demo.py             # GLiNER 2.5 tour: entities,
                                         # classification, relations, JointIE,
                                         # span attributes, records
.venv/Scripts/python demo_gliformer.py   # GLiFormer tour + embeddings
                                          # (--model base or large)
.venv/Scripts/python demo_laya.py        # Laya tour + multilingual Router
.venv/Scripts/python demo_jev.py         # Jev tour (2 paid API requests)

.venv/Scripts/python bench.py            # flat suite, 5 runs per case
                                         # (--repeats N) → bench_results.json
.venv/Scripts/python bench_spectrum.py --system <name>   # one mixed pool per
                                         # system → bench_spectrum_results.json
                                         # (13 systems; von runs the same
                                         # command under .venv-von/Scripts/python)
.venv/Scripts/python bench_graded.py     # tiered suite →
                                         # bench_graded_results.json
.venv/Scripts/python bench_multilingual.py  # 9 languages, no English →
                                         # bench_multilingual_results.json
.venv/Scripts/python app.py              # web UI at http://127.0.0.1:7860
```

The web UI has a live tab per family — GLiNER 2.5, GLiFormer, GLiClass
(all four v3.0 sizes), Laya, von, so1 and Jev — plus **Classification
benchmark** and **Extraction benchmark** tabs (mixed-pool scores, latency
and accuracy-along-the-difficulty-spectrum charts from
`bench_spectrum_results.json`) and a **Compare** tab (feature matrix).
von runs in its own venv (`.venv-von`) through `von_demo.py`.

## Family maps

Complete version coverage per family (all sizes that exist are benchmarked):

| Checkpoint | Params | Notes |
|---|---|---|
| `fastino/gliner2.5-small-v1` | 74M | DeBERTa-v3-xsmall, fast CPU — benchmarked |
| `fastino/gliner2.5-base-v1` | 194M | default English — benchmarked |
| `fastino/gliner2.5-multi-v1` | 287M | mDeBERTa, multilingual — benchmarked; largest 2.5, no `large` exists |
| `knowledgator/gliformer-base-v1` | ~190M | benchmarked; best NER F1 here |
| `knowledgator/gliformer-large-v1` | 575.6M | benchmarked; family is base+large only, no small |
| `convaiinnovations/laya` (+multilingual, typed-decisions) | 421M / 322M | English root benchmarked; subfolders exist for the other two |
| `fastino/gliner2-{base,large,multi}-v1` | — | older span-architecture line, different loader — not run |
| `gliner-community/gliner_*-v2.5` | — | classic `gliner` package line — not run |

## Repo layout

| File | What it is |
|---|---|
| `demo.py` / `demo_gliformer.py` / `demo_laya.py` / `demo_jev.py` | scripted tours, one per system, shared sample texts |
| `app.py` | Gradio web UI: live tab per family + benchmark + compare |
| `von_demo.py` | one-shot von runner inside `.venv-von`, spawned by the von tab |
| `jev_client.py` | dependency-free Python client for the TypeSafe System One API |
| `bench.py` | flat-suite benchmark driver (classification + NER, 5× determinism) |
| `bench_spectrum.py` | the headline mixed-pool benchmark, one run per `--system` |
| `bench_graded.py` | tiered suite; also the question pool `bench_spectrum.py` draws from |
| `bench_multilingual.py` | 9-language zero-shot suite (Sinhala/Icelandic/Welsh in the rare tier) |
| `bench_newcomers.py` | legacy: gliclass/von/so1 on the tiered suite (superseded by `bench_spectrum.py`) |
| `bench_*_results.json` | latest results, rendered by the web UI |

## License

MIT — see [LICENSE](LICENSE). Model licenses belong to their authors
(Apache 2.0 for the open model families; so1's library is MIT); Jev access
is subject to TypeSafe AI's terms.
