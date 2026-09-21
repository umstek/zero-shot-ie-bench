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
| [von-1.0](https://huggingface.co/wfzyx/von-1.0) (`von-sdk`) | local typed-decision engine (System One protocol) | 396M | Apache 2.0, free |
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

Every case runs 5 times; accuracy below is from the first run, and the
determinism section shows whether repeats changed anything (spoiler: no).

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
engines have no span output; refreshed after correcting document identity):

| System | Precision | Recall | F1 | s/text |
|---|---|---|---|---|
| GLiNER2.5-small | 0.78 | 0.97 | 0.87 | 0.051±0.015 |
| GLiNER2.5-base | 0.94 | 1.00 | 0.97 | 0.111±0.007 |
| GLiNER2.5-multi | 0.97 | 1.00 | 0.98 | 0.127±0.037 |
| GLiFormer-base | 1.00 | 1.00 | **1.00** | 0.129±0.023 |
| GLiFormer-large | 0.97 | 1.00 | 0.98 | 0.447±0.066 |

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
(σ ≈ 0.002–0.07 s; largest for GLiFormer-large). Zero-shot outputs are
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

## Mixed-pool spectrum benchmark (all 15 systems)

`bench_spectrum.py` — the headline comparison. Every system answers the
same **one mixed pool** of 48 classification questions (sentiment + topic,
varying difficulty); the five extractors also answer 18 NER questions.
After all systems have run, each question's difficulty is **measured** as
the fraction of answering systems that got it wrong (continuous 0–1); the
web UI plots each system's accuracy along that spectrum, and per-question
predictions live in `bench_spectrum_results.json`.

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
| von-1.0 (option-marker) | 79.2% | 0.226 | n/a | |
| Kev 0.8B (local) | 72.9% | 0.587 | n/a | |
| GLiNER2.5-small | 70.8% | 0.043 | 33% | 0.053 |
| Laya typed-decisions | 70.8% | 1.243 | n/a | |
| Laya (local) | 68.8% | 0.143 | n/a | |
| GLiFormer-base | 66.7% | 0.139 | 56% | 0.145 |
| gliclass-edge | 66.7% | **0.016** | n/a | |
| gliclass-modern-base | 66.7% | 0.051 | n/a | |
| so1 + Qwen2.5-0.5B | 43.8% | 0.237 | n/a | |

Takeaways: the pool is deliberately mixed, so absolute numbers run lower
than on the flat suite above. **gliclass-large is the best local sarcasm
reader found** (the hardest sentiment questions; second only to cloud Jev)
and gliclass-edge is the speed king at 16 ms/question. GLiNER2.5-multi is
the most robust extractor. von's trained option-marker backend scores
79.2% here; its earlier 62.5% row is withdrawn: the PyPI SDK's default NLI
loader initialized an untrained classifier instead of loading the separate
decision weights. Kev 0.8B — Jared Palmer's open-weights reconstruction
of Jev's architecture (LoRA + pointer head over a frozen Qwen3.5-0.8B
base, served locally on the same System One contract) — lands at 72.9%,
the best score among the fully-local decision engines in this pool. The so1
technique works mechanically on any ChatML LLM, but a 0.5B base model is
not enough brain for it — the harness is the contribution, swap in a
bigger LLM. Note: gliclass PyPI metadata asks for transformers ≥5 but runs
fine on the pinned 4.57.6; von genuinely needs transformers 5, hence the
separate `.venv-von`. Model download and loading are excluded from the
latency measurements; the first forward pass is included.

The web UI renders these as interactive charts; the same charts, as
images (regenerate after re-running the benchmarks with
`python make_chart_images.py`):

**Classification (48-question mixed pool)**

![Classification accuracy, mixed pool](docs/charts/cls_accuracy.png)

![Speed vs accuracy — up and left is better](docs/charts/cls_tradeoff.png)

![Accuracy vs question difficulty](docs/charts/cls_spectrum.png)

![Mean latency per question](docs/charts/cls_latency.png)

**NER exact-match (18 questions, extractors only)**

![NER exact-match rate, mixed pool](docs/charts/ner_exact.png)

![Exact match vs question difficulty](docs/charts/ner_spectrum.png)

## Multilingual benchmark (9 languages, no English)

`bench_multilingual.py` — the same six sentence meanings per language
(2 positive / 2 negative / 2 neutral sentiment), labels in English.
Popular: Spanish, French, Chinese · Medium: Vietnamese, Turkish, Ukrainian ·
Rare: **Sinhala**, Icelandic, Welsh. Sentences were verified by blind
back-translation through an independent model instance (it caught 8 errors,
including a sentiment-flipping Sinhala word) and the full table is in the
PR for native-speaker review. All 15 systems answer the same 54 texts.

| System | Popular | Medium | Rare | All |
|---|---|---|---|---|
| Jev (cloud) | **100%** | **100%** | **100%** | **100%** |
| GLiNER2.5-multi (mDeBERTa) | 100% | 100% | 67% | 89% |
| gliclass-large | 100% | 89% | 56% | 81% |
| Kev 0.8B (local) | 94% | 89% | 50% | 78% |
| Laya Router (mmBERT) | 89% | 94% | 44% | 76% |
| gliclass-base | 94% | 67% | 39% | 67% |
| GLiFormer-large | 94% | 61% | 33% | 63% |
| Laya typed-decisions | 100% | 44% | 33% | 59% |
| GLiNER2.5-base | 89% | 50% | 28% | 56% |
| GLiFormer-base | 83% | 44% | 33% | 54% |
| von-1.0 (option-marker) | 72% | 33% | 39% | 48% |
| gliclass-modern-base | 44% | 33% | 39% | 39% |
| so1 + Qwen2.5-0.5B | 39% | 39% | 33% | 37% |
| GLiNER2.5-small | 56% | 22% | 28% | 35% |
| gliclass-edge | 50% | 22% | 22% | 31% |

![Accuracy by system and language](docs/charts/ml_heatmap.png)

![Overall multilingual accuracy](docs/charts/ml_overall.png)

Per-language highlights: GLiNER2.5-multi is perfect through Ukrainian but
drops on Welsh (67%) and Sinhala (33%); gliclass-large transfers
surprisingly well for an English-family release (100% on Chinese, and the
best local Sinhala score at 67%); Jev is the only system at 100% on
Sinhala. Kev 0.8B is the second-best local system at 78% (94/89/50 across
tiers) — the Qwen3.5 backbone carries far more multilingual pretraining
than any encoder here, though Welsh (33%) still trips it. Laya's
English-only `typed-decisions` specialist — its
strongest checkpoint on the vendor's own workflow benchmark — holds 100%
on the popular tier here but collapses on non-Latin scripts (33% rare,
59% overall; the Router-based row stays the family's best multilingual
entry). English-only encoders degrade with distance from English as
expected. von's complete option-marker checkpoint scores 48% overall,
including 50% on Sinhala. The old 33–48% runs used randomly initialized
classifier weights and are invalid; they do not demonstrate instability
of the trained model. The rerun uses pinned model and SDK revisions, saved
in the results file. Laya Router preloads and retains both selected
checkpoints before timing, so script changes do not include weight loading.

## Feature comparison

| Capability | GLiNER 2.5 | GLiFormer | GLiClass | Laya | von | so1 | Jev | Kev |
|---|---|---|---|---|---|---|---|---|
| Ability group | Extractor | Extractor | Classifier | Decision engine | Decision engine | Decision engine (BYO LLM) | Decision engine (cloud) | Decision engine (local, open weights) |
| Zero-shot NER spans | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Text classification | ✅ | ✅ | ✅ | ✅ choice | ✅ choice | ✅ choice | ✅ choice | ✅ choice |
| All labels scored in one pass | ✅ | ✅ | ✅ (its core design) | ✅ | ✅ | ✅ packed | ✅ one request | ✅ one request |
| Relations | ✅ + JointIE graph | ✅ joint head | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Span attributes (per-entity sentiment) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Structured records | ✅ flat, anchor-based | ✅ nested Pydantic | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Ordinal score rubrics | ❌ | ❌ | ❌ | ✅ score | ✅ rate | ✅ | ✅ score | ✅ score |
| Yes/no judgments | ❌ | ❌ | ❌ | ✅ noul | ✅ judge | ✅ yes_no | ✅ noul | ✅ noul |
| Text embeddings | ❌ | ✅ 1024-d | ❌ (reranker-capable) | ❌ | ❌ | ❌ | ❌ | ❌ |
| Multilingual | ✅ multi ckpt (89% over 9 langs here) | ❌ English (63%) | ✅ large 81% over 9 langs | ✅ Router, 100+ langs (76%) | option-marker: 48% over 9 langs | = base LLM's languages (37%) | ✅ 100% incl. Sinhala | ✅ 78% over 9 langs |
| Runs offline / data local | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| Cost | free | free | free | free | free | free | $0.042/1M input | free (CPU time) |
| License | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 | MIT (lib) | proprietary API | Apache 2.0 |
| Batch shape | per text | per text (batch_size) | per text, all labels | all questions, one pass | per text | one packed prompt | all questions, one request | all questions, one request |

## Setup

Python 3.10+ for the main environment; 3.12+ for von (tested on 3.13,
Windows, CPU-only).

```bash
uv venv .venv
uv pip install --python .venv -r requirements.txt --overrides overrides.txt
# so1 is not on PyPI (needed by the so1 tab + benchmarks):
uv pip install --python .venv "open-alternative-jev @ git+https://github.com/ikermoel/open-alternative-jev"
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
```

Use uv for this install: `overrides.txt` deliberately overrides GLiClass
0.1.20's transformers ≥5 metadata with the validated 4.57.6 version required
by GLiNER2. This is an explicit compatibility exception for that pinned
GLiClass release; plain pip dependency resolution cannot install this mix.
Altair is included explicitly for the benchmark charts.

Notes: `protobuf` and `sentencepiece` are included explicitly because
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
                                         # (15 systems; von runs the same
                                         # command under .venv-von/Scripts/python)
.venv/Scripts/python bench_multilingual.py --system <name>
                                         # 9 languages, all 15 systems →
                                         # bench_multilingual_results.json
.venv/Scripts/python app.py              # web UI at http://127.0.0.1:7860
```

The web UI has a live tab per family — GLiNER 2.5, GLiFormer, GLiClass
(all four v3.0 sizes), Laya (local), von, so1 and Jev (cloud) — plus three
benchmark tabs (**Classification benchmark**, **Extraction benchmark**,
**Multilingual benchmark**; tables and charts from the
`bench_*_results.json` files) and a **Compare** tab (feature matrix).
Benchmark charts are altair-based: sorted bars, a speed-accuracy scatter,
accuracy-vs-difficulty lines and a system × language heatmap.
von runs in its own venv (`.venv-von`) through `von_demo.py`. Its shared
loader (`von_client.py`) pins the upstream SDK and model revision and
requires the complete `option_marker.pt` state dict. Missing or incompatible
weights fail before inference; there is no random-head fallback. First use
downloads both the encoder and trained option-marker state (about 3 GB total).

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
| `knowledgator/gliformer-base-v1` | ~190M | benchmarked; best NER F1 here |
| `knowledgator/gliformer-large-v1` | 575.6M | benchmarked; family is base+large only, no small |
| `convaiinnovations/laya` (+multilingual, typed-decisions) | 421M / 322M | English root benchmarked; subfolders exist for the other two |
| `jaredpalmer/kev-0.8b` | 0.8B (9.3M trained) | Qwen3.5 base + LoRA/head; 4B/9B siblings exist but are impractical on CPU — 0.8B benchmarked |
| `fastino/gliner2-{base,large,multi}-v1` | — | older span-architecture line, different loader — not run |
| `gliner-community/gliner_*-v2.5` | — | classic `gliner` package line — not run |

## Repo layout

| File | What it is |
|---|---|
| `demo.py` / `demo_gliformer.py` / `demo_laya.py` / `demo_jev.py` | scripted tours, one per system, shared sample texts |
| `app.py` | Gradio web UI: live tab per family + benchmark + compare |
| `von_demo.py` | one-shot von runner inside `.venv-von`, spawned by the von tab |
| `von_client.py` | pinned, complete option-marker checkpoint loader shared by demo and benchmarks |
| `jev_client.py` | dependency-free Python client for the TypeSafe System One API |
| `bench.py` | flat-suite benchmark driver (classification + NER, 5× determinism) |
| `bench_spectrum.py` | the headline mixed-pool benchmark, one run per `--system` |
| `bench_graded.py` | question pools for the mixed-pool benchmark (source for `bench_spectrum.py`) |
| `bench_multilingual.py` | 9-language zero-shot suite, all 15 systems (Sinhala/Icelandic/Welsh in the rare tier) |
| `make_chart_images.py` | renders the benchmark charts to `docs/charts/*.png` for this README |
| `bench_*_results.json` | latest results, rendered by the web UI |

## License

MIT — see [LICENSE](LICENSE). Model licenses belong to their authors
(Apache 2.0 for the open model families; so1's library is MIT); Jev access
is subject to TypeSafe AI's terms.
