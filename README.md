# zero-shot-ie-bench

Four zero-shot information-extraction and classification systems — two local
extractor encoders and two typed-decision engines (one local, one cloud) —
demoed, benchmarked, and cross-compared in one repo with a web UI.

| System | Kind | Size | License / cost |
|---|---|---|---|
| [GLiNER 2.5](https://github.com/fastino-ai/GLiNER2) (`fastino/gliner2.5-*`) | local extractor encoder (boundary arch) | 74M / 194M / 287M | Apache 2.0, free |
| [GLiFormer](https://github.com/Knowledgator/GLiFormer) (`knowledgator/gliformer-*`) | local extractor encoder (layout-aware DeBERTa) | ~190M / 575.6M | Apache 2.0, free |
| [Laya](https://huggingface.co/convaiinnovations/laya) (`laya`) | local typed-decision engine (choice/score/noul) | 421M (322M multilingual) | Apache 2.0, free |
| [Jev](https://www.typesafe.ai/) (`jev-latest` via System One API) | cloud typed-decision engine (choice/score/noul) | closed | paid API |

The GLiNER lineage forked: Urchade Zaratiana (original GLiNER author,
ex-Knowledgator) is on the GLiNER2 paper with Fastino; Knowledgator kept the
classic `gliner` package and built GLiFormer on it. Laya's card positions it
explicitly as the open local counterpart of cloud Jev. Two different animals:

- **Extractors** (GLiNER 2.5, GLiFormer): spans, entities, relations,
  records — per-text calls, run offline.
- **Decision engines** (Laya, Jev): you ask typed questions (`choice`,
  `score`, `noul`) over a JSON state; all questions in one call are answered
  together (one forward pass / one request).

## Benchmark results

Measured on CPU, zero-shot, identical label sets and descriptions,
out-of-the-box defaults. Full detail and per-item misses in
[`bench_results.json`](bench_results.json); methodology notes in `bench.py`.

Classification accuracy:

| System | Sentiment (24) | Topic (12) | s/text |
|---|---|---|---|
| GLiNER2.5-small | 83.3% | 100% | 0.05 |
| GLiNER2.5-base | 100% | 100% | 0.13 |
| GLiNER2.5-multi | 100% | 100% | 0.13 |
| GLiFormer-large | 100% | 83.3% | 0.42 |
| Laya (local) | 95.8% | 83.3% | 0.37–0.42 (batched ÷ n) |
| Jev (cloud) | 100% | 100% | 0.04–0.07 (batched ÷ n) |

NER, strict span+label match (decision engines have no span output):

| System | Precision | Recall | F1 | s/text |
|---|---|---|---|---|
| GLiNER2.5-small | 0.77 | 0.96 | 0.86 | 0.07 |
| GLiNER2.5-base | 0.93 | 1.00 | 0.97 | 0.17 |
| GLiNER2.5-multi | 0.97 | 1.00 | 0.98 | 0.15 |
| GLiFormer-large | 0.97 | 1.00 | 0.98 | 0.45 |

Honest caveats:

- The sentiment set is easy (everything ≥83%); misses for the small GLiNER
  model are neutrals drifting to pos/neg; GLiFormer's topic misses lean
  "business".
- Jev ran as one batched request per task (0.8 s for 24 texts); Laya as one
  batched forward pass per task — per-text latency is batch ÷ n for both.
- **Laya prompt-format gotcha found here:** dict-shaped `instructions`
  (which Jev handles fine) collapse Laya onto one label (58.3% sentiment);
  with string instructions it scores 95.8%. Benchmarked with strings.
- Laya's own card is candid: base checkpoints are classification-shaped,
  near-chance zero-shot on nuanced decision workflows, probabilities ship
  over-confident before temperature fitting, and >20-option `choice`
  questions are weak without raising `head_max_len`.

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
.venv/Scripts/python demo_laya.py        # Laya tour + multilingual Router
.venv/Scripts/python demo_jev.py         # Jev tour (2 paid API requests)

.venv/Scripts/python bench.py            # full benchmark → bench_results.json
.venv/Scripts/python app.py              # web UI at http://127.0.0.1:7860
```

The web UI has one live tab per system, a **Benchmark** tab (tables + chart
from `bench_results.json`) and a **Compare** tab (feature matrix).

## GLiNER 2.5 family map

| Checkpoint | Params | Notes |
|---|---|---|
| `fastino/gliner2.5-small-v1` | 74M | DeBERTa-v3-xsmall, fast CPU |
| `fastino/gliner2.5-base-v1` | 194M | default English, benchmarked |
| `fastino/gliner2.5-multi-v1` | 287M | mDeBERTa, multilingual — largest 2.5; no `large` exists |
| `fastino/gliner2-{base,large,multi}-v1` | — | older span-architecture line, different loader |
| `gliner-community/gliner_*-v2.5` | — | classic `gliner` package line |

## Repo layout

| File | What it is |
|---|---|
| `demo.py` / `demo_gliformer.py` / `demo_laya.py` / `demo_jev.py` | scripted tours, one per system, shared sample texts |
| `jev_client.py` | dependency-free Python client for the TypeSafe System One API |
| `bench.py` | benchmark driver (classification × 6 systems, NER × 4) |
| `bench_results.json` | latest results, rendered by the web UI |
| `app.py` | Gradio web UI (live demos + benchmark + compare) |

## License

MIT — see [LICENSE](LICENSE). Model licenses belong to their authors
(Apache 2.0 for the three open families); Jev access is subject to TypeSafe
AI's terms.
