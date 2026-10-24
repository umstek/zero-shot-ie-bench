# gliner runner island

The GLiNER lineage as one independent uv project: GLiFormer, GLiREL,
GLiNER-relex, GLiClass and Laya on the validated transformers **4.57.6** CPU
stack (torch 2.14.0 CPU build). One FastAPI process serves the bench API and
would mount a Gradio UI when one is added; it speaks the shared runner
protocol from `runners/_protocol`.

## What it serves

| model_id (wire) | HF checkpoint | package | tracks | RAM (fp32 CPU) |
|---|---|---|---|---|
| `gliformer-base-v1.local` | knowledgator/gliformer-base-v1 (~190M) | gliformer | classify, ner | ~1 GB |
| `gliformer-large-v1.local` | knowledgator/gliformer-large-v1 (~576M) | gliformer | classify, ner | ~3 GB |
| `gliner-relex-multi-v1.0.local` | knowledgator/gliner-relex-multi-v1.0 (~319M) | gliner (classic) | ner, relation, classify* | ~2 GB |
| `glirel-large-v0.local` | jackboyla/glirel-large-v0 (~467M) | glirel | relation | ~3 GB |
| `gliclass-base-v3.0.local` | knowledgator/gliclass-base-v3.0 (187M) | gliclass | classify | ~2 GB |
| `gliclass-modern-base-v3.0.local` | knowledgator/gliclass-modern-base-v3.0 (151M) | gliclass | classify | ~1 GB |
| `gliclass-large-v3.0.local` | knowledgator/gliclass-large-v3.0 (439M) | gliclass | classify | ~3 GB |
| `laya.local` | convaiinnovations/laya (421M) | laya | decision | ~3 GB |
| `laya.router` | convaiinnovations/laya (+ typed-decisions, via `laya.Router`) | laya | decision | ~3 GB resident |
| `laya-typed-decisions.local` | convaiinnovations/laya-typed-decisions | laya | decision | ~3 GB |

`model_id`s are the catalog deployment ids (`runner: gliner` in
`catalog/models/{knowledgator,jackboyla,convaiinnovations}/*.yaml`); `version`s
are the catalog version ids (`...@unversioned`).

\* relex classification rides the vendor multitask prompt (the classic-gliner
classification surface ported from `legacy/engines/glinerx_client.py`) and is
a census row at best — the card claims NER + relations only. The GLiNER-X
checkpoints that mapping was measured on are culled and stay in their own
legacy venv (`.venv-glinerx`, `gliner[stanza]` wants transformers 5.x); they
are **not** served here. GLiNER2.5 (`gliner2`) is likewise not in this
island's catalog slice.

## Wire contract

Shared with every runner island (see `runners/_protocol` and
`catalog/schema/protocol.json` + `track-request.json` + `track-response.json`):

- `GET /healthz`
- `GET /info` — runner name, protocol, the model table above, host fingerprint
- `POST /v1/<track>` for `track` in `ner | classify | decision` —
  `{"model_id", "items": [{"item_id", "input", "question", "params"}]}`.
  `compute_ms` is measured inside the server. One item failing never kills
  the request: it comes back as that item's `error`.

`predict(model_id, payload)` is called once per item with the item's
`{"input", "question", "params"}`. The track is not echoed to it, so it is
inferred from the question shape (the same shapes v0 benches used):

| question shape | track | models |
|---|---|---|
| `{"type": "choice"\|"score"\|"noul", "instructions", "criteria"}` | decision | laya* |
| `{"relations": [...], "entity_labels": [...]}` | relation | relex, glirel |
| `{"labels": {label: description}}` | classify | gliformer, gliclass, relex* |
| `{"labels": [str, ...]}` | ner | gliformer, relex |

Laya's question dicts are the Jev/Laya typed-question shape
(`engines/jev_client.choice/score/noul`): a choice answers its label string
with the answer's `confidence` as `prob`; score/noul answer their number.
NER answers are sorted, deduped `(start, end, label)` char-span tuples.

**Known gap:** GLiREL is relation-only, but the tracks list is currently
`["ner", "classify", "decision"]`, so no `/v1/relation` route exists yet.
GLiREL and relex relations are reachable through the `predict` dispatch (and
tested directly); adding `"relation"` to `TRACKS` in `src/gliner/server.py`
is the one-line change once the protocol's route set grows.

GLiREL needs entity spans as input (its catalog note: "needs entity spans as
input"). Its question may carry them directly — `"ner"` in GLiREL token
format (`[start_token, end_token_inclusive, type, text]`) or `"entities"` as
char spans — otherwise the island lazily loads the relex checkpoint as the
span producer (legacy used GLiNER 2.5 spans, which this island does not
serve; pass entities explicitly for parity).

## Start

```bash
cd runners/gliner
uv sync                                  # one-time (torch CPU ~200 MB; no weights)
uv run uvicorn gliner_runner.server:app --port 8009
# or: uv run python -m gliner_runner.server     (same app, prints the host fingerprint first)
```

Weights download from the HF cache on first use per model (never at import,
never in tests): set `HF_HOME` to a persistent location, e.g.

```bash
HF_HOME=D:/hf-cache uv run uvicorn gliner_runner.server:app --port 8009
```

Docker (build from the repo root; `_protocol` comes in as a named build
context because the island's `../_protocol` path dependency is outside the
island directory):

```bash
docker build -f runners/gliner/Dockerfile \
    --build-context protocol=runners/_protocol \
    -t gliner-runner runners/gliner
docker run -p 8009:8009 -v hf-cache:/root/.cache/huggingface gliner-runner
```

## Hardware

CPU-only — every checkpoint here was benched on fp32 CPU (catalog
deployments: `device: cpu`, `accel: cpu`, no VRAM). Any laptop core works;
the mixed 48-question pools ran in seconds per model on this machine's iGPU
class CPU. Budget RAM per the table above (one model resident at a time; the
server lazy-loads per model and caches). No GPU, no quantization.

## Tests and checks

```bash
uv run pytest            # offline tests always run; weight tests skip unless
                         # the checkpoint is already in the local HF cache —
                         # CI never downloads weights
uv run pyrefly check .   # type check
```

The heavy per-family tests (real load + one tiny inference) are guarded by
`tests/hf_weights.py:weights_cached`, which inspects the HF hub cache the
same way `huggingface_hub` resolves it (`HF_HUB_CACHE`/`HF_HOME`/default).
They are skipped silently on any machine without the weights.

## Provenance (legacy -> island)

| legacy | here |
|---|---|
| `bench_spectrum.py` classify_extractor (gliformer branch) | `models/gliformer.py` |
| `bench_spectrum.py` classify_gliclass | `models/gliclass.py` |
| `bench_spectrum.py`/`bench_multilingual.py` laya branches + `demos/demo_laya.py` | `models/laya.py` |
| `app.py` GLiNER-relex tab (+ `demos/demo_gliner_relex.py`) | `models/gliner_relex.py` |
| `app.py` GLiREL tab (+ `demos/demo_glirel.py`) | `models/glirel.py` |
| `engines/glinerx_client.py` vendor multitask mapping (prompt + reduction, thresholds) | `models/gliner_relex.py` |
| `demos/demo_gliformer.py` build_sample_pdf + `tests/test_gliformer_pdf_stop.py` | `models/gliformer.py` + `tests/test_gliformer_pdf_stop.py` |
| `tests/test_glinerx_client.py` mapping tests | `tests/test_glinerx_mapping.py` |

The GLiNER-X registry/warm-up/load-guard tests stay with the legacy engine:
that family is culled and runs in its own venv, so those fixtures have no
island counterpart.
