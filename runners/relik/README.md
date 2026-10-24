# relik runner island

Serves **relik-ie ReLiK** — the retriever-reader relation-extraction
pipeline (SapienzaNLP, ACL 2024) and its joint NER+relation sibling —
behind the shared runner protocol, on its own pins: **relik 1.0.7,
torch 2.3.1, faiss-cpu, Python 3.11, CPU**. This is the repo's only
Python 3.11 island: relik 1.0.7 pins torch==2.3.1, which has no 3.13
wheels (every other island is 3.13 on a newer torch line).

An E5-small retriever fetches the most similar relation definitions from a
fixed index and a DeBERTa-v3 reader scores span pairs against only those
candidates. The vocabulary is therefore **closed** — Wikidata properties
(618 in the small checkpoint) / the 10 Wikipedia entity types of the
sibling — you cannot pass free-text labels like the GLiNER lineage;
every prediction maps onto a real vocabulary entry. ~180M params, ~700 MB
on disk for the small checkpoint.

| wire model_id | checkpoint | output |
|---|---|---|
| `relik-ie/relik` | relik-ie/relik-relation-extraction-small (the benched catalog deployment) | triplets over Wikidata properties + untyped `--NME--` spans |
| `relik-ie/relik-relation-extraction-ner-small` | the joint NER+relation sibling (demoed only, no catalog deployment) | triplets + spans typed from the 10-type Wikipedia set |

## Pins (why this island exists)

- **relik[faiss]==1.0.7** (legacy/requirements-relik.txt, verbatim) — the
  release the csv shim and the two load-time fixes are tuned to. It pins
  torch==2.3.1 and transformers[sentencepiece]>=4.41,<4.42 itself.
- **numpy<2** — the legacy comment "numpy<2 comes with that torch" was a
  transitive resolution in 2024; faiss-cpu 1.8.0 and torch 2.3.1 both
  predate numpy 2 and are ABI-incompatible with it, so the bound is now
  declared explicitly.
- **setuptools<81** — relik's import chain needs `pkg_resources`
  (lightning/fabric imports it at module load); the legacy pip-built venv
  had setuptools implicitly, uv venvs do not — and the setuptools 81+
  line no longer ships pkg_resources at all.
- **en_core_web_sm 3.7.1** (release wheel, not on PyPI) — relik's
  annotator segments text with it and AUTO-INSTALLS it via pip at load
  time when missing, which uv venvs (no pip) cannot; without it the
  first predict 500s. The legacy .venv-relik carried the same model.
- **checkpoint** relik-ie/relik-relation-extraction-small (HF card
  Apache 2.0; the relik repo has no LICENSE file and its README footer
  says CC BY-NC-SA 4.0 — recorded as-is in the catalog).

## The Windows csv shim + load fixes

`import relik` crashes on Windows: relik/retriever/indexers/document.py
runs `csv.field_size_limit(sys.maxsize)` at module import and Windows'
C long is 32-bit (SapienzaNLP/relik#39, also #14/#15). Importing
`relik_runner.relik` installs the shim (clamps to INT32_MAX, no-arg
queries stay queries) **before** relik is ever imported — do not patch
site-packages.

Inference is pinned to `num_workers=0` (relik's gradio frontend does the
same): the retriever's DataLoader defaults to 4 worker **processes**, and
a Windows spawn child re-imports relik before anything can install the
shim (the legacy one-shot demo only survived because
`import engines.relik_client` sat at the top of `__main__`, which spawn
children re-execute — a server or pytest process has no such `__main__`).
On CPU the in-process batches are the right trade regardless.

The joint NER+relation sibling additionally cannot load as published on
relik 1.0.7; `relik_runner.relik` applies two process-local fixes at load
time (ported verbatim from legacy engines/relik_client.py):

1. `GoldenSillyRetriever` subclasses `nn.Module` but never calls
   `super().__init__()` — completing it makes the loader's `.eval()` work.
2. The span document index is built from a plain list of entity-type
   strings (delivered as an OmegaConf ListConfig) and the loader injects
   `use_faiss`/`precision` kwargs the index signature rejects — the
   wrapper normalizes the documents and retries once without the junk
   kwargs.

## Layout

- `src/relik_runner/relik.py` — the engine: csv shim, load fixes, lazy
  per-model loading, and the protocol `predict` (ported from
  `legacy/engines/relik_client.py` + the row mappings of
  `legacy/demos/relik_demo.py`)
- `src/relik_runner/server.py` — `create_app` wiring, `python -m
  relik_runner.server` entry, default port 8031
- `tests/` — offline fake-relik tests (the sys.modules patching legacy
  tests used), the runner-protocol TestClient suite, and the cache-gated
  slow test over the real checkpoints

## Hardware

CPU-only, ~3 GB RAM (`requires.ram_gb: 3` in the catalog, estimated).
No VRAM needed. First use downloads ~700 MB (relation checkpoint) /
~586 MB (sibling reader) into the local HF cache; afterwards it runs
offline.

## Run

```sh
uv sync
uv run python -m relik_runner.server --host 127.0.0.1 --port 8031
```

(or `uv run uvicorn relik_runner.server:app --port 8031`).

- `GET /healthz` — liveness.
- `GET /info` — models, runtime, host fingerprint.
- `POST /v1/relation` and `POST /v1/ner` — items `{item_id, input,
  question, params}`, `input` a text string. The prediction is a list of
  rows: relation → `{subject, relation, object, confidence}` triplets,
  NER → `{start, end, label, text}` char-offset spans. `prob` is null —
  per-row confidences ride inside the rows. The protocol does not echo
  the route to `predict`, so the track comes from the question shape: an
  explicit `"track": "ner" | "relation"` on the question/params wins;
  `relations`/`relation_labels`/`triplets` keys ask for relation;
  `entities`/`labels`/`spans` ask for NER (relik's vocabulary is closed —
  labels are never shown to the model); an empty question falls back to
  the checkpoint's primary (relation for the benched deployment, NER for
  the sibling). Params: `top_k` (relation candidates the retriever hands
  the reader, the legacy demo's `--top-k`). Weights load lazily on the
  first predict (~40 s warm cache, minutes on first download), so startup
  is instant and the load never happens at import.

## Tests

```sh
uv run pytest
```

Offline by construction: the engine is exercised against a fake `relik`
package that replicates the two 1.0.7 defects (the same sys.modules
patching legacy tests/test_relik_loader_fixes.py used), and the FastAPI
contract is exercised with a stubbed pipeline over `TestClient`. The
tests that load the real checkpoints skip unless the snapshot is already
in the local HF cache (`cached_snapshot()`, `local_files_only=True`; CI
never downloads weights). To seed the cache locally:

```sh
uv run python -c "from relik_runner.relik import load_relik; load_relik()"
```

Machine note (this desktop): the data disk filled up while installing the
torch stack. If `uv sync` or pytest fails with "not enough space on the
disk", put this island's venv on the system drive instead of the default
in-project `.venv`:

```sh
export UV_PROJECT_ENVIRONMENT="$LOCALAPPDATA/uv/venvs/relik-runner"
```

(every `uv` command in this project then needs the same variable set).

## Docker

From the repo root (so the `runner-protocol` path dependency is inside
the build context):

```sh
docker build -f runners/relik/Dockerfile -t relik-runner .
docker run -v relik-hf-cache:/root/.cache/huggingface -p 8031:8031 relik-runner
```

## Provenance

Every prediction path shares the pipeline loaded from the pinned
checkpoint; `relik_runner.relik.provenance()` reports the identity row
(`model`, `backend: retriever-reader (faiss index + DeBERTa reader)`)
the legacy bench recorded alongside its results.
