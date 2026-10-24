# von runner island

Serves **wfzyx/von-1.0** — the complete, trained option-marker checkpoint
(396M ModernBERT + option-marker head + marker calibration, Apache 2.0) —
behind the shared runner protocol, on its own pins: **transformers 5.17.0,
Python 3.13, CPU**. Every other island in this repo pins the transformers
4.57.x line, which is why von gets its own dependency island.

The model packs each question into one sequence whose `[MASK]` slots are
read in a single forward pass — no generation. The three System One
question types map onto the pinned SDK's backend:

| question type | SDK call | answer |
|---|---|---|
| `choice` | `evaluate_choice` | argmax over described options (`probabilities`, `confidence`) |
| `noul` | `evaluate_noul` | `P(condition true)` in 0.0-1.0 (no argmax pick) |
| `score` | `evaluate_score` | probability-weighted expectation over an ordered rubric |

## Pins (why this island exists)

- **von-sdk** @ git `a94aa368ff4dcc58cdfd569cf0b4b1ae7d592d54` — the PyPI
  1.0.1 wheel omits the trained option-marker backend; its default NLI
  loader would invent a random classifier for this checkpoint. Never move
  this rev without re-validating predictions against the legacy runs.
- **checkpoint** `wfzyx/von-1.0` @ `fbc0196efa4fa1825557a8d5adad464fa1e872bb`
  (the HF card now describes von-1.2; the catalog pins the von-1.0
  weights). Loading refuses to run if `option_marker.pt` is missing —
  no random-scorer fallback, ever.
- **transformers==5.17.0** (legacy/requirements-von.txt, verbatim);
  torch >= 2 resolved through the CPU wheel index.

## Layout

The Python package is **`von_runner`**, not `von` — the pinned von-sdk git
dependency installs a top-level `von` package (`von.types`, `von.backends.
option_marker_backend`) and the island must never shadow it.

- `src/von_runner/von.py` — the engine, ported from
  `legacy/engines/von_client.py` + the row mappings of
  `legacy/demos/von_demo.py` (same checkpoint pin, same SDK call path)
- `src/von_runner/server.py` — `create_app` wiring, `python -m
  von_runner.server` entry, default port 8030
- `tests/` — offline SDK-stub tests, the runner-protocol TestClient suite,
  and the cache-gated slow test over the real weights

## Hardware

CPU-only, ~3 GB RAM (`requires.ram_gb: 3` in the catalog, estimated).
No VRAM needed. First use downloads the ~0.4 GB checkpoint into the
local HF cache; afterwards it runs offline.

## Run

```sh
uv sync
uv run python -m von_runner.server --host 127.0.0.1 --port 8030
```

(or `uv run uvicorn von_runner.server:app --port 8030`).

- `GET /healthz` — liveness.
- `GET /info` — models (`wfzyx/von`, version `von-1.0@fbc0196…`), runtime,
  host fingerprint.
- `POST /v1/decision` and `POST /v1/score` — items `{item_id, input,
  question, params}`; the response carries `compute_ms` measured inside
  the server. `question.type` dispatches: `choice` (options list or a
  `{label: description}` `choices` object), `noul`, `score` (a `rubric`
  of at least two levels, lowest to highest). Weights load lazily on the
  first predict, so startup is instant and the load never happens at
  import.

## Tests

```sh
uv run pytest
```

Offline by construction: the engine is exercised against SDK stubs (the
same sys.modules patching legacy tests used), and the FastAPI contract is
exercised with a stubbed backend over `TestClient`. The one test that
loads the real weights skips unless the pinned snapshot — including
`option_marker.pt` — is already in the local HF cache
(`cached_snapshot()`, `local_files_only=True`; CI never downloads
weights). To seed the cache locally:

```sh
uv run python -c "from von_runner.von import load_von_backend; load_von_backend()"
```

Machine note (this desktop): the data disk filled up while installing the
torch stack. If `uv sync` or pytest fails with "not enough space on the
disk", put this island's venv on the system drive instead of the default
in-project `.venv`:

```sh
export UV_PROJECT_ENVIRONMENT="$LOCALAPPDATA/uv/venvs/von-runner"
```

(every `uv` command in this project then needs the same variable set).

## Docker

From the repo root (so the `runner-protocol` path dependency is inside
the build context):

```sh
docker build -f runners/von/Dockerfile -t von-runner .
docker run -v von-hf-cache:/root/.cache/huggingface -p 8030:8030 von-runner
```

## Provenance

Every prediction path shares the backend loaded from the pinned
checkpoint; `von_runner.von.provenance()` reports the identity tuple
(`model`, `model_revision`, `sdk_revision`, `backend: option-marker`)
the legacy bench recorded alongside its results.
