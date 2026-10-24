# systemone-local runner

The local System One-contract decision engines, served as runner models on
one FastAPI app (`POST /v1/decision` + `GET /info` + `GET /healthz`, the
shared `runners/_protocol` factory). One process, one venv, weights loaded
in-process and lazily — the first predict pays the load.

| model_id | catalog version | engine | weights | where the engine lives |
|---|---|---|---|---|
| `kev-0.8b` | `kev-0.8b@2026-09` | Qwen3.5-0.8B + LoRA + pointer head (System One contract) | HF `jaredpalmer/kev-0.8b` (run: head.pt + adapter) | the jaredpalmer/kev clone — not on PyPI |
| `decider-0.8b` | `decider-0.8b@unversioned` | Qwen3.5-0.8B fine-tune, 752M (System One contract) | HF `Mapika/decider-0.8b` | PyPI `decider-ai` |
| `openthai-systemone` | `openthai-systemone@unversioned` | Qwen3.5 Gated DeltaNet hybrid + 256-way slot head, Thai/English | HF `iapp/OpenThai-SystemOne` (gated) | PyPI `openthai-systemone` |
| `agentjev-v1` | `agentjev-v1@unversioned` | candidate head over Qwen3-0.6B — **own `/api/evaluate` contract, not System One** | `agentjev_v1.pt` + a Qwen3-0.6B snapshot | the malevrigns/agent-jev clone — not on PyPI |

All four serve the **decision** track (the catalog's only claimed track with
suites); choice, score and noul/boolean questions are all accepted.

## Where the engines come from

The System One trio used to run as three separate uvicorn servers, one port
each (see `legacy/demos/systemone_locals_demo.py` and the serve logs):

    # kev 0.8B — historical port 8009
    cd ../kev && uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009
    # decider 0.8B — historical port 8018
    DECIDER_MODEL=Mapika/decider-0.8b DECIDER_DEVICE=cpu uv run --with "decider-ai[serve]" \
        python -m uvicorn decider.serve:app --host 127.0.0.1 --port 8018
    # OpenThai-SystemOne — historical port 8029
    OPENTHAI_SYSTEMONE_MODEL=iapp/OpenThai-SystemOne uv run --with "openthai-systemone[server]" \
        python -m uvicorn openthai_systemone.server:app --host 127.0.0.1 --port 8029
    # AgentJev 0.6B, own /api/evaluate contract — historical port 8149
    cd ../agent-jev && python -m jev_service.server --checkpoint agentjev_v1.pt \
        --model-path <Qwen3-0.6B snapshot> --temperatures temperatures.json --port 8149 --device cpu

This island replaces the HTTP hop with an in-process import of each engine
(one module per engine under `src/systemone_local/models/`), keeping the
ports above as the historical record — the island itself serves all four
models on one port and does not bind them.

## Run

    uv sync                       # torch downloads (~200 MB CPU wheel); NO model weights
    uv run --project runners/systemone-local python -m systemone_local.server
    # RUNNER_HOST / RUNNER_PORT override; default 127.0.0.1:8000

## Engine locations (D1_HOME-style env vars)

| Engine | Clone home | Weights |
|---|---|---|
| kev | `KEV_HOME` (default: the `kev` clone next to this repo) | `KEV_WEIGHTS` (run dir) or `KEV_RUN` (hub id, default `jaredpalmer/kev-0.8b`); resolved through the HF cache, downloaded on first predict if missing |
| decider | PyPI (`decider-ai`) | `DECIDER_WEIGHTS` (snapshot dir) or `DECIDER_MODEL` (default `Mapika/decider-0.8b`) |
| openthai | PyPI (`openthai-systemone`) | `OPENTHAI_WEIGHTS` (snapshot dir) or `OPENTHAI_MODEL` (default `iapp/OpenThai-SystemOne`); **gated on HF — needs an HF token to download** |
| agentjev | `AGENTJEV_HOME` (default: the `agent-jev` clone next to this repo) | `AGENTJEV_CHECKPOINT` (default `HOME/agentjev_v1.pt`), `AGENTJEV_MODEL_PATH` (default `HOME/models/Qwen3-0.6B-Base`, else the `Qwen/Qwen3-0.6B` HF-cache snapshot), `AGENTJEV_TEMPERATURES` (default `HOME/temperatures.json` when present) |

Device selection follows the catalog (`requires: {accel: cpu}`): every engine
defaults to CPU/fp32; `KEV_DEVICE` / `DECIDER_DEVICE` / `OPENTHAI_DEVICE` /
`AGENTJEV_DEVICE` override.

A model whose weights are absent still appears on `/info` (the island serves
it); its items come back with a per-item `error` naming the fix — the
protocol's per-item isolation means one absent engine never takes a run down.
`uv run python -c "from systemone_local.models import kev, decider, openthai, agentjev; print(kev.available(), decider.available(), openthai.available(), agentjev.available())"`
checks what can actually load (offline; the tests skip on the same check).

## Hardware

0.8B-class fp32 on CPU: ~3–5 GB RAM per engine (one at a time in-process;
the engines are loaded lazily and kept). No GPU needed; the catalog's
`requires` for all four deployments is `accel: cpu`. Cold load runs minutes
(OpenThai worst — the reason the legacy bench used a 600 s warm-up timeout).

## Tests

    uv run pytest

- `tests/test_server.py` — the app under TestClient with the engines
  stubbed; every engine module must also re-import cleanly with
  torch/transformers blocked (nothing loads at import time).
- `tests/test_payloads.py` — the offline payload translation: suites'
  decision shape to System One questions, answer rows to
  `{prediction, prob}`, and the AgentJev row shapes checked against the
  rules `jev_service.contract.prepare()` enforces (ported from the legacy
  demo tests).
- `tests/test_weights_live.py` — one real question per engine; **skips
  unless that engine's weights are locally present** (`available()`), so CI
  never downloads weights (R2/CI rule in docs/plan/02).

## Pins

The stack is anchored on the kev clone's validated serving venv: torch
2.8.0 + transformers 5.17.0 + tokenizers 0.23.2 + peft 0.21.0 (kev needs
transformers >=5.17 and torch 2.6..2.9 — the reason it never lived in the
legacy main venv's transformers 4.57.6). decider-ai (transformers>=5) and
openthai-systemone (transformers>=5.0, torch>=2.4) fit the same env, so all
four share this island. AgentJev's clone wants torch>=2.0 /
transformers>=4.40 — satisfied, but the clone itself is not pip-installable,
so it stays optional: without it the island imports clean and only AgentJev
items error.

`decider-ai` pulls `flash-linear-attention` (its Qwen3.5 Gated DeltaNet
kernels); the historical 8018/8029 servers ran without it and transformers
fell back to its reference implementation — it rides along here but is not
load-bearing.

## Docker

    docker build -t runner-systemone-local .
    docker run -p 8000:8000 runner-systemone-local
