# nanodiff runner island

Serves **pngwn/nanodiff-350m-typed-decisions-lam1** and **pngwn/nanodiff-350m-typed-decisions-v2**
— the two 350M diffusion-LM decision checkpoints (MIT) — behind the shared
runner protocol, on the main-env CPU stack: **Python 3.13, torch 2.x (CPU
wheels), tiktoken**. The only non-autoregressive system in this repo: the
answer-letter position is the only `[MASK]`ed token, ONE bidirectional
forward scores every option (softmax restricted to the option-letter token
ids `' A'..' J'`, GPT-2 BPE), argmax = decision. No decoding loop, no
generation. Slow on CPU (~10 s/question) — kept as the diffusion datapoint.

The pair is **culled from the benchmark** (chance-level on the 12-item
trial, TRIALS.md) and has **no catalog yaml** (trial-level systems, not
v0 system-table rows — `v0.benched` would be false; the only recorded
legacy entries are leftover `/timing` keys). The island exists so the
diffusion engine keeps its pins, tests and a serving path for the day it
gains catalog identity. The wire identity is the HF repo id; the version
tags name the step-3000 artifacts the checkpoints carry.

## Pins (why this island exists)

- **torch** — the engine ran in the MAIN venv (`legacy/requirements.txt`
  has no torch line; it resolved transitively — the main `.venv` carries
  torch 2.14.0 and ran the nanodiff benches). Current-major floor here,
  exact version from the lock, CPU wheel index.
- **tiktoken>=0.14** — `decision_format.py` is GPT-2-BPE token math
  (PROMPT_LEN budgeting, option-letter token ids); the main-env tiktoken
  it was validated on.
- **checkpoints** — `CKPT` (v1 lam1 release) and `CKPT_V2` (the v2
  retrain, step 3000; its embedded run args say lam=1.0, seed 1337 — the
  same run the empty `…-v2-lam1` repo was meant to publish). Same
  architecture and state-dict layout, loadable unchanged:
  `n_layer=16, n_head=20, n_embd=1280, block_size=512`.

## Layout

- `src/nanodiff_runner/decision_format.py` — the typed-decision
  single/multi-choice serialization, verbatim from the pngwn release
  (`code/decision_format.py`)
- `src/nanodiff_runner/nanodiff/` — `Config` + `NanoDiff`, verbatim from
  github.com/BY571/nanoDiff (bidirectional LLaMA-style transformer: RMSNorm,
  SwiGLU, RoPE, no causal mask); the vendored package keeps its upstream
  name because the checkpoint pickle resolves classes through it
  (`engine.py` aliases `sys.modules["nanodiff"]`)
- `src/nanodiff_runner/engine.py` — checkpoint loading + the released
  scoring path, ported from `legacy/engines/nanodiff_engine/runner.py`
  (itself adapted from the release's `code/eval_calibration.py`), plus the
  island glue: `cached_checkpoint()`, per-checkpoint lazy singletons,
  `provenance()`
- `src/nanodiff_runner/server.py` — `create_app` wiring, `python -m
  nanodiff_runner.server` entry, default port 8032
- `tests/` — the ported multi-format tests, the runner-protocol TestClient
  suite over a stub engine, and the cache-gated slow tests

## Hardware

CPU-only, ~3 GB RAM (the ~380M-param fp32 state dict plus activations). No
VRAM needed. First use downloads the checkpoint (~1.5 GB fp32) into the
local HF cache; afterwards it runs offline.

## Run

```sh
uv sync
uv run python -m nanodiff_runner.server --host 127.0.0.1 --port 8032
```

(or `uv run uvicorn nanodiff_runner.server:app --port 8032`).

- `GET /healthz` — liveness.
- `GET /info` — the two checkpoints with runtime (`torch`, `bfloat16`
  autocast compute, fp32 weights, `cpu`), host fingerprint.
- `POST /v1/decision` — items `{item_id, input, question, params}`;
  `question.type` dispatches:
  - `choice` — one masked slot. `question.options` (list) or
    `question.choices` ({label: description}) supply the ordered option
    texts (letters A-J map by position, 1..10); the question string comes
    from `question.text` / `question.instructions` / `question.question`,
    falling back to the bench phrasing for a known `params.task`
    (`sentiment`, `topic`), then to the generic "Choose the best option."
    (the suites' decision items carry options only). Answer: `prediction`
    = the option text, `prob` = its answer-slot softmax mass.
  - `multi` — the release's multi-decision form: `question.questions` is
    a list of `{text, options}` answered on ONE numbered line, k masked
    slots, one bidirectional forward. Answer: `prediction` = one pick per
    question in order, `prob` = null (the wire prob is a single float; the
    per-slot distributions are a `predict_multi` return value).

Weights load lazily on the first predict, so startup is instant and the
load never happens at import.

## Tests

```sh
uv run pytest
```

Offline by construction: the multi-format tests are pure tiktoken math,
and the TestClient suite runs the real format/softmax path over a tiny
seeded NanoDiff (`conftest.tiny_nanodiff`, `load_model` patched — same
class, same forward, ~1/1000 the trained size). The slow tests load the
real checkpoints and skip per checkpoint unless the blob is already in
the local HF cache (`cached_checkpoint()`, `local_files_only=True`; CI
never downloads weights):

```sh
uv run pytest -m slow
```

To seed the cache locally:

```sh
uv run python -c "from nanodiff_runner import engine; engine.load_model('cpu')"
uv run python -c "from nanodiff_runner import engine; engine.load_model('cpu', engine.CKPT_V2)"
```

(then `uv run pytest -m slow` runs the real-weight suite offline too).

Machine note (this desktop): the data disk is nearly full. Put this
island's venv on the system drive instead of the default in-project
`.venv`:

```sh
export UV_PROJECT_ENVIRONMENT="$LOCALAPPDATA/uv/venvs/nanodiff-runner"
```

(every `uv` command in this project then needs the same variable set).

## Docker

From the repo root (so the `runner-protocol` path dependency is inside
the build context):

```sh
docker build -f runners/nanodiff/Dockerfile -t nanodiff-runner .
docker run -v nanodiff-hf-cache:/root/.cache/huggingface -p 8032:8032 nanodiff-runner
```

## Training scripts

The island vendored only the inference path (`Config`, `NanoDiff`, the
typed-decision format). The training scripts stay upstream at
github.com/BY571/nanoDiff (MIT) and are NOT re-vendored:

- `pretrain/` + `sft/` — the two training entry points; each takes a
  config under `pretrain/configs/` / `sft/configs/` that instantiates the
  vendored `Config` dataclass with overrides (copy a config, edit
  `n_layer` / `n_embd` / `n_head` / the optimizer fields — the same
  `Config` this island ships in `src/nanodiff_runner/nanodiff/config.py`)
- the typed-decisions SFT recipe that produced both checkpoints lives in
  the pngwn release repos
  ([lam1](https://huggingface.co/pngwn/nanodiff-350m-typed-decisions-lam1)
  — `code/decision_format.py`, `code/eval_calibration.py`; this island's
  `decision_format.py` is that file verbatim). The checkpoints were
  trained with `sft.encode_sft_example`'s layout: right-aligned prompt
  left-padded with EOT, response left-aligned with an EOT end marker,
  exactly what `encode_example` mirrors.
- the 350M base the cards cite is `Sebasdi/nanodiff-350m-base`

To retrain or fine-tune: clone BY571/nanoDiff, point its data prep at a
typed-decisions JSONL (state / question / options / answer letter), and
load the result here by adding its `repo::filename` to
`engine.CHECKPOINTS` — `load_model` needs only the same state-dict
layout (`ARCH`).

## Provenance

`nanodiff_runner.engine.provenance(ckpt)` reports the identity tuple a
run record carries (`model` repo, `checkpoint` filename, vendored engine,
`format: typed-decisions`); the legacy bench recorded the same repo ids
alongside its `/timing` entries.
