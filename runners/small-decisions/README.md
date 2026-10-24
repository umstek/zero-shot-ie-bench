# small-decisions runner

The catch-all local decision-model island: one process serves the vLLM
Semantic Router **Decision 2.0** trio, the **Decision 3.0** trio,
IFM's **K2-Type-0.9B**, MoLeMo-Lab's **MoJev**, internlm's
**Intern-Decision** trio (all loaded in-process on the transformers 5.17
CPU stack) and the ten **Ollaya** daemon models (proxied over HTTP on the
TypeSafe System One wire contract — the daemon does the serving, this
process only forwards).

Ported from `legacy/engines/{decision2_client,d3_client,k2type_client,
intern_decision_client,julia_client,d1_client,ollaya_client}.py` and
`legacy/engines/{certo_engine,mojev_engine,rlcd_engine}/`.

## Served models (POST /v1/decision, /v1/score, /v1/classify)

### In-process (lazy load on first predict; fp32 on CPU)

| model_id | version | engine |
|---|---|---|
| `vllm-sr/decision-2.0-kai` | `decision-2.0-kai-0.6b@unversioned` | Decision 2.0 Kai 0.6B |
| `vllm-sr/decision-2.0-eos` | `decision-2.0-eos-0.8b@unversioned` | Decision 2.0 Eos 0.8B |
| `vllm-sr/decision-2.0-sol` | `decision-2.0-sol-2b@unversioned` | Decision 2.0 Sol 2B |
| `vllm-sr/d3-edge` | `d3-edge@unversioned` | Decision 3.0 Edge 0.6B |
| `vllm-sr/d3-lite` | `d3-lite@unversioned` | Decision 3.0 Lite 0.85B |
| `vllm-sr/d3-nano` | `d3-nano@unversioned` | Decision 3.0 Nano 2B |
| `ifm/k2-type` | `k2-type-0.9b@unversioned` | K2-Type 0.9B |
| `molemo-lab/mojev` | `mojev@0c8695b6252f4205907433d4e196a94f032e60c3` | MoJev 0.85B |
| `internlm/intern-decision-0.8b` | `intern-decision-0.8b@unversioned` | Intern-Decision 0.8B |
| `internlm/intern-decision-2b` | `intern-decision-2b@unversioned` | Intern-Decision 2B |
| `internlm/intern-decision-4b` | `intern-decision-4b@unversioned` | Intern-Decision 4B |

### Ollaya HTTP proxies (require the local Ollaya daemon)

| model_id | version | Ollaya tag |
|---|---|---|
| `vllm-sr/decision` | `decision-1.0@unversioned` | `decision` |
| `moritzlaurer/nli-deberta-v3-large` | `deberta-v3-large@unversioned` | `nli` |
| `moritzlaurer/nli-modernbert-large` | `modernbert-large@unversioned` | `nli:modernbert-large` |
| `alibiserikbay/jevk5-4b` | `jevk5-4b@unversioned` | `jevk5` |
| `eldanring/winnow-e4b` | `winnow-e4b@unversioned` | `winnow:e4b` |
| `madani/decima` | `decima-base-2.0@unversioned` | `decima` |
| `madani/decima-small` | `decima-small-1.1@unversioned` | `decima:small` |
| `logitlab/snap` | `snap1-2b@unversioned` | `snap` |
| `codekins/arbiter` | `arbiter-4b@unversioned` | `arbiter` |
| `txoka/credence` | `credence-v1@unversioned` | `credence` |

**The daemon dependency:** the ten proxied models answer through the local
Ollaya daemon (`ollaya serve`, <https://ollaya.dev/download>, default port
11435; this repo runs 0.12.1). Nothing loads here — `POST /v1/systemone`
on `OLLAYA_BASE_URL` (default `http://127.0.0.1:11435`, a full
`.../v1/systemone` URL wins as-is). Pull each tag first (`ollaya pull
<tag>`); a dead daemon fails those items' predictions with the HTTP error,
not the run.

## Also ported, not registered on /info

- **d1** (`models/d1.py`): the catalog assigns the local d1 deployments
  (`liquid/d1-3b`, `liquid/d1-omni-600m`) to `runner: von` — they need the
  transformers ≥5.14 trust_remote_code route and the von island owns them.
  The loader is ported and offline-tested here so a catalog reassignment
  is a one-line registration.
- **julia** (`models/julia.py`), **certo** (`models/certo.py`),
  **rlcd** (`models/rlcd.py`): no catalog yaml exists yet, so there is no
  catalog identity to serve; the engines are ported (Julia with the
  transformers 5.17 compat guard, certo as the vendored ModernBERT
  scorer, RLCD as the vendored constrained-decoding engine) and
  offline-tested, ready for catalog entries.

## Hardware needs

CPU-only (every catalog deployment here is `device: cpu`); RAM is the
binding constraint — the biggest in-process load is Decision 2.0 Sol-2B at
~4.8 GB fp32, Intern-Decision 4B at ~18 GB fp32. Weight loads are lazy per
model id; an absent snapshot fails that model's items with a named error
instead of taking the run down.

## Snapshot homes (env-overridable)

| engine | env | default |
|---|---|---|
| Decision 2.0 | `DECISION2_HOME` | `C:\src` prefix (snapshots `Decision-2.0-*.`) |
| Decision 3.0 | `D3_HOME` | `C:\src` prefix (snapshots `d3-*`) |
| K2-Type | `K2TYPE_HOME` | `C:\src\K2-Type-0.9B` |
| Intern-Decision | `INTERN_DECISION_HOME` | `C:\src` prefix (snapshots `Intern-Decision-*`) |
| d1 | `D1_HOME` | `C:\src` prefix (snapshots `d1-3B`, `d1-omni-600M`) |
| Julia | `JULIA_HOME` | `..\Julia-1` next to the repo |
| certo | `CERTO_WEIGHTS` | HF cache of `altslate/certo-decision-model` |
| MoJev / RLCD | — | HF hub (cache after first pull) |

Nothing is pip-installed for Decision 2.0/3.0, K2-Type, Intern-Decision or
Julia: their runtimes ship inside the snapshots and are sys.path-imported
(the house pattern). MoJev's scorer class and RLCD's weights load from the
HF hub via trust_remote_code / pinned revisions. certo and the RLCD
inference code are vendored verbatim under `src/small_decisions/models/
_vendored/` with their attribution headers.

## Run

    uv sync
    uv run uvicorn small_decisions.server:app --host 127.0.0.1 --port 8000
    # or
    uv run python -m small_decisions.server

Tests: `uv run pytest` (offline — stubs everywhere, the Ollaya proxies
against a stubbed HTTP server). `uv run pytest -m slow` additionally loads
real weights where the snapshots are locally present. Type check:
`uv run pyrefly check` (config-driven; note a path argument — `check .` —
makes pyrefly ignore the configured `project-excludes`, which silence the
verbatim vendored upstream files).
