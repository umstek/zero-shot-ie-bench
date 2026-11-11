# Pending access & follow-ups (2026-11-11 round)

Systems found in the 2026-11-11 fan-out but not yet integrated, split by
what unblocks them. Everything under "Needs a key" is otherwise ready to
wire once the credential arrives; the gate-12 trial run decides each one.

## A. Needs a key / access (get these)

| System | Vendor | What to get | Notes when it arrives |
|---|---|---|---|
| meraGPT Decider 1 (sd-1) | meraGPT | `MERAGPT_API_KEY` (meragpt.com) | Speaks TypeSafe's `/v1/systemone` wire per its docs — likely reuses the `typesafe.systemone` adapter against its base URL; verify wire-compat first |
| Cohere Rerank 5 / 5 Fast | Cohere | `COHERE_API_KEY` (native API) | Rerank 4 tiers left OpenRouter; Rerank 5 is Cohere-native `/v2/rerank` — near-identical to the `openrouter.rerank` contract, needs a small `cohere.rerank` adapter |
| Voyage rerank-3 / rerank-3-lite | Voyage AI | `VOYAGE_API_KEY` (native API) | Same story as Cohere: rerank-2.5 left OpenRouter; native `/v1/rerank` needs a small adapter |
| Bespoke Nimble (9B) | Bespoke Labs | `BESPOKE_API_KEY` (docs.bespokelabs.ai) | Hosted REST classification with per-answer probabilities; custom contract → new adapter |
| Parallel Decisions Fast | Parallel | nothing new — free `:free` route on OpenRouter | Custom contract (state seed + ≤100 mixed questions + citations); needs a new adapter, no key blocker |
| Hanzo Kai (Kai 1 / 1.9) | Hanzo | `HANZO_API_KEY` (hanzo.ai) — or confirm an OpenRouter slug | Hanzo's blog documents `/v1/decisions` (OpenRouter Decisions wire); no OR listing found as of 2026-11-11 (`hanzo/kai` 404) |
| Snowflake Decision / Decision Fast | Snowflake | Private-preview access + `SNOWFLAKE_API_KEY` | Cortex AI_COMPLETE decision primitive (SQL/REST); custom contract, in-warehouse governance |
| djev | Maisa | invite / recovery key (djev.dev) | Custom yes/no/choice/score API over text/JSON/images; invite-gated |
| Hume Audio Decision API | Hume | `HUME_API_KEY` (waitlist) | Audio-native decisions — likely out of scope for the text/image suites; listed for completeness |
| Tazo | Tazo Labs | request API access | Audio-native (voice agents) — likely out of scope |
| Jev Router | TypeSafe | nothing to get — retry the endpoint later | Web listing (openrouter.ai/typesafe/jev-router) resolves, but the API 400s "Model typesafe/jev-router does not exist" (2026-11-11). Stub listing; catalog yaml + failed run were removed. Re-add one yaml when the route serves |

## B. Engineering deferrals (no key needed; unblock condition stated)

- **GLiDE-0.4B island module** (catalog entry `fastino/glide-0.4b` exists):
  the glide Python API ships inside the HF repo (`Glide.from_pretrained`,
  `glide.server:app`), not on PyPI — the PyPI `glide` name is an unrelated
  ETL package. Unblock: load via the in-repo API (transformers 5.17.0 =
  island pin, fine; repo pins torch 2.14.0 vs island 2.8.0 — validate) or
  point the bench at its `glide.server:app` System One endpoint like any
  hosted-locally model.
- **Liquid d1-2B / d1-Omni-2B open weights**: demand transformers >=5.19;
  the systemone-local island pins 5.17.0. Unblock: island bump decision.
- **lumma-fev-0.1B-v2**: needs the `lumma-fev>=0.3` package (new engine dep,
  unvalidated against the small-decisions island's pinned stack; the island
  currently serves no lumma models at all — v0 ran them out-of-island).
- **jevk5-2b**: no Ollaya daemon tag exists (`jevk5` serves only 4B v0.3)
  and no v0.3 2B GGUF has been published (HF has 2B **v0.2** only). Unblock:
  an `jevk5:2b` daemon tag or a v0.3 2B GGUF; then it's one TAGS row + yaml.
- **decima:agent** (integrated — catalog + island row are in): to actually
  run it the Ollaya daemon must be upgraded to **0.13.0+** (repo's machines
  run 0.12.1; late-interaction-v2 layout).
- **gliclass v4 / v4.1-edge**: gliclass 0.1.22 + transformers-5-declaring
  metadata; needs re-validation against the gliner island's
  `transformers==4.57.6` override before wiring.
- **gliformer-multi-base-v1**: same — validate against `gliformer==0.1.2`.
- **mxbai-rerank-v3.1-listwise**: serving route unclear (API vs weights).

## C. Too big for the CPU laptop (noted, not benched locally)

GLiDE 0.8B/2B/9B/27B · kev 9B/27B v2 · winnow-12B · Drex Max 27B weights ·
d3-mini 5B / d3-flash 8B / d3 / d3-turbo / d3-ultra 26B tier ·
Nemotron-3.5-Rerank-8B · kanana-2-v-decide-3B (borderline, ~15GB fp32) ·
THX-02 4.5B · iota-decide-4b. Hosted variants of these are still fair game
via OpenRouter.

## D. Next-round small local candidates (all CPU-fine, none gated)

tasksource-decider-nano (149M) · semgram decider-75M · OneJev1.6-0.8B
(ONNX int8 port exists) · QEV-0.8B (own `qev` pip) · THX-01 (322M) ·
vela3.1-pico-380m · Matilda-Decision-v1-Edge · sura-0.8b · tev1-0.8B ·
chinese-decision-unified-0.8b · finecat-nli-v2-xxs (17M) ·
UltraBERT-150M-nli · AbsoluteZero-nli-2b · SauerkrautLM-GLiNEREL-128m
(glinerel-fast pip) · grt-30m-reranker · Yuanbao-Reranker-0.6B ·
gaggle-reranker-0.9b · MoDe-S (non-commercial license — decide policy).
Each needs a runner module in an existing island (or a new island where
pins conflict); none need credentials.
