# OpenRouter hosted round — 2026-10-18

Follow-up to the 2026-10-10 sweep, triggered by the Oct 14–17 arrivals
and reframed by a user constraint: no disk space for local weights, so
this round verifies which new systems are reachable hosted —
specifically through OpenRouter, which is what the bench harness
speaks. Sources: OpenRouter's per-model endpoints API (authoritative
for slugs/endpoints/pricing/context), the SSR decisions catalog page
(`openrouter.ai/models?output_modality=decisions` — surfaces the
decisions-modality models the public `/api/v1/models` catalog omits),
org pages, the MarkTechPost Oct-17 roundup of 20 Jev alternatives, HF
model cards, and docs.typesafe.ai (no new TypeSafe model; jev-1.14.0
from 2026-10-14 remains newest). Research + gate probes only; no full
pool.

## Verified on OpenRouter (endpoints API, 2026-10-18)

| Model | id | endpoint | ctx | price | listed |
|---|---|---|---|---|---|
| Decider V1.2 27B | `perplexity/pplx-decider-v1.2` | `pplx-decider-v1.2-20261016` | 262,144 | $0.02/M input, output free | 2026-10-16 |
| Decider Flash V1.2 | `perplexity/pplx-decider-flash-v1.2` | `pplx-decider-flash-v1.2-20261016` | 262,144 | $0.01/M input | 2026-10-16 |
| d1-ultra | `liquid/d1-ultra` | `d1-ultra-20261015` | 32,768 | $0.042/M input, output free | 2026-10-15 |
| ScaleDown Extract | `scaledown/extract` | `extract-20260928` | 1,000,000 | $0.04/M | 2026-09-28 |

- **Decider V1.2** is the first hosted entry claiming to beat Jev:
  Perplexity's 11-task panel puts it at 85.71% vs Jev's 84.51%
  (borrowed, not measured here). V1.2/Flash V1.2 are absent from the
  Perplexity org page — its FAQ still calls V1.1 (Oct 7) the newest —
  while the decisions catalog shows both since Oct 16. The org pages
  cannot be trusted for this modality: Liquid's org page (10 models)
  omits the whole d1 decisions line, d1-ultra included.
- **d1-ultra** is Liquid's step up from the benched `liquid/d1` row
  (same $0.042/M input price as Jev, 32k ctx, output free — measured
  `output_tokens: 0` on the gate run).
- **ScaleDown Extract** is extraction-lane, not typed decisions:
  description-driven entity extraction over a chat-completions payload
  (`provider.options.scaledown.entities`, 1M ctx). If the extraction
  comparison grows, it needs its own trial mapping against the NER
  pool, the way the hosted GLiNER twins ride `engines/fastino_client.py`.
- Provider data-retention policies for the new endpoints are **not yet
  probed** (the benched `liquid/d1` row carries the † non-ZDR marker;
  Decider V1/V1.1 passed this account's ZDR). Check before any recorded
  README row.

## Trial gate (measured here, 2026-10-18)

`_trials/trial_13_openrouter_oct18.py` — the standard fixed 12-question
mini-pool, one warm-up + one batched request per model, whole
three-model run ≈ $9.2e-5:

| Model | Gate | Latency/q | Batch cost |
|---|---|---|---|
| Decider V1.2 27B | 9/12 | 0.131 s | $2.1e-5 |
| Decider Flash V1.2 | 10/12 | 0.140 s | $2.8e-5 |
| d1-ultra | 10/12 | 0.058 s | $4.3e-5 |

All three miss the same two mixed-sentiment questions ("great camera,
but…", "Service was slow…, though…" — the conceding clause reads as
neutral; the same family jev-1.14.0 newly misses); the two 10/12s fix
"Not the cheapest option, but easily the most reliable", where V1.2
also reads neutral. None clears the 12/12 gate, but the jev-1.14.0
override (TRIALS.md) is the standing caution that this all-sentiment
slice under-sees hosted models — the benched hosted incumbents
(Decider V1/V1.1, d1) all recorded 93.8% on the full 48-question mixed
pool. Full-pool runs are a user call.

## On the catalog but unresolved

An unnamed **respan** entry dated 2026-10-16 at $0.01/M input sits on
the SSR decisions catalog; the respan org page lists only Span-01
($0.02/M) and the two Lite/free variants. Could be a repriced endpoint
or a new variant — no slug resolved. Check before the next Span-01
re-run.

## Hosted, off OpenRouter (own APIs only)

From the Oct-17 MarkTechPost roundup plus probes; none found on
OpenRouter:

- **meraGPT Decider 1** (Sep 22) — own `/v1/systemone`, $0.04/M, 4k
  ctx, 10-label cap: the context and label cap make it a weak fit for
  the pools here.
- **Standard Thinking Standard One** — 3B/8B (Ministral-3-8B LoRA),
  hosted + open weights.
- **AWS Strands Labs Strands Decider 2B** (Oct 1) — Qwen3.5-2B pointer
  head, Apache-2.0, claims 77.9% JevBench public (borrowed);
  AWS-distributed.
- **razorback16 OpenJev-DiffusionGemma-26B-A4B** — free community
  endpoint with a 100M-token cap (availability untested).
- **InnerJev-4B/27B** (Fudan/Tongji/KCL) — weights only, no hosted
  endpoint found.
- **Respan's third-party behavior benchmark** of decision models
  (Oct 16): flip rate under paraphrase, injection robustness, ECE —
  methodology worth a look for this repo's calibration tables.

## Method notes (what worked / what didn't)

- `GET /api/v1/models/{slug}/endpoints` is the reliable probe for this
  modality: exact slug, dated endpoint, context length, pricing. It
  answers for decisions models the public catalog hides; wrong guesses
  return `data: null`.
- The frontend find API
  (`/api/frontend/models/find?...&output_modalities=decisions`) 404s —
  the endpoint does not exist. The SSR catalog page bleeds text models,
  so treat it as a hint list and confirm every slug via the endpoints
  API.
- HF API (`/api/models?search=…&sort=createdAt`, `?author=…`) plus raw
  README fetches remain the best open-weights enumeration lane. This
  round surfaced the OneJev 1.5/1.6, Jebadiah v2/v2.1, StartLux v1.1,
  kenning-xl-v0.6, ezjev, and razorback16 families and the
  `multimodalart/jev-decision-index` space (Decision Index 0.3: 37
  public benchmarks, 140k requests, 83 models <8B; OneJev1.5-4B 54.89 =
  2nd <8B) — all skipped for benching here under the no-disk
  constraint this round.

## Next steps

- Full-pool runs for the three gate passers — a user call, cf. the
  jev-1.14.0 override.
- Probe data-retention policy on the new endpoints before any recorded
  row († bookkeeping in the README).
- Resolve the respan Oct-16 entry.
- If the extraction lane grows, map `scaledown/extract` onto the NER
  pool trial.
