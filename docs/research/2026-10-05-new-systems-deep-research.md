# New systems deep research — 2026-10-05

Sweep for zero-shot-IE systems **not yet integrated**, run as four parallel web lanes
(extractors / rerankers / local decision engines / hosted APIs) against the integrated
list as of `main@ccfda92` (TypeLLM, in-flight on `feat/typellm`, excluded). Window:
released or materially updated since ~April 2026. Research only — nothing downloaded
or run.

## Top candidates

### Local — rides an existing engine (cheap)

1. **GLiNER-X** — knowledgator/gliner-x-{small,base,large} · 300M/494M/865M · Apache-2.0
   · 2026-04-29 · https://huggingface.co/knowledgator/gliner-x-large
   New multilingual GLiNER generation (mT5 backbone, FineWeb-2 synthetic multilingual
   training, ~21 langs incl. lt/et/lv/sl/uk/da/sv/no/cs/pl + ar/zh/hi — near-perfect
   coverage of the bench's medium/rare tiers). Rides the GLiNER engine unchanged.
   Caveats: `gliner[stanza]` tokenizer dep (Windows check needed); HF-card-only
   release, no blog/paper; English accuracy vs GLiNER2.5 unverified.
2. **GLiClass Multilang** — knowledgator/gliclass-multilang-{mini,edge,ultra} ·
   143M/284M/1.7B · Apache-2.0 · 2026-04-30 ·
   https://huggingface.co/knowledgator/gliclass-multilang-ultra
   Multilingual zero-shot classification, ~20 langs, notably Hebrew (absent from
   GLiNER-X's list); the v3 "logic" generation. Rides the gliclass engine.
   Caveat: "ultra" 1.7B is heavy for an encoder classifier; verify `pip gliclass` compat.
3. **nanodiff v2** — pngwn/nanodiff-350m-typed-decisions-v2 (and -v2-lam1) · 2026-09-16/17
   Refresh of the only non-autoregressive (diffusion) system; the vendored
   `engines/nanodiff_engine/` makes this a drop-in. Also new in-org:
   pngwn/system-one-qwen3.5-4b-scorer.
4. **Certo 4B line** — altslate/certo-r1-qwen3-4b (2026-09-25) + altslate/certo-unified-4b
   (2026-09-28), plus certo-decision-v2.2 ·
   https://huggingface.co/altslate/certo-r1-qwen3-4b
   Material family update (421M → 4B class). Rides `engines/certo_engine/`.
5. **NVIDIA llama-nemotron-rerank-1b-v2 (local)** — 1.24B · OpenMDW-1.1 · weights
   updated 2026-08-26 · https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2
   Pointwise cross-encoder, multilingual (26 eval langs incl. bn/he/th), 8k ctx —
   maps exactly onto the (instruction, label)→argmax contract. Rides the local
   reranker engine (gte/bge/mxbai pattern). Distinct from the hosted
   nemotron-rerank-vl-1b already benched. No major caveats.
6. **jina-reranker-v3.5 (local)** — 0.6B (Qwen3-0.6B) · CC-BY-NC-4.0 · 2026-07-14 ·
   https://huggingface.co/jinaai/jina-reranker-v3.5
   Listwise "last-but-not-late" (one pass, per-candidate cosine scores); beats
   Qwen3-Reranker-4B on BEIR at 6.7× smaller; 131k ctx; GGUF available.
   Caveats: non-commercial license (bench precedent: deberta NLI card notes);
   listwise interface needs a thin adapter over the pointwise contract.
7. **mxbai-rerank-large-v2** — 1.54B · Apache-2.0 · multilingual ·
   https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2
   Size addition to the already-covered v2 family; zero friction. (Core v2 release
   Mar 2025 — a size add, not a new version.)
8. **Decision-Tune 1.0** — decision-tune/decisiontune-1.0 · ModernBERT-large · ONNX ·
   Apache-2.0 · 2026-10-04 · zero-shot-classification pipeline — drops straight into
   the Verdict/Julia in-process pattern. Brand-new, low traffic.
9. **Vela-Decision-170M** — ParallaxOpen · 170M · 2026-10-04 · custom_code ·
   calibration showcase (ECE ~6× better than laya, lower accuracy). Niche
   calibration test case only.

### Local — new runtime / heavier (medium effort)

10. **OpenJev** — openjev/openjev (CC-BY-NC-4.0) vs AlexWortega/openjev (MIT) ·
    2026-09-16/20 · https://huggingface.co/AlexWortega/openjev
    Open-weights Jev-alike: choice/noul/score, ≤52 options, one forward per question,
    ships a `/v1/systemone` vLLM-style endpoint; en/de/fr/hi/zh/ja. Sibling line:
    ZefanCai/Open-Jev-{2B,9B,27B}-v1.1; ggml-org/OpenJev-GGUF exists. Not on Ollaya.
    Pick the MIT variant deliberately; own-runtime Windows risk.
11. **Jeeves** — PostHog · Ollaya (`jeeves`) · 2026-09-29 · Qwen3.5-9B + merged LoRA +
    pointer head scoring every option at its own marker; 0.680 on Ollaya's suite.
    Heavy (9B, ~838 ms/q); English.
12. **Nimble** — Bespoke Labs · Ollaya (`nimble`) · Qwen3.5-9B contrastive LoRA; scores
    options by next-token logit of option codes, **up to 255 options**, calibrated;
    0.665. The 255-option ceiling is the draw for wide IE schemas. Same 9B caveat.
13. **Clef-Flash on Ollaya (local twin of hosted Clef)** — 2026-09-30 · Apache-2.0 ·
    https://ollaya.dev/library/clef
    Joint schema head: scores **every option of every question in one forward per
    request**, probabilities straight from the head (ECE 0.020, no fitted
    temperature) — architecturally the best IE match in the decision lane. Caveats:
    19 GB BF16 → 24 GB GPU; Ollaya caps 4096 ctx, text-only; ggml-org/Clef-Flash-GGUF
    as llama.cpp fallback.
14. **retrico-lm-4b** — knowledgator · Qwen3.5-4B · 2026-06-30 ·
    https://huggingface.co/knowledgator/retrico-lm-4b
    Generative universal IE: text/Markdown/HTML/XML + arbitrary JSON schema →
    schema-conformant JSON (NER, relations, nested). First *local open* model in the
    TypeLLM/Span-01 generation-extractor mold. Caveats: card recommends vLLM
    (bf16, 64k ctx) — Windows in-process infeasible, no GGUF yet; **license unstated —
    verify before investing**.
15. **GLiNER-multitask-large-v0.5** — knowledgator · ~large · Apache-2.0 · 2026-04-29
    One encoder, prompt-tunable across NER/RE/QA/keyphrase/sentiment-spans/OIE.
    English-only; prompt format needs a small adapter.

### Hosted APIs

16. **GLiNER2.5 + GLiNER2.5-Decide on the Fastino API** —
    https://fastino.ai/models/gliner2-5 · blog 2026-09-24 · Apache-2.0 weights
    Hosted twins of models we already run local (GLiNER2.5-Decide: 340M, joint
    constrained decoding, 60.1% avg on Fastino's 17-dataset Fast Decisions suite,
    p50 38 ms GPU / 167 ms CPU). Enables a same-weights local-vs-hosted cost/latency
    comparison — the bench's favorite kind of datapoint. New GLiNER REST adapter
    (not the System One contract); check whether -Decide is served under
    /v1/systemone like GLiDE. Vendor-internal benchmark only.
17. **typesafe/jev-router** — OpenRouter, created 2026-09-25 · routing endpoint over
    jev-latest ("picks best model + reasoning effort per request"). Model-id swap on
    the existing Jev adapter. Caveats: opaque per-request pricing (metered $/q gets
    messy); nondeterministic model selection muddies attribution.
18. **Schematron V2 Turbo/Small** — Inference.net, on OpenRouter 2026-09-12
    (`inference-net/schematron-v2-turbo` / `-small`) · 3B constrained-generation
    HTML→JSON extraction via `response_format` schemas · $0.03–0.05/M input ·
    OpenAI-compatible → rides the TypeLLM adapter pattern once that branch merges.
    Caveats: HTML-tuned rather than plain-text; gateway ZDR stance unverified.
19. **Jina Rerank API v3.5** — hosted twin of #6 (SaaS/AWS/Azure/GCP); pricing
    unverified.
20. *Friction note:* TypeSafe's direct API waitlist was removed 2026-09-27 (open
    signup at console.typesafe.ai). No new Jev version (still ~1.13.0; GLiDE leads
    their Decision Index 64.81 vs 57.91).

## False positives & closed questions

- **Laya (convaiinnovations)** — lane-3 flag was a dupe: `laya`/`laya-multilingual`/
  `laya-typed-decisions` are exactly what the bench runs (README systems table).
  Only third-party finetunes/quants since.
- **vllm-sr's other collections** — Vela 1.0 is a 307M fixed-taxonomy suite
  (PII/FactCheck/Guard heads — task-specific, not schema-driven); **MoM 1.0 / MoM
  Nano don't exist as IE models**; Decision-1.0-Route-0.6B is a routing-tuned Kai
  derivative. Nothing more to take from that org.
- **GLiFormer repos re-uploaded 2026-09-14/18** ("Upload folder using
  huggingface_hub" on base/large) — possibly refreshed weights. Worth hashing
  against the bench's pinned snapshots in case accuracy shifted silently.
- **Ollaya catalog (16 models)** otherwise stable: kev/decider/decision/jevk5/
  winnow/nli unchanged in mechanism; gliclass + qwen3guard predate the window;
  cygnet = training-free Gemma-4-12B prompt baseline; clm weak (0.357); jeb
  (Arabic typed-decisions GGUFs) viable but niche.
- **Cloudflare Workers AI** (catalog 2026-08-12): no new decision/extraction models
  beyond clef/clef-flash; new arrivals are generic structured-output LLMs.
- **Not new / dormant:** BAAI (nothing since May 2025), Snowflake arctic-rerank
  (dormant), Qwen3-Reranker sizes (pre-window), ReLiK (Feb 2026, unchanged),
  Cohere Rerank 4.x / Voyage rerank-2.5 (no newer signal — weakly verified, see
  below), Fastino GLiDE itself (unchanged).
- **Coverage gaps:** OpenRouter's public `/api/v1/models` doesn't enumerate rerank
  endpoints (UI diff needed for full confidence); search backends were degraded
  during part of the sweep (searxng google engine suspended, mwmbl empty,
  marginalia 429), so the Cohere/Voyage and small-entrant sweeps are weakly
  verified.

## Recommended integration order

1. **nanodiff v2 + Certo 4B line** — vendored engines exist; drop-in refresh/extension.
2. **GLiNER-X** — biggest multilingual-track payoff; clear the stanza dep on Windows.
3. **nemotron-rerank-1b-v2 (local) + jina-reranker-v3.5** — local reranker engine;
   thin listwise adapter + license note for jina.
4. **GLiClass Multilang** — multilingual classification lane.
5. **Hosted: Fastino GLiNER2.5/-Decide twins + Schematron V2** — rides the TypeLLM
   adapter pattern after `feat/typellm` merges.
6. **Ollaya locals: Jeeves / Nimble (9B heavies), Decision-Tune 1.0 (light).**
7. **Investigate before committing:** retrico-lm-4b (license unstated), OpenJev
   (pick the MIT repo), Clef-Flash local twin (needs ≥24 GB GPU).
