# Decision & Extraction (IE) Model Landscape — October 2026

*Research date: 2026-10-24. Scope: **decision models** (System One wave — typed choice/score/probability in one non-autoregressive pass) and **information-extraction models** (NER, relation extraction, schema→JSON extraction from text). Deliberately OUT of scope: OCR/document-PDF-parsing pipelines, guards/moderation models, reward models, LLM-judge platforms. Method: three parallel web-research agents (~100 searches/fetches total; sources at the end). Standalone note, independent of anything else in this repo. Shaky items are flagged **(unverified)**.*

---

## TL;DR

1. **Two ecosystems, one convergence story.** The decision-model wave (born Sep 15 with Jev, now 100+ entrants per Forbes) and the GLiNER-lineage extraction ecosystem evolved in parallel — but the same orgs now straddle both: Fastino ships GLiNER2.5 (extraction) *and* GLiDE + GLiNER2.5-Decide (decisions); ScaleDown ships both Decide and Extract. Typed outputs + calibrated confidence are flowing from decisions into extraction (ScaleDown Extract returns confidence + char offsets) and encoder-speed extraction ideas are flowing into decisions (GLiNER2.5-Decide runs decisions on CPU).
2. **Extraction is encoder-dominated again.** The GLiNER ecosystem (Knowledgator + Fastino) owns zero-shot NER/RE: GLiNER2.5 (boundary prediction, any-length spans, 4K context), GLiNER-bi-v2 (million-label ontologies), GLiFormer (nested JSON schemas, zero generated tokens), GLiNER-Relex (joint NER+RE). LLM-based schema extraction is a thin layer: NuExtract3 (4.5B, ExtractBench 82.35) and ScaleDown Extract ($0.04/1M, offsets + confidence).
3. **Event extraction and Open IE are dead zones.** Zero new models in-window; ADELIE (2024) remains the last instruction-tuned IE-LLM with event coverage. Multimodal NER: papers only, no shipped model.
4. **The decision wave has real infrastructure now.** Artificial Analysis' Decision Index v1.0 (Oct 13) is the neutral leaderboard; TypeSafe's `/v1/systemone` contract is the de-facto interop standard (Clef, Kev, Semif, Jeff implement it); OpenRouter has a decisions catalog with usage rankings. Perplexity open-sourced Decider v1.1-27B (Apache 2.0) and Microsoft shipped Decision-1 (9B, MIT).
5. **The big independent critique:** per Artificial Analysis, **every decision model is overconfident** — stated probabilities overshoot accuracy, worst on knowledge/reasoning; LLMs still win pure knowledge tasks, and Claude Haiku 5.5-Low outscores Jev on the Decision Index (61 vs 58) at 11× the latency and 4.4× the cost.
6. **A third category crystallized in late October: single-pass typed-value extractors.** Cloudflare's **Clairic** (Oct 19, preview) is the flagship — a schema of typed fields (text/number/date/duration/list/estimate/forecast), every answer with confidence + `p_null` + char offsets, up to 4 images, zero output tokens, $0.10/1M input, open weights "soon". The category is otherwise thin: ScaleDown Extract, Schemer (211M on-device), Inference.net's extraction endpoint.
7. **The AA Decision Index top is now a three-way tie at ~65**: d1-ultra, **SubQ Snap 1.1** (Subquadratic, Oct 22 — first long-context decision model, 384K ctx) and **Elephant-9B** (Base Labs, Oct 2026, MIT open weights + Baseten).

---

## Cluster A — Decision models (System One wave)

### The big four + key challengers

| Model | Org | Released | Open/API | Size | Text | Image | Video | Audio | Output types | Calibrated | Latency | Headline benchmark | Price (per 1M in) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Jev 1.14** (1.13 Sep 15) | TypeSafe | Oct 14 | API | undisclosed | ✔ | ✘ | ✘ | ✘ | choice, score, "noul" predicate | ✔ RLCD | 0.13 s/task median | AA Decision Index 57–58 | $0.042, output free |
| **Decider v1.1-27B / v1.2 / Flash** | Perplexity | Oct 1 / 6 / 16 | **Open Apache 2.0** + API | 27B (Qwen3.5) / Flash 4B | ✔ | ✔ (27B) | ✘ | ✘ | choice, probability | ✔ | — | ClaimBench 99.69 (#1, vals.ai); Decision Bench 94.5 | ~$0.02 hosted; weights free |
| **d1-ultra** | Liquid AI | Oct 15 | API | undisclosed | ✔ | ✔ | ✘ | ✘ | typed answers, probs | ✔ | — | **AA Decision Index 65 — since tied by SubQ Snap 1.1 & Elephant-9B** | $0.042, output free |
| **d1-3B / d1-omni-600M** | Liquid AI | Oct 7 | Open (LFM Open Lic. — free commercial <$10M rev) | 3.1B / 600M | ✔ | ✔ | omni: ✘ | omni: ✔ | typed answers, probs | ✔ | 8 ms/decision (RTX 4090) | Decision Index 48.6 (best <10B) | self-host |
| **Decisions API (GPT-6 Luna)** | OpenAI | GA Oct 22 (beta Oct 6) | API | undisclosed | ✔ | ✔ | ✘ | ✘ | predicate, choice, score; 400 Q/request | ✔ probabilities | ~10× faster than Responses API | Luna-Low 55; below Jev 1.13 at launch | $0.10, output free |
| **Clef / Clef-flash / Clef-omni** | Cloudflare | Oct 1 / Oct 9 | **Open weights** + Workers AI | 27B / 9B / 30B-A3B MoE | ✔ | Clef+omni ✔ | omni ✔ | omni ✔ | scored decisions | ✔ Brier+RLCD | flash: 38.8 ms median | BFCL case-exact 98.47; flash #4 OpenRouter usage | ~$0.15 (omni) |
| **GLiDE** | Fastino | Sep 30 | API + weights on HF | undisclosed | ✔ | ✘ | ✘ | ✘ | choices + confidence | ✔ | — | When2Call 84 (co-#1), WildGuardTest 95 | $0.04 |
| **Microsoft Decision-1** | Microsoft | Oct 9 | **Open MIT** + Foundry + OpenRouter | 9B (Qwen3.5-9B) | ✔ (EN) | ✘ | ✘ | ✘ | calibrated probabilities | ✔ | claims 35× faster than GPT-6 Sol | AA 54; 83.5% avg on internal 36-bench suite | $0.042 |
| **Mercury Decide** | Inception Labs | early Oct | API (OpenRouter free + paid) | diffusion-LLM | ✔ | ✔ | ✘ | ✘ | choice, score, yes/no | ✔ | 14 decisions/s claimed; p95 >1.3 s reported (HN) | ClaimBench 99.08 (#2) | free tier; paid TBD |
| **Celeris-1 Decision** | Celeris | Oct 2026 (date unstated) | API (`/v1/systemone` at inference.celeris.ai) | diffusion (company thesis; variant unstated) | ✔ | ✔ (billed as input) | ✘ | ✘ | choice/score + probs, free one-sentence explanations | ✔ Brier 0.079 typed (best of 5 hosted) | **67 ms median (1 q)**, 81 ms @50 q | jev-bench 80.0 (Mercury 79.0, Jev 1.1 73.1) | $0.04, output free |
| **SubQ Snap 1.1** | Subquadratic (subq.ai) | Oct 12 (Snap) → Oct 22 (1.1) | API (NanoGPT; OpenRouter listing unverified) | undisclosed | ✔ | ✔ multimodal | ? | ? | choice / rank / yes-no | ✔ | 100K tokens < 1 s (sparse attention, 8–128× compute savings claimed) | **AA Decision Index 65 — tied #1**; When2Call 91.0; BFCL 96.6; RAGTruth 69.0 (Jev 1.14: 48.1) | $0.04, output free; 100M free tokens |
| **Elephant-9B** (elephant-1) | Base Labs | Oct 2026 | **Open MIT** (HF `baselabs/elephant-1`) + Baseten | 9B (sibling mammoth-1 27B) | ✔ | ✔ | ✘ | ✘ | choice, noul, score | ✔ | single forward pass | AA 65.2 — tied top; Jev Decision Index 0.4: 62.6 (top 9B) | hosted pricing n/p |

### The long tail (verified, fewer public details)

- **ScaleDown Decide / Classify-1** — API + OpenRouter, 1M ctx, text+images, $0.04 in / output free.
- **Solar Decide / Flash** (Upstage), **Tev1-4B-Experimental** (Together AI, Qwen3.5-4B, weights + API), **Drex** (Nace.AI, multi-question probabilities in one pass), **Strands Decider** (Amazon Strands Labs, Apache 2.0), **Laya** (Convai, 421M EN / 322M multilingual encoders, 100+ langs), **Kev** (indie, 0.8–27B, Apache 2.0, implements the Jev API), **Semif** (ex-OpenJev, Qwen3.5-4B adapters), **Jeff** (0.8B, home-trainable in ~30 min), plus openjev-sglang / open-jev-fast serving runtimes.
- **GLiNER2.5-Decide** (Fastino, Sep 24, 340M, Apache 2.0) — a decision model built from the NER encoder lineage; runs decisions on CPU, schema-constrained outputs (bridge model; see Cluster B).

### Decision-model notes that matter when choosing

- **Jev**: 64K/request + 32K/question context (262K via a "danger-zone" header, slower/less accurate); no fine-tuning offered, ever; text-only; ~1.05B OpenRouter requests through Oct 11. Param count and architecture still undisclosed.
- **Perplexity Decider** is the surprise open-weights flagship — Qwen3.5-27B base, image input, "Decider for Mac" runs the 4B locally. v1.2 (Oct 16) tops ClaimBench.
- **d1-ultra** beats Jev 1.14 on every AA task at the same price; API-only, ~33K context.
- **Clef-omni** is the only decision model with native audio+video in a single call.
- **Licenses**: Liquid's "open" license restricts free commercial use to <$10M revenue; Clef/Decider/Decision-1/Kev are genuinely Apache/MIT.
- **Ecosystem**: interop = TypeSafe `POST /v1/systemone`; gateways = OpenRouter (usage-ranked decisions page), Vercel AI Gateway, Zuplo, Opper; benchmarks = AA Decision Index v1.0 (AA-Omniscience-Decision, HLE-MC, When2Call, WildGuardTest), Jev Decision Index (HF Space, 30+ open reproductions), JevBench v1.6, ClaimBench (vals.ai). Skepticism exists too: "Jev Is Dead" (Reddit), "most overrated model of 2026" (Medium), Arize's "Can Decision Models Replace LLM Judges?", and TypeSafe's own 1.13-jaggedness doc (accuracy drifts as state grows).

---

## Cluster B — Extraction models (NER / RE / schema→JSON from text)

### B0. Clairic-class — single-pass typed-value extraction (the newest "extractor" category)

System One mechanics applied to extraction: a schema of **typed fields** filled in one non-autoregressive pass, every answer carrying `value` + `confidence` + `p_null`, zero output tokens, input-only billing.

| Model | Org | Released | Open/API | Size | Modality | Field types | Grounding | Confidence | Context | Price |
|---|---|---|---|---|---|---|---|---|---|---|
| **Clairic** | Cloudflare | Oct 19, 2026 (preview) | API (Workers AI `@cf/cloudflare/clairic`); open weights "soon" | undisclosed | text + up to 4 images (PNG/JPEG/WebP) | text (exact spans), number (unit/decimals/min/max), date/time/datetime, duration, list, **estimate (median + p10–p90, all 99 percentiles on request)**, forecast | ✔ char offsets | ✔ + `p_null` (nulls when input doesn't determine the value) | 65,536 | $0.10/1M input, output free |
| **ScaleDown Extract** | ScaleDown | 2026 | API (OpenRouter) | undisclosed small LM | text (+ base64 image/PDF passthrough) | description-driven fields, nested JSON, enum/oneOf | ✔ offsets + context snippet | ✔ | 1M | $0.04/1M in, output free |
| **Schemer** | Desert Ant Labs | 2026 **(date unverified)** | Open weights (custom license), ONNX int4 ~111 MB | 211M | text, 14 languages | typed JSON against caller-supplied schema | ✘ | ? | — | $0 on-device |

Clairic notes: fields beyond an 8,192-token pass are split into parallel passes (usage reports `passes` and zero `output_tokens`); `reference_date` anchors year-less dates; "thirty fields costs about the same as three". Self-reported: TSA forecast within 3.3% avg (17/21 actuals inside the 80% interval); a Clairic→Clef→GLM-5.3 pipeline runs 3.2× faster and ~49% cheaper than all-LLM. Adjacent but not the same thing: NuExtract3 (LLM decoding, B2), GLiNER2.5 encoders (B1), Inference.net's throughput-oriented extraction endpoint on OpenRouter. **The category is thin — Clairic is effectively alone at API scale with typed distributions + `p_null` semantics.**

### B1. Zero-shot NER & IE encoders (the GLiNER lineage)

| Model | Org | Released | Open | Size | Zero-shot NER | Few-shot | Relation extraction | Nested/records | Output | Languages | Benchmark |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **GLiNER2.5** small/base/multi | Fastino | Aug 24, 2026 | Apache 2.0 | 74M/194M/287M | ✔ arbitrary labels | ✔ label descriptions | ✔ joint typed relation graphs (beam-search decoding) | ✔ one schema mixes entities, classes, records, relations, attributes | JSON: char offsets + scores | EN / multi | 16-dataset zero-shot macro-F1 56.17 (multi); Few-NERD 52.37 |
| **GLiNER-bi-large-v2.0** (+base) | Knowledgator | Feb 11, 2026 | Apache 2.0 | 530M / 194M | ✔ **~1M entity labels** (bi-encoder) | ✔ | ✘ (entity linking via GLiNKER instead) | ✘ | spans + offsets + score | EN | CrossNER 61.5 F1; 19-dataset avg 49.7; 130× throughput @1,024 labels |
| **gliner-decoder-large-v1.0** | Knowledgator | Jan 2026 | Apache 2.0 | encoder + 135M decoder | ✔ generates label names — open ontology | ✔ | ✘ (entity linking over huge label sets) | ◐ | spans + generated labels | EN | none published |
| **GLiFormer large v1** | Knowledgator | late 2025 **(date unverified)** | Apache 2.0 | 575.6M | ✔ schema at inference | ✔ | ◐ joint head only — weak (DocRED 12.78) | ✔ **nested Pydantic schemas, zero generated tokens** | validated JSON + offsets + embeddings | EN | CrossNER 64.35; 26-dataset NER avg 50.91; JSON F1 91.10 |
| **GLiNER-Relex large v1.0** | Knowledgator + IDfy | Mar 19, 2026 | Apache 2.0 | ~0.5B (DeBERTa-v3-large + BiLSTM) | ✔ | ✔ | ✔ **joint NER+RE, one pass — the RE flagship** | ✘ | spans + triples + scores | EN | zero-shot RE avg 25.6 micro-F1 (beats GPT-5-mini 22.1) at ~70× speed |
| **GLiNER-Relex multi v1.0** | Knowledgator | Mar 19, 2026 | Apache 2.0 | ~0.3B | ✔ | ✔ | ✔ joint | ✘ | spans + triples + scores | multilingual | none published |
| **OpenBioNER-v2** (base/compact/tiny/deid) | U. Bologna | Mar 9, 2026 | open checkpoints | 110M/65M/15M | ✔ description-conditioned | ✔ up to 5 examples | ✘ | ✘ (no nested spans) | BIO spans + scores | EN (biomedical) | beats GLiNER-459M by +1.6 F1 across 11 biomed sets; 15M tiny beats Llama-3.1-8B by ~8 F1 |
| **JPT** Qwen3-4B/8B | WitnessAI | Apr 2026 (ACL) | demo only — **weights not found (unverified)** | frozen Qwen3 + LoRA | ✔ | ◐ | ✘ | ✘ | token spans (discriminative) | EN | **CrossNER+MIT avg 74.1 F1** — beats UniNER-7B (61.8) and GLiNER-L (60.9); 20× faster than generative NER |

Architectural story of the year: GLiNER2.5 dropped span enumeration for **boundary prediction** (any-length spans, 4,096-token context); Knowledgator went bi-encoder (million-label ontologies) and generative-labels (decoder variant). Pre-window incumbents still in service: GLiNER2 (Jul 2025, ~214k monthly downloads), GLiREL (Jan 2025, zero-shot RE), original NuNER encoders (2024 — **no refresh shipped**).

### B2. LLM-based schema→JSON extraction

| Model | Org | Released | Open | Size | Modality | Arbitrary schema | Few-shot (in-context) | Output | Context | Benchmark / price |
|---|---|---|---|---|---|---|---|---|---|---|
| **NuExtract3** | NuMind | Apr 29 → updated Aug 20, 2026 | Apache 2.0 | 4.5B (Qwen3.5-4B) | text + images + PDF pages | ✔ JSON template + freeform instructions | ✔ | JSON (optional reasoning); ~338 thinking tokens vs ~6,552 for Qwen3.6-27B | 131K | **ExtractBench 82.35** (Short 88.06) |
| **NuExtract-2.0-8B** | NuMind | May 2025, major update May 2026 | MIT | 8.3B (Qwen2.5-VL-7B) | text + images | ✔ | ✔ | JSON, verbatim-string/enum/date types, null-when-absent | — | ExtractBench 49.42 |
| **ScaleDown Extract** | ScaleDown | 2026 **(date unverified)** | API (OpenRouter) | undisclosed small LM | text (+ base64 image/PDF passthrough) | ✔ description-driven fields, nested schema, enum/oneOf constrained decoding | ◐ | **JSON + confidence + char offsets + context snippet** | 1M | $0.04/1M in, output free; no official benchmarks (third-party CUAD blog only) |
| Schematron V2 Small/Turbo | Inference.net | Apr 16, 2026 | API (V1 3B/8B open) | ~3B class | text (HTML→JSON web scraping) | ✔ strict JSON from Pydantic/Zod | ◐ | JSON | 128K | beats DeepSeek V3.2 + GPT-5.4-nano on judged eval; Turbo $0.06/1M |

### B3. Confirmed dead zones (searched hard, nothing shipped)

- **Event extraction / event-argument extraction:** no new model Oct 2025–Oct 2026. EACL/TACL/arXiv papers exist (boundary-aware augmentation, "Are Triggers Needed?", schema-constrained EAE) with no released checkpoints. **ADELIE (THU-KEG, EMNLP 2024) remains the last instruction-tuned IE-LLM covering events** — no update to its HF collection.
- **Open IE:** nothing new; practice has shifted to LLM prompting for triples.
- **UIE / InstructUIE / GoLLIE successors:** none shipped; the niche is now occupied by GLiNER-family schema-conditioned encoders.
- **Multimodal NER (entities from images/video):** academic papers and surveys only — no shipped model from any org.
- **Domain NER beyond biomedical:** only John Snow Labs' supervised fixed-schema clinical ONNX models (EN/IT/ES); no notable new legal/financial zero-shot model.
- **Supervised RE SOTA:** 2025–26 movement is methods papers (RAG4RE, DOREMI, ATGL), not released checkpoints.

### B4. Extraction benchmarks in 2026

CrossNER remains the zero-shot NER leaderboard (JPT 74.1 > GLiFormer 64.35 > gliner-bi-v2 61.5); papers increasingly pair it with MIT-Movie/Restaurant or 19–20-dataset suites, and cherry-pick. **Universal NER v2** (LREC 2026) added 11 gold datasets across 10 languages; MultiClinNER (SMM4H 2026) covers clinical multilingual. NuNER and Few-NERD are inconsistently reported (JPT skips both). For schema-JSON extraction: NuMind's ExtractBench and Interfaze's value-accuracy SOB leaderboard (schema compliance ≠ value correctness).

---

## The map — unified capability matrix

Legend: ✔ ◐ partial ✘ no ? unknown

| Model | Category | NER (zero-shot) | Relation extr. | Event extr. | Arbitrary JSON schema | Char offsets / grounding | Calibrated confidence | Text | Image | Video | Audio | Open weights | Size |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jev 1.14 | decision | ✘ (bounded fields only) | ✘ | ✘ | ◐ typed answers | ✘ | ✔ | ✔ | ✘ | ✘ | ✘ | ✘ | ? |
| Perplexity Decider v1.2 | decision | ✘ | ✘ | ✘ | ◐ choice/probability | ✘ | ✔ | ✔ | ✔ | ✘ | ✘ | ✔ Apache | 27B / 4B |
| d1-ultra / d1-3B | decision | ✘ | ✘ | ✘ | ◐ typed answers | ✘ | ✔ | ✔ | ✔ | omni ✘ | omni ✔ | 3B ✔ (rev-capped lic.) / ultra ✘ | 3.1B / ? |
| OpenAI Decisions | decision | ◐ "extraction for agents" | ✘ | ✘ | ✔ predicate/choice/score | ✘ | ✔ | ✔ | ✔ | ✘ | ✘ | ✘ | ? |
| Clef / Clef-omni | decision | ✘ | ✘ | ✘ | ◐ scored decisions | ✘ | ✔ | ✔ | ✔ | omni ✔ | omni ✔ | ✔ | 9–30B |
| Microsoft Decision-1 | decision | ✘ | ✘ | ✘ | ◐ fixed options | ✘ | ✔ | ✔ | ✘ | ✘ | ✘ | ✔ MIT | 9B |
| SubQ Snap 1.1 | decision | ✘ | ✘ | ✘ | ◐ | ✘ | ✔ | ✔ | ✔ | ✔ (multimodal) | ? | ? | ✘ API (384K ctx) | ? |
| Elephant-9B | decision | ✘ | ✘ | ✘ | ◐ | ✘ | ✔ | ✔ | ✔ | ✔ | ✘ | ✘ | ✔ MIT | 9B |
| GLiNER2.5 (multi) | extraction | ✔ | ✔ | ✘ | ◐ records in one schema | ✔ | ◐ scores | ✔ | ✘ | ✘ | ✘ | ✔ Apache | 287M |
| GLiNER-bi-v2.0 | extraction | ✔ (1M labels) | ✘ | ✘ | ✘ | ✔ | ◐ | ✔ | ✘ | ✘ | ✘ | ✔ Apache | 530M |
| gliner-decoder v1.0 | extraction | ✔ (open ontology) | ✘ | ✘ | ◐ generated labels | ✔ | ◐ | ✔ | ✘ | ✘ | ✘ | ✔ Apache | ~0.5B |
| GLiFormer large v1 | extraction | ✔ | ◐ weak | ✘ | ✔ nested Pydantic, 0 tokens out | ✔ | ◐ | ✔ | ✘ | ✘ | ✘ | ✔ Apache | 576M |
| GLiNER-Relex v1.0 | extraction | ✔ | ✔ joint | ✘ | ✘ | ✔ | ◐ | ✔ (multi: multiling.) | ✘ | ✘ | ✘ | ✔ Apache | 0.3–0.5B |
| OpenBioNER-v2 | extraction | ✔ (biomed) | ✘ | ✘ | ✘ | ✔ (BIO) | ◐ | ✔ | ✘ | ✘ | ✘ | ✔ | 15–110M |
| JPT | extraction | ✔ SOTA | ✘ | ✘ | ✘ | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ | ? (demo only) | Qwen3-4B/8B + LoRA |
| NuExtract3 | extraction (LLM) | ✔ via schema | ◐ via schema | ✘ | ✔ | ◐ | ✘ | ✔ | ✔ | ✘ | ✘ | ✔ Apache | 4.5B |
| ScaleDown Extract | extraction (LLM) | ✔ via schema | ◐ | ✘ | ✔ nested + enum/oneOf | ✔ + context | ✔ | ✔ | ◐ passthrough | ✘ | ✘ | ✘ API | ? small |
| Clairic | extractor (typed values) | ◐ text-span fields | ✘ | ✘ | ✔ typed field schema | ✔ char offsets | ✔ + p_null | ✔ | ✔ (4 imgs) | ✘ | ✘ | ✘ API (weights "soon") | ? |
| Schematron V2 | extraction (LLM) | ✔ via schema | ◐ | ✘ | ✔ strict (Pydantic/Zod) | ✘ | ✘ | ✔ (HTML) | ✘ | ✘ | ✘ | ✘ API (V1 open) | ~3B |

## Quick lookup — "who can do X"

- **Zero-shot NER with arbitrary labels, cheapest/fastest:** GLiNER2.5-small (74M) → multi (287M); OpenBioNER-v2 for biomedical; JPT if you need max accuracy and can wait on weights.
- **Relation extraction:** GLiNER-Relex (only dedicated 2026 release; joint with NER); GLiNER2.5 (joint graph decoding); GLiREL (2025 incumbent).
- **Event extraction:** nothing new — ADELIE (2024) or prompt a general LLM.
- **Million-label ontologies / entity linking at scale:** GLiNER-bi-v2.0 + GLiNKER; gliner-decoder for open ontologies.
- **Nested/deep JSON schemas without generation latency:** GLiFormer (576M, validated JSON, zero output tokens).
- **Schema→JSON with an LLM (complex instructions, long docs):** NuExtract3 (open) > NuExtract-2.0-8B; ScaleDown Extract (hosted, $0.04/1M, offsets+confidence); Schematron V2 (HTML).
- **Grounded output (char offsets + confidence):** Clairic (offsets + confidence + `p_null`), ScaleDown Extract, GLiNER family (offsets + scores), NuExtract3 (◐).
- **Typed values with distributions (numbers/dates/estimates):** Clairic — alone: estimate fields return median + p10–p90 (all 99 percentiles on request).
- **Decisions with calibrated probabilities:** d1-ultra / SubQ Snap 1.1 / Elephant-9B (AA index ~65, three-way tie) > Claude Haiku 5.5-Low (61, but 11× slower) ≈ Jev 1.14 (58); GLiDE (best When2Call/WildGuard mix); Celeris-1 (fastest, 67 ms, best typed Brier); GLiNER2.5-Decide (CPU, air-gapped, 340M).
- **Decisions over images/video/audio:** Clef-omni (all three), Perplexity Decider 27B (image), d1-omni-600M (audio, open).
- **Fully self-hostable stack:** GLiNER2.5 + GLiNER-Relex + GLiFormer (Apache) for extraction; Perplexity Decider (Apache), Microsoft Decision-1 (MIT), Clef (open), Kev (Apache) for decisions. Watch Liquid's revenue-capped license.

## Trends & gaps

1. **Decision↔extraction convergence is the story to watch.** ScaleDown sells both endpoints with identical pricing; Fastino's encoder lineage now does both; OpenAI markets Decisions for extraction-for-agents; ScaleDown Extract already returns confidence + offsets like a decision model — and **Clairic is the convergence made explicit**: typed-value extraction on System One mechanics, from the same org as Clef. Expect schema-constrained single-pass extraction (GLiFormer-style) to absorb small-field decision workloads, and vice versa.
2. **Encoders reclaimed zero-shot IE from LLMs** — 74M–576M Apache models beat or match 7–8B LLMs on NER at 20–70× the speed, and the only LLM survivors are purpose fine-tunes (NuExtract3).
3. **Benchmark fragmentation on both sides:** decision models have five+ leaderboards (AA, Jev Index, JevBench, ClaimBench, Decision Bench) that disagree; NER papers cherry-pick suites. Independent numbers (AA, vals.ai) are the only cross-vendor currency.
4. **Calibration is the open wound:** AA shows every decision model overconfident; extraction models rarely publish calibration at all (ScaleDown Extract is the exception exposing it).
5. **Underserved:** event extraction, Open IE, multimodal NER, relation extraction at document level (DocRED-level scores remain dismal: 12–31 F1), and Jev's undisclosed architecture/size.

## Sources

**Decision models:** docs.typesafe.ai/models · artificialanalysis.ai/articles/artificial-analysis-decision-index · forbes.com (Oct 12, 2026 decision-models guide) · openai.com/index/decisions-api + OpenAI community GA post · blog.cloudflare.com (clef-decision-models, clef-omni) · liquid.ai/News/Models + activepieces.com (d1-ultra) · huggingface.co/perplexity-ai/pplx-decider-v1.1-27b + community.perplexity.ai + vals.ai · fastino.ai (GLiDE) · commandline.microsoft.com (Decision-1) · openrouter.ai (rankings/decisions, mercury-decide, classify-1, microsoft-decision-1) · baseten.co/library/mercury-decide · celeris.ai · github.com (strands-decider, kev, semif, typesafe-jev-ecosystem, awesome-jev-alternatives) · HF Spaces (jev-decision-index, JevBench) · xhinker.medium.com (Jev critique) · opper.ai/jevman-benchmark · zuplo.com changelog · langchain.com (ScaleDown deep agents)

**Late-October arrivals & Clairic-class:** subq.ai (SubQ Snap 1.1) · baseten.co/library/elephant-1 (Elephant-9B, HF baselabs/elephant-1) · celeris.ai/decisions + celeris.ai/pricing · developers.cloudflare.com/workers-ai/models/clairic · blog.cloudflare.com/clairic-extraction-model · developers.cloudflare.com/changelog/post/2026-10-19-clairic-workers-ai · openrouter.ai/models + openrouter.ai/collections/discounted-models (ScaleDown Extract, Inference.net extraction) · huggingface.co/desert-ant-labs/schemer

**NER / IE models:** huggingface.co/fastino (gliner2.5-multi-v1, GLiNER2.5-Decide, gliner2-large-v1) · fastino.ai/blog (gliner2-5, decide) · marktechpost.com (GLiNER2.5) · huggingface.co/knowledgator (gliner-bi-large-v2.0, gliner-decoder-large-v1.0, gliner-relex-large-v1.0, gliner-relex-multi-v1.0, gliformer-large-v1) · arxiv.org/abs/2602.18487 (gliner-bi) · arxiv.org/html/2605.10108v1 (GLiNER-Relex) · github.com/urchade/GLiNER/releases · huggingface.co/blog/alecocc/openbioner-v2 + sciencedirect.com (paper) · arxiv.org/html/2604.05158v1 + witness-ai-jpt-ner.hf.space (JPT) · arxiv.org/abs/2604.12744 + lrec-conf.org (UNER v2) · aclanthology.org/2026.smm4h-1.19 (MultiClinNER) · johnsnowlabs.com (clinical ONNX NER) · huggingface.co/numind (NuExtract3, NuExtract-2.0-8B) · openrouter.ai/scaledown/extract + docs.scaledown.ai · inference.net/blog/schematron-v2 · github.com/jackboyla/GLiREL · huggingface.co/THU-KEG/ADELIE-SFT · developers.googleblog.com (LangExtract) · interfaze.ai (SOB)
