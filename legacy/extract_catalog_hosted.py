"""One-off extractor: catalog/models/<org>/<model>.yaml for every system whose
org has a HOSTED presence (phase-1 spine, docs/plan/05-migration.md).

Run from the repo root:

    .venv/Scripts/python.exe legacy/extract_catalog_hosted.py

Sources (engines code wins for technical facts, README for license/links/size):
- README.md                       system table + benchmark sections
- legacy/engines/openrouter_client.py   OPENROUTER_SYSTEMONE + rerank registry
- legacy/engines/{jev,clef,fastino,typellm}_client.py   endpoints, model ids, pins
- legacy/bench_spectrum.py, legacy/bench_multilingual.py   ALL_SYSTEMS + CULLED
- legacy/results/*.json           ground truth for v0.benched (membership = benched)

Scope: this script owns the org dirs of every org with a hosted/metered/cloud
system in the README system table, PLUS the whole org dir of mixed orgs that
also have local-only systems (fastino GLiNER checkpoints, liquid d1 pair,
jaredpalmer kev-0.8b). Local-only orgs (knowledgator, jackboyla, relik-ie,
mixedbread-ai, baai, alibaba-nlp, nvidia, convaiinnovations, wfzyx, mapika,
iapp, alibiserikbay, molemo-lab, frontiersmind, internlm, ifm, vllm-sr, madani,
logitlab, codekins, txoka, eldanring, moritzlaurer, ...) belong to the other
extraction pass and are NOT written here.

File shape follows docs/plan/04-data-model.md and catalog/schema/model.json.
The transitional `v0: {benched: ...}` flag is true iff any of the v0 system
names mapped to the file appears in legacy/results/*.json (dropped later).
"""

from __future__ import annotations

import json
import os
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO, "catalog", "models")

# ---------------------------------------------------------------------------
# v0 benched set: every system name that appears in any legacy results file.
# ---------------------------------------------------------------------------


def load_benched_names() -> set:
    """System names from legacy/results/*.json (spectrum `systems` map,
    multilingual `by_language` map, flat-suite classification/ner maps)."""
    results_dir = os.path.join(REPO, "legacy", "results")
    names: set = set()

    def collect(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, dict):
                    names.add(key)
                    collect(value)

    spectrum = os.path.join(results_dir, "bench_spectrum_results.json")
    if os.path.exists(spectrum):
        with open(spectrum, encoding="utf-8") as fh:
            data = json.load(fh)
        names.update(data.get("systems", {}))

    multilingual = os.path.join(results_dir, "bench_multilingual_results.json")
    if os.path.exists(multilingual):
        with open(multilingual, encoding="utf-8") as fh:
            data = json.load(fh)
        names.update(data.get("by_language", {}))

    flat = os.path.join(results_dir, "bench_results.json")
    if os.path.exists(flat):
        with open(flat, encoding="utf-8") as fh:
            data = json.load(fh)
        collect(data.get("classification", {}))
        collect(data.get("ner", {}))
    return names


# ---------------------------------------------------------------------------
# Small builders (keep the ~40 entries below readable).
# ---------------------------------------------------------------------------


def version(vid, api_slug=None, hf_revision=None, released=None):
    return {"id": vid, "hf_revision": hf_revision, "api_slug": api_slug,
            "released": released}


def hosted(dep_id, ver, provider, endpoint, adapter, notes, **extra):
    dep = {"id": dep_id, "kind": "hosted", "version": ver,
           "provider": provider, "endpoint": endpoint, "adapter": adapter}
    if extra:
        dep.update(extra)
    if notes:
        dep["notes"] = notes
    return dep


def local(dep_id, ver, runner, notes, engine=None, dtype=None, quant=None,
          device="cpu", batch=None, vram=None, ram=None, accel=None, **extra):
    dep = {"id": dep_id, "kind": "local", "version": ver, "runner": runner,
           "runtime": {"engine": engine, "dtype": dtype, "quant": quant,
                       "device": device, "batch": batch},
           "requires": {"vram_gb": vram, "ram_gb": ram, "accel": accel}}
    if extra:
        dep.update(extra)
    if notes:
        dep["notes"] = notes
    return dep


def model(mid, family, org, org_dir, license_, open_weights, params_b, claims,
          links, versions, deployments, v0_names, notes=None, status="active"):
    return {
        "id": mid,
        "family": family,
        "org": org,
        "_org_dir": org_dir,
        "license": license_,
        "open_weights": open_weights,
        "params_b": params_b,
        "claims": claims,
        "status": status,
        "links": links,
        "versions": versions,
        "deployments": deployments,
        "notes": notes,
        "_v0_names": v0_names,
    }


def claims(tracks, modalities, languages):
    return {"tracks": tracks, "modalities": modalities, "languages": languages}


# ---------------------------------------------------------------------------
# The catalog. One entry per MODEL file. Mixed orgs (fastino, liquid,
# jaredpalmer) include their local-only systems; the three Fastino hosted
# GLiNER twins fold into their local checkpoint's file as a second deployment.
# ---------------------------------------------------------------------------

LAN_EN_NOTE = ("card language claim not recorded in v0; English is the only "
               "measured baseline (see the multilingual suite for measured "
               "reach)")

MODELS = [
    # ------------------------------------------------------------- typesafe
    model(
        "typesafe/jev", "jev", "TypeSafe", "typesafe",
        "proprietary", False, None,
        claims(["decision", "score"], ["text", "image"], "en"),
        {"card": "https://www.typesafe.ai/", "paper": None},
        [version("jev-1.13.0", api_slug="jev-1.13.0"),
         version("jev-1.14.0", api_slug="jev-1.14.0", released="2026-10-14")],
        [hosted(
            "jev.typesafe-systemone", "jev-1.14.0", "typesafe",
            "https://api.typesafe.ai/v1/systemone", "typesafe.systemone",
            "POST {model, state, questions} (Bearer TYPESAFE_API_KEY), the "
            "TypeSafe System One wire format (@typesafe-ai/sdk 0.6.x). Pinned "
            "to jev-1.14.0 because the jev-latest alias flips with releases "
            "(~2026-10-21 for 1.14). Context ceilings 64k per request / 32k "
            "per question; 262k behind the TypeSafe-Danger-Zone-Long-Context "
            "header; image assets + 1.14 box questions behind "
            "TypeSafe-Danger-Zone-Images (benched text-only). Provider "
            "reports tokens but no $ - v0 cost derived: $5.3e-6/q at "
            "$0.042/M input.")],
        ["Jev"],
        notes="1.13 and 1.14 both record 45/48 on the mixed pool (they trade "
              "exactly one question each way); 1.13 is kept as a version for "
              "the recorded history. claims.languages is the v0-measured "
              "baseline only - Jev measured 100% on the 9-language suite.",
    ),

    # ---------------------------------------------------------- jaredpalmer
    model(
        "jaredpalmer/kev-0.8b", "kev", "jaredpalmer", "jaredpalmer",
        "Apache-2.0", True, 0.8,
        claims(["decision", "score"], ["text"], "en"),
        {"card": "https://github.com/jaredpalmer/kev", "paper": None},
        [version("kev-0.8b@2026-09")],
        [local(
            "kev-0.8b.local-systemone", "kev-0.8b@2026-09", "systemone-local",
            "Open-weight Jev lookalike (Qwen3.5 base + LoRA/head, 9.3M "
            "trained) serving the System One contract on 127.0.0.1:8009 "
            "(uv run --extra serve --project ../kev python -m kev.serve --run "
            "jaredpalmer/kev-0.8b); needs transformers >=5.17, so it runs "
            "from its own clone. 4B/9B siblings exist but are impractical on "
            "CPU. ram_gb estimated (0.8B fp32, no overhead).",
            engine="torch", ram=3.2)],
        ["Kev 0.8B (local)"],
        notes="ram_gb is estimated from the 0.8B size (4 bytes/param fp32, "
              "no overhead).",
    ),
    model(
        "jaredpalmer/kev-4b", "kev", "jaredpalmer", "jaredpalmer",
        "proprietary", False, 4,
        claims(["decision", "score"], ["text"], "en"),
        {"card": "https://openrouter.ai/jaredpalmer/kev-4b", "paper": None},
        [version("kev-4b@2026-09", api_slug="jaredpalmer/kev-4b")],
        [hosted(
            "kev-4b.openrouter", "kev-4b@2026-09", "openrouter",
            "jaredpalmer/kev-4b", "openrouter.systemone",
            "Same System One contract as local Kev, hosted (OpenRouter "
            "/api/v1/systemone, Bearer OPENROUTER_API_KEY). Non-ZDR (†). "
            "Batches each 24-question task into one request; measured "
            "$1.5e-6/q.")],
        ["Kev 4B (OpenRouter)"],
        notes=None,
    ),

    # --------------------------------------------------------------- upstage
    model(
        "upstage/solar-decide", "solar-decide", "Upstage", "upstage",
        "proprietary", False, 35,
        claims(["decision", "score"], ["text"], ["en", "ko", "ja"]),
        {"card": "https://openrouter.ai/upstage/solar-decide", "paper": None},
        [version("solar-decide@2026-09", api_slug="upstage/solar-decide")],
        [hosted(
            "solar-decide.openrouter", "solar-decide@2026-09", "openrouter",
            "upstage/solar-decide", "openrouter.systemone",
            "Upstage System One endpoint on Solar Mini 4 (35B MoE, 3B "
            "active): choice/score/noul with calibrated probabilities, no "
            "prose, output tokens free, 524K context. Non-ZDR (†) - default "
            "routing; a ZDR Upstage endpoint exists on OpenRouter. Batches "
            "each 24-question task into one request; Solar's tokenizer runs "
            "long on English (~8.7k tokens per batched request), landing at "
            "$1.8e-5/q measured.")],
        ["Solar Decide (OpenRouter)"],
        notes="params_b is the Solar Mini 4 MoE total (3B active). "
              "claims.languages: the Solar Mini 4 backbone is built "
              "Korean/English/Japanese-first (README).",
    ),
    model(
        "upstage/solar-decide-flash", "solar-decide", "Upstage", "upstage",
        "proprietary", False, None,
        claims(["decision", "score"], ["text"], ["en", "ko", "ja"]),
        {"card": "https://openrouter.ai/upstage/solar-decide-flash",
         "paper": None},
        [version("solar-decide-flash@2026-10",
                 api_slug="upstage/solar-decide-flash")],
        [hosted(
            "solar-decide-flash.openrouter", "solar-decide-flash@2026-10",
            "openrouter", "upstage/solar-decide-flash", "openrouter.systemone",
            "Solar Decide's low-latency sibling on the same Upstage System "
            "One endpoint and Solar Mini 4 base: same choice/score/noul "
            "contract, output tokens free; 4.5x faster than Solar Decide at "
            "the identical $0.05/M input ($0.00087 per 48-question run). "
            "Non-ZDR (†) - default routing.")]
        ,
        ["Solar Decide Flash (OpenRouter)"],
        notes="Size undisclosed on the card (Solar Mini 4 base) - params_b "
              "null. claims.languages: Solar Mini 4 backbone, built "
              "Korean/English/Japanese-first (README).",
    ),

    # ------------------------------------------------------------- cloudflare
    model(
        "cloudflare/clef", "clef", "Cloudflare", "cloudflare",
        "Apache-2.0 (weights on HF; metered API)", True, 27,
        claims(["decision", "score"], ["text", "image"], "en"),
        {"card": "https://developers.cloudflare.com/workers-ai/models/clef/",
         "paper": None},
        [version("clef@2026-10", api_slug="@cf/cloudflare/clef")],
        [hosted(
            "clef.workers-ai", "clef@2026-10", "cloudflare-workers-ai",
            "@cf/cloudflare/clef", "workers-ai.clef",
            "POST https://api.cloudflare.com/client/v4/accounts/<account>/ai/"
            "run/@cf/cloudflare/clef - the System One wire format behind "
            "Cloudflare's REST v4 envelope ({success, result}) (cf CLI OAuth "
            "session or CLOUDFLARE_AUTH_TOKEN + CLOUDFLARE_ACCOUNT_ID). "
            "Frozen Qwen3.8-27B backbone, 64k ctx, vision-capable. ZDR: "
            "Cloudflare commits to not reading/storing/training on Clef "
            "requests. Reports tokens but no $ - derived $2.3e-5/q at "
            "$0.24/M input.")],
        ["Clef (Workers AI)"],
        notes="Weights are Apache 2.0 on HF, so the system is locally "
              "runnable in principle; no local engine or runner island "
              "exists in v0. claims.languages is the v0-measured baseline "
              "(Clef measured 100% on the 9-language suite).",
    ),
    model(
        "cloudflare/clef-flash", "clef", "Cloudflare", "cloudflare",
        "Apache-2.0 (weights on HF; metered API)", True, 9,
        claims(["decision", "score"], ["text"], "en"),
        {"card":
         "https://developers.cloudflare.com/workers-ai/models/clef-flash/",
         "paper": None},
        [version("clef-flash@2026-10", api_slug="@cf/cloudflare/clef-flash")],
        [hosted(
            "clef-flash.workers-ai", "clef-flash@2026-10",
            "cloudflare-workers-ai", "@cf/cloudflare/clef-flash",
            "workers-ai.clef",
            "Same System One contract behind the v4 envelope; frozen "
            "Qwen3.5-9B backbone (~39 ms median, self-reported). ZDR. "
            "Derived $8.6e-6/q at $0.09/M input.")],
        ["Clef-flash (Workers AI)"],
        notes="Weights are Apache 2.0 on HF; no local engine or runner "
              "island exists in v0.",
    ),
    model(
        "cloudflare/clef-omni", "clef", "Cloudflare", "cloudflare",
        "Apache-2.0 (weights on HF; metered API)", True, 30,
        claims(["decision", "score"], ["text", "image", "audio", "video"],
               "en"),
        {"card":
         "https://developers.cloudflare.com/workers-ai/models/clef-omni/",
         "paper": None},
        [version("clef-omni@2026-10", api_slug="@cf/cloudflare/clef-omni",
                 released="2026-10-09")],
        [hosted(
            "clef-omni.workers-ai", "clef-omni@2026-10",
            "cloudflare-workers-ai", "@cf/cloudflare/clef-omni",
            "workers-ai.clef",
            "Clef's MoE member: Qwen3-Omni-30B-A3B fine-tune (30B total / 3B "
            "active) on the same System One questions, plus audio[]/videos[] "
            "state inputs alongside the Clef images[] extension (benches "
            "ride the text path). ZDR. Derived $1.4e-5/q at $0.15/M input.")],
        ["Clef-omni (Workers AI)"],
        notes="Weights are Apache 2.0 on HF, but ~60 GB bf16 towers past the "
              "v0 CPU box - the local twin was never run. claims.languages "
              "is the v0-measured baseline (98% on the 9-language suite).",
    ),

    # ---------------------------------------------------------------- fastino
    model(
        "fastino/glide", "glide", "Fastino", "fastino",
        "metered API (research tier)", False, None,
        claims(["decision", "score"], ["text"], "en"),
        {"card": "https://fastino.ai/", "paper": None},
        [version("glide@2026-10", api_slug="fastino/GLiDE",
                 released="2026-09-30")],
        [hosted(
            "glide.fastino-systemone", "glide@2026-10", "fastino",
            "fastino/GLiDE", "fastino.systemone",
            "POST https://api.fastino.ai/v1/systemone (FASTINO_API_KEY) - "
            "TypeSafe's System One wire format behind Fastino's REST, flat "
            "payload. One fast scoring pass plus adaptive reasoning when the "
            "leading option is uncertain; thinking output tokens are priced "
            "$0, only input bills ($0.15/M). 40k context, served on one "
            "B200; Fastino's model catalog flags it ZDR. Measured $1.2e-5/q "
            "(derived from ~1.9k input tokens per batched 24-question "
            "request).")],
        ["GLiDE (Fastino)"],
        notes="No open-weights release recorded in v0 - open_weights false. "
              "Fastino markets GLiDE as an English-first Decision Index "
              "model, hence claims.languages 'en' (it measured 100% on the "
              "9-language suite anyway).",
    ),
    model(
        "fastino/gliner2.5-small", "gliner", "Fastino", "fastino",
        "Apache-2.0", True, 0.074,
        claims(["ner", "classify", "relation"], ["text"], "en"),
        {"card": "https://github.com/fastino-ai/GLiNER2", "paper": None},
        [version("gliner2.5-small@2026-09")],
        [local(
            "gliner2.5-small.local", "gliner2.5-small@2026-09", "gliner",
            "fastino/gliner2.5-small-v1 (DeBERTa-v3-xsmall) via the gliner2 "
            "AutoExtractor, transformers 4.57.x pin. Fastest local system "
            "in v0 (43 ms/question). ram_gb estimated (74M fp32, no "
            "overhead). No hosted twin: Fastino deliberately does not host "
            "the small checkpoint (404 'not a recognised model id' on the "
            "chat-completions endpoint).",
            engine="torch", ram=0.3)],
        ["GLiNER2.5-small"],
        notes="HF repo fastino/gliner2.5-small-v1 (in links.card). ram_gb is "
              "estimated from the 74M size.",
    ),
    model(
        "fastino/gliner2.5-base", "gliner", "Fastino", "fastino",
        "Apache-2.0", True, 0.194,
        claims(["ner", "classify", "relation"], ["text"], "en"),
        {"card": "https://github.com/fastino-ai/GLiNER2", "paper": None},
        [version("gliner2.5-base@2026-09")],
        [local(
            "gliner2.5-base.local", "gliner2.5-base@2026-09", "gliner",
            "fastino/gliner2.5-base-v1 (default English checkpoint) via the "
            "gliner2 AutoExtractor, transformers 4.57.x pin. ram_gb "
            "estimated (194M fp32, no overhead).",
            engine="torch", ram=0.8),
         hosted(
            "gliner2.5-base.fastino-chat", "gliner2.5-base@2026-09",
            "fastino", "fastino/gliner2.5-base-v1", "fastino.chat",
            "Hosted twin of the local base checkpoint: POST "
            "https://api.fastino.ai/v1/chat/completions with a top-level "
            "schema body (entities/classifications), one request per "
            "question. $0.03/M input, output $0; measured $3.9e-7/q - the "
            "cheapest metered row in v0. Classification answers "
            "bit-identical to the local run; NER differs on one question "
            "(serving-stack wobble, not a weight change). ZDR catalog.")],
        ["GLiNER2.5-base", "GLiNER2.5-base (Fastino)"],
        notes="Two v0 systems, one model: the Fastino row is the hosted twin "
              "of the same open-weight checkpoint. ram_gb is estimated from "
              "the 194M size.",
    ),
    model(
        "fastino/gliner2.5-multi", "gliner", "Fastino", "fastino",
        "Apache-2.0", True, 0.287,
        claims(["ner", "classify", "relation"], ["text"], "multilingual"),
        {"card": "https://github.com/fastino-ai/GLiNER2", "paper": None},
        [version("gliner2.5-multi@2026-09")],
        [local(
            "gliner2.5-multi.local", "gliner2.5-multi@2026-09", "gliner",
            "fastino/gliner2.5-multi-v1 (mDeBERTa, multilingual - the "
            "largest 2.5; no 'large' exists) via the gliner2 AutoExtractor, "
            "transformers 4.57.x pin. Most robust local extractor in v0. "
            "ram_gb estimated (287M fp32, no overhead).",
            engine="torch", ram=1.2),
         hosted(
            "gliner2.5-multi.fastino-chat", "gliner2.5-multi@2026-09",
            "fastino", "fastino/gliner2.5-multi-v1", "fastino.chat",
            "Hosted twin of the local multi checkpoint over the "
            "chat-completions schema endpoint; $0.03/M input, output $0; "
            "measured $3.9e-7/q. Repeats the local run's answers exactly on "
            "classification and across all nine languages. ZDR catalog.")],
        ["GLiNER2.5-multi", "GLiNER2.5-multi (Fastino)"],
        notes="Two v0 systems, one model (hosted twin of the same "
              "checkpoint). ram_gb is estimated from the 287M size.",
    ),
    model(
        "fastino/gliner2.5-decide", "gliner", "Fastino", "fastino",
        "Apache-2.0", True, 0.34,
        claims(["classify", "ner", "relation"], ["text"], "en"),
        {"card": "https://huggingface.co/fastino/GLiNER2.5-Decide",
         "paper": None},
        [version("gliner2.5-decide@2026-09")],
        [local(
            "gliner2.5-decide.local", "gliner2.5-decide@2026-09", "gliner",
            "fastino/GLiNER2.5-Decide - the decision-tuned sibling "
            "(classification primary, extraction still supported) via the "
            "gliner2 AutoExtractor, transformers 4.57.x pin. Best local "
            "classifier on the v0 mixed pool and best local NER-capable "
            "classifier. ram_gb estimated (340M fp32, no overhead).",
            engine="torch", ram=1.4),
         hosted(
            "gliner2.5-decide.fastino-chat", "gliner2.5-decide@2026-09",
            "fastino", "fastino/GLiNER-2.5-Decide", "fastino.chat",
            "Hosted twin - note the hosted id is hyphenated "
            "(fastino/GLiNER-2.5-Decide) where the HF repo is not. Rides the "
            "chat-completions schema, NOT /v1/systemone, despite being a "
            "decision model (docs.fastino.ai/concepts/gliner-2-5-decide). "
            "$0.03/M input, output $0; measured $3.9e-7/q. ZDR catalog.")],
        ["GLiNER2.5-Decide", "GLiNER2.5-Decide (Fastino)"],
        notes="Two v0 systems, one model (hosted twin). ram_gb is estimated "
              "from the 340M size.",
    ),

    # ---------------------------------------------------------------- typellm
    model(
        "typellm/typellm-latest", "typellm", "TypeLLM", "typellm",
        "Apache-2.0 (harness); hosted API proprietary", False, 27,
        claims(["decision", "score"], ["text", "image"], "en"),
        {"card": "https://typellm.ai/", "paper": None},
        [version("typellm-latest@2026-10", api_slug="typellm-latest")],
        [hosted(
            "typellm-latest.generate", "typellm-latest@2026-10", "typellm",
            "typellm-latest", "typellm.generate",
            "POST https://api.typellm.ai/v1/generate (TYPELLM_API_KEY): "
            "{context, questions, images?, model?} - JSON-Schema fields "
            "answered by one-token constrained enum picks (up to 512 values "
            "since v0.6.9; up to 64 questions and 8 base64 images per call) "
            "over SGLang on their hosted Qwen3.8-27B. The October releases "
            "also expose Jev-compatible /v1/systemone and "
            "OpenAI-Decisions-compatible /v1/decisions (each billed as a "
            "generate call) - the benches use /v1/generate. Input $0.05/M, "
            "answer tokens free; measured $2.6e-6/q plain. No published "
            "data-retention commitment (†).")]
        ,
        ["TypeLLM (hosted)"],
        notes="Version id dated to the 2026-10-17 re-bench after the "
              "v0.6.x -> v0.7.x releases (512-value enums, described "
              "choices, box detection, the Jev-compatible /v1/systemone); "
              "no single release date, so 'released' is null. claims."
              "languages is the v0-measured baseline (100% on the "
              "9-language suite).",
    ),
    model(
        "typellm/typellm-thinking", "typellm", "TypeLLM", "typellm",
        "Apache-2.0 (harness); hosted API proprietary", False, 27,
        claims(["decision", "score"], ["text", "image"], "en"),
        {"card": "https://typellm.ai/", "paper": None},
        [version("typellm-latest@2026-10", api_slug="typellm-latest")],
        [hosted(
            "typellm-thinking.generate", "typellm-latest@2026-10", "typellm",
            "typellm-latest", "typellm.generate",
            "Same model, endpoint and adapter as typellm/typellm-latest with "
            "per-question thinking enabled (thinking: true, thinking_budget "
            "capped 4096): reasoning tokens bill at their own $0.50/M on top "
            "of input's $0.05/M, answers free; measured $6.1e-5/q after the "
            "2026-10-17 retune (reasoning shrank 9,376 -> 5,530 tokens and "
            "accuracy dropped 97.9% -> 91.7% - plain is now both cheaper and "
            "more accurate on the pool). Split into its own file so each "
            "benched v0 system is one catalog row. No published "
            "data-retention commitment (†).")]
        ,
        ["TypeLLM thinking (hosted)"],
        notes="Same version as typellm/typellm-latest - the thinking mode is "
              "a request parameter, not a distinct version or endpoint.",
    ),

    # ----------------------------------------------------------------- respan
    model(
        "respan/span-01", "span-01", "Respan", "respan",
        "proprietary", False, None,
        claims(["classify"], ["text"], "en"),
        {"card": "https://openrouter.ai/respan/span-01", "paper": None},
        [version("span-01@2026-09", api_slug="respan/span-01")],
        [hosted(
            "span-01.openrouter", "span-01@2026-09", "openrouter",
            "respan/span-01", "openrouter.systemone",
            "Behavior scorer addressed through OpenRouter's /api/v1/"
            "systemone: one noul (yes-probability) question per label, "
            "argmax = decision. Rejects an explicit criteria: null (omit the "
            "field) and wants the text as a plain-string state (it scores "
            "conversation spans). Non-ZDR (†) - ZDR-enforcing account "
            "settings exclude it. Measured $9.0e-7/q (payloads are tiny).")]
        ,
        ["Span-01"],
        notes="Native output is one probability per label - claimed track is "
              "classify (argmax), not decision. Size closed, params_b null.",
    ),
    model(
        "respan/span-01-lite", "span-01", "Respan", "respan",
        "proprietary", False, None,
        claims(["classify"], ["text"], "en"),
        {"card": "https://openrouter.ai/respan/span-01-lite", "paper": None},
        [version("span-01-lite@2026-09", api_slug="respan/span-01-lite")],
        [hosted(
            "span-01-lite.openrouter", "span-01-lite@2026-09", "openrouter",
            "respan/span-01-lite", "openrouter.systemone",
            "Span-01's free-tier sibling, same noul-per-label behavior "
            "scoring. Billed $0 on both v0 runs (the plain id is priced "
            "$0.0, same as a :free twin). Non-ZDR (†).")]
        ,
        ["Span-01 Lite"],
        notes="Size closed, params_b null.",
    ),

    # ------------------------------------------------------------- perplexity
    model(
        "perplexity/pplx-decider-v1", "pplx-decider", "Perplexity",
        "perplexity", "proprietary", False, 27,
        claims(["decision", "score"], ["text"], "en"),
        {"card": "https://openrouter.ai/perplexity/pplx-decider-v1-27b",
         "paper": None},
        [version("pplx-decider-v1@2026-10",
                 api_slug="perplexity/pplx-decider-v1-27b")],
        [hosted(
            "pplx-decider-v1.openrouter", "pplx-decider-v1@2026-10",
            "openrouter", "perplexity/pplx-decider-v1-27b",
            "openrouter.systemone",
            "Perplexity's Decisions-API entry: typed probabilistic answers - "
            "choice with per-option probabilities, noul, score rubrics - up "
            "to 128 questions per request. 262K ctx per the v0 bench notes. "
            "Reports retainsPrompts: false (no †). $0.04/M input; measured "
            "$4.4e-6/q.")]
        ,
        ["Decider V1 27B (OpenRouter)"],
        notes="The README row lists an image modality for the Decider family "
              "but records it as not live yet - claims.modalities stays "
              "[text]. claims.languages is the v0-measured baseline (100% "
              "on the 9-language suite).",
    ),
    model(
        "perplexity/pplx-decider-v1.1", "pplx-decider", "Perplexity",
        "perplexity", "proprietary", False, 27,
        claims(["decision", "score"], ["text"], "en"),
        {"card": "https://openrouter.ai/perplexity/pplx-decider-v1.1-27b",
         "paper": None},
        [version("pplx-decider-v1.1@2026-10",
                 api_slug="perplexity/pplx-decider-v1.1-27b")],
        [hosted(
            "pplx-decider-v1.1.openrouter", "pplx-decider-v1.1@2026-10",
            "openrouter", "perplexity/pplx-decider-v1.1-27b",
            "openrouter.systemone",
            "The value pick of the v0 Decisions arrivals: same 5,273 input "
            "tokens as V1 at half the list price ($0.02 vs $0.04/M - "
            "measured $2.2e-6/q, the cheapest paid Decisions arrival). "
            "retainsPrompts: false (no †).")]
        ,
        ["Decider V1.1 27B (OpenRouter)"],
        notes="Same family caveats as pplx-decider-v1 (image listed, not "
              "live; languages = measured baseline).",
    ),
    model(
        "perplexity/pplx-decider-v1.2", "pplx-decider", "Perplexity",
        "perplexity", "proprietary", False, 27,
        claims(["decision", "score"], ["text"], "en"),
        {"card": "https://openrouter.ai/perplexity/pplx-decider-v1.2",
         "paper": None},
        [version("pplx-decider-v1.2@2026-10",
                 api_slug="perplexity/pplx-decider-v1.2")],
        [hosted(
            "pplx-decider-v1.2.openrouter", "pplx-decider-v1.2@2026-10",
            "openrouter", "perplexity/pplx-decider-v1.2",
            "openrouter.systemone",
            "Ties the 93.8% incumbents tier (45/48) at $1.0e-6/q - a fifth "
            "of V1's price, half of Jev's - and holds 54/54 multilingual. "
            "V1.2 lifts context to 262k (per the card). The V1.2/Flash "
            "endpoints expose no per-endpoint retention policy, so the "
            "family's no-† carries.")]
        ,
        ["Decider V1.2 27B (OpenRouter)"],
        notes="Same family caveats (image listed, not live; languages = "
              "measured baseline).",
    ),
    model(
        "perplexity/pplx-decider-flash-v1.2", "pplx-decider", "Perplexity",
        "perplexity", "proprietary", False, 4,
        claims(["decision", "score"], ["text"], "en"),
        {"card": "https://openrouter.ai/perplexity/pplx-decider-flash-v1.2",
         "paper": None},
        [version("pplx-decider-flash-v1.2@2026-10",
                 api_slug="perplexity/pplx-decider-flash-v1.2")],
        [hosted(
            "pplx-decider-flash-v1.2.openrouter",
            "pplx-decider-flash-v1.2@2026-10", "openrouter",
            "perplexity/pplx-decider-flash-v1.2", "openrouter.systemone",
            "The budget pick: 85.4% on the mixed pool at $1.1e-6/q (53/54 "
            "multilingual, one Sinhala miss). No per-endpoint retention "
            "policy exposed - the family's no-† carries.")]
        ,
        ["Decider Flash V1.2 (OpenRouter)"],
        notes="Same family caveats (image listed, not live; languages = "
              "measured baseline).",
    ),

    # -------------------------------------------------------------- microsoft
    model(
        "microsoft/microsoft-decision-1", "microsoft-decision-1",
        "Microsoft", "microsoft", "proprietary", False, 9,
        claims(["decision", "score"], ["text"], "en"),
        {"card": "https://openrouter.ai/microsoft/microsoft-decision-1",
         "paper": None},
        [version("microsoft-decision-1@2026-10",
                 api_slug="microsoft/microsoft-decision-1")],
        [hosted(
            "microsoft-decision-1.openrouter",
            "microsoft-decision-1@2026-10", "openrouter",
            "microsoft/microsoft-decision-1", "openrouter.systemone",
            "Microsoft's Model Foundry entry: Qwen3.5-9B post-trained for "
            "single-pass scoring. Azure is the only endpoint and satisfies "
            "this account's enforced ZDR (no †). 33K ctx per the v0 bench "
            "notes; $0.042/M input; measured $2.1e-6/q. Perfect 54/54 "
            "multilingual run in v0.")]
        ,
        ["Microsoft Decision-1 (OpenRouter)"],
        notes="The card says updated continually while the API shape stays "
              "the same - a standing silent-change risk for the version pin.",
    ),

    # ---------------------------------------------------------------- nace-ai
    model(
        "nace-ai/drex-v1.5", "drex", "Nace.AI", "nace-ai",
        "Nace.AI Open RAIL-M (weights); Apache-2.0 (runtime)", True, 9,
        claims(["decision", "score"], ["text"], "en"),
        {"card": "https://openrouter.ai/nace-ai/drex-v1.5", "paper": None},
        [version("drex-v1.5@2026-10", api_slug="nace-ai/drex-v1.5")],
        [hosted(
            "drex-v1.5.openrouter", "drex-v1.5@2026-10", "openrouter",
            "nace-ai/drex-v1.5", "openrouter.systemone",
            "Nace.AI's MiMo-V2.6-Distill-Qwen-9B on the System One contract "
            "(card claims Decision Index 0.2.1 rank 1). 131K ctx per the v0 "
            "bench notes; the DeepInfra endpoint satisfies the account's "
            "ZDR (no †). $0.04/M input; measured $1.4e-6/q with the leanest "
            "prompts of the hosted field (1.7k tokens for the whole "
            "48-question run). Ties the Decider tier at 93.8%.")]
        ,
        ["Drex v1.5 (OpenRouter)"],
        notes="Open weights, but no local deployment: the fp32 twin needs "
              "~36 GB (the v0 box has 31.4) and GGUF serving waits on "
              "Nace's llama.cpp/Ollaya fork - benched hosted only.",
    ),

    # ----------------------------------------------------------------- liquid
    model(
        "liquid/d1", "d1", "Liquid AI", "liquid", "proprietary", False, None,
        claims(["decision", "score"], ["text"], "en"),
        {"card": "https://openrouter.ai/liquid/d1", "paper": None},
        [version("d1@2026-10", api_slug="liquid/d1")],
        [hosted(
            "d1.openrouter", "d1@2026-10", "openrouter", "liquid/d1",
            "openrouter.systemone",
            "Liquid AI's structured decisions on the System One contract: "
            "choice/score/noul with calibrated probabilities. $0.04/M "
            "input; measured $2.7e-6/q - at 0.031 s/question the fastest "
            "hosted decision of the v0 pool. One of its providers retains "
            "prompts, so the row carries †.")]
        ,
        ["D1 (OpenRouter)"],
        notes="Hosted size undisclosed - params_b null. claims.languages is "
              "the v0-measured baseline (100% on the 9-language suite); the "
              "open-weights sibling has a 17-language card.",
    ),
    model(
        "liquid/d1-ultra", "d1", "Liquid AI", "liquid", "proprietary", False,
        None,
        claims(["decision", "score"], ["text"], "en"),
        {"card": "https://openrouter.ai/liquid/d1-ultra", "paper": None},
        [version("d1-ultra@2026-10", api_slug="liquid/d1-ultra",
                 released="2026-10-15")],
        [hosted(
            "d1-ultra.openrouter", "d1-ultra@2026-10", "openrouter",
            "liquid/d1-ultra", "openrouter.systemone",
            "The larger 2026-10-15 step-up, 32k context. Joined the hosted "
            "pool lead on arrival: 95.8% (46/48) at 0.027 s/question - "
            "fastest at the top of the v0 table. Measured $2.2e-6/q; the + "
            "is inherited from the same Liquid routing.")]
        ,
        ["D1-ultra (OpenRouter)"],
        notes="Hosted size undisclosed - params_b null. claims.languages is "
              "the v0-measured baseline (100% on the 9-language suite).",
    ),
    model(
        "liquid/d1-3b", "d1", "Liquid AI", "liquid",
        "LFM Open License v1.0", True, 3.12,
        claims(["decision", "score"], ["text", "image"], "multilingual"),
        {"card": "https://huggingface.co/LiquidAI/d1-3B", "paper": None},
        [version("d1-3b@2026-10")],
        [local(
            "d1-3b.local-von", "d1-3b@2026-10", "von",
            "LiquidAI/d1-3B snapshot loaded in-process via the card's "
            "trust_remote_code route (D1_HOME overrides the C:/src base; "
            "runs under .venv-von, transformers >=5.14). LFM2.5-VL-3B "
            "backbone: the state is encoded once and every typed question "
            "answered from one batched scoring forward, no generation; "
            "text + image state; 17-language card. ram_gb estimated (3.12B "
            "fp32, no overhead).",
            engine="torch", ram=12.5)],
        ["D1 3B (local)"],
        notes="HF repo LiquidAI/d1-3B (in links.card). ram_gb is estimated "
              "from the 3.12B size.",
    ),
    model(
        "liquid/d1-omni-600m", "d1", "Liquid AI", "liquid",
        "LFM Open License v1.0", True, 0.587,
        claims(["decision", "score"], ["text", "image", "audio"],
               "multilingual"),
        {"card": "https://huggingface.co/LiquidAI/d1-omni-600M",
         "paper": None},
        [version("d1-omni-600m@2026-10")],
        [local(
            "d1-omni-600m.local-von", "d1-omni-600m@2026-10", "von",
            "LiquidAI/d1-omni-600M snapshot via the same trust_remote_code "
            "route (D1_HOME; .venv-von). The omni sibling: same "
            "trunk+branch scoring plus a 16 kHz mono int16 speech-state "
            "tower (the system_one call takes a 4th audio arg; benches "
            "exercise the text path). Ties the 3B on English at ~25x the "
            "speed but collapses off-English (51.8%) - the English-centric "
            "card reads true. ram_gb estimated (587M fp32, no overhead).",
            engine="torch", ram=2.4)],
        ["D1-omni 600M (local)"],
        notes="HF repo LiquidAI/d1-omni-600M (in links.card). ram_gb is "
              "estimated from the 587M size.",
    ),

    # -------------------------------------------------------------- base-labs
    model(
        "base-labs/elephant-1", "elephant", "Base Labs", "base-labs",
        "MIT (weights on HF); free API tier", True, 9,
        claims(["decision", "score"], ["text"], "en"),
        {"card": "https://openrouter.ai/baseten/elephant-1", "paper": None},
        [version("elephant-1@2026-10", api_slug="baseten/elephant-1",
                 released="2026-10-20")],
        [hosted(
            "elephant-1.openrouter-free", "elephant-1@2026-10", "openrouter",
            "baseten/elephant-1:free", "openrouter.systemone",
            "Base Labs' Qwen3.5-9B decision model served by Baseten on the "
            "System One contract (choice/noul/score with calibrated "
            "probabilities). The plain baseten/elephant-1 slug answers no "
            "endpoints - all traffic rides :free (20 req/min paced the "
            "multilingual run to 2.97 s/text). MIT weights on HF "
            "baselabs/elephant-1. The free endpoint documents no retention "
            "policy, so the row carries †. Measured $0.")]
        ,
        ["Elephant-1 9B (OpenRouter)"],
        notes="claims.languages is the v0-measured baseline (perfect 54/54 "
              "multilingual run).",
    ),

    # ----------------------------------------------------------------- openai
    model(
        "openai/gpt-6-luna-decisions", "gpt6-luna-decisions", "OpenAI",
        "openai", "proprietary", False, None,
        claims(["decision", "score"], ["text"], "en"),
        {"card": "https://openrouter.ai/openai/gpt-6-luna-decisions",
         "paper": None},
        [version("gpt-6-luna-decisions@2026-10",
                 api_slug="openai/gpt-6-luna-decisions")],
        [hosted(
            "gpt-6-luna-decisions.openrouter", "gpt-6-luna-decisions@2026-10",
            "openrouter", "openai/gpt-6-luna-decisions",
            "openrouter.systemone",
            "GPT-6 Luna behind OpenAI's Decisions API on the System One "
            "contract - typed choice/noul/score with probabilities, "
            "input-only billing ($0.10/M), 1.05M context. The endpoint "
            "retains prompts (†); benched 2026-10-24 only after the "
            "account's OpenAI-group ZDR toggle was lifted by user "
            "direction. Measured $1.3e-5/q; easy calls return degenerate "
            "probabilities (1.0/0.0, confidence 1.0) - calibration-table "
            "fodder.")]
        ,
        ["GPT-6 Luna Decisions (OpenRouter)"],
        notes="Size undisclosed - params_b null. claims.languages is the "
              "v0-measured baseline (perfect 54/54 multilingual run).",
    ),

    # -------------------------------------------------------- togethercomputer
    model(
        "togethercomputer/tev1-4b", "tev1", "Together", "togethercomputer",
        "proprietary API (open recipe + training code)", False, 4,
        claims(["decision"], ["text"], "en"),
        {"card": "https://openrouter.ai/togethercomputer/tev1-4b-experimental",
         "paper": None},
        [version("tev1-4b@2026-10",
                 api_slug="togethercomputer/tev1-4b-experimental")],
        [hosted(
            "tev1-4b.openrouter", "tev1-4b@2026-10", "openrouter",
            "togethercomputer/tev1-4b-experimental", "openrouter.systemone",
            "Experimental Qwen3.5-4B SFT choosing one of 2-24 labeled "
            "options; keeps Qwen's next-token head, and OpenRouter serves it "
            "on the systemone schema anyway. The data recipe and training "
            "code are published. Reports retainsPrompts: false (no †). "
            "$0.042/M input; measured $4.8e-6/q. Trails the other Decisions "
            "arrivals at 79.2%.")]
        ,
        ["Tev1 4B (OpenRouter)"],
        notes="Choice-only in the recorded v0 contract - claims.tracks is "
              "[decision] without score. claims.languages is the "
              "v0-measured baseline (100% on the 9-language suite).",
    ),

    # -------------------------------------------------------------- inception
    model(
        "inception/mercury-decide", "mercury-decide", "Inception",
        "inception", "proprietary", False, None,
        claims(["decision", "score"], ["text"], "en"),
        {"card": "https://openrouter.ai/inception/mercury-decide:free",
         "paper": None},
        [version("mercury-decide@2026-10",
                 api_slug="inception/mercury-decide:free")],
        [hosted(
            "mercury-decide.openrouter-free", "mercury-decide@2026-10",
            "openrouter", "inception/mercury-decide:free",
            "openrouter.systemone",
            "Inception's diffusion LM making typed decisions with calibrated "
            "probabilities, up to 14 decisions/s; same /v1/systemone schema "
            "as Jev. Free tier (20 req/min); the free-tier provider retains "
            "prompts (†). Measured $0 on both v0 runs.")]
        ,
        ["Mercury Decide (OpenRouter)"],
        notes="Size undisclosed - params_b null. claims.languages is the "
              "v0-measured baseline (100% on the 9-language suite).",
    ),

    # ------------------------------------------------------------------- qwen
    model(
        "qwen/qwen3-reranker-8b", "rerank-hosted", "Qwen", "qwen",
        "proprietary", False, 8,
        claims(["rerank"], ["text"], "en"),
        {"card": "https://openrouter.ai/qwen/qwen3-reranker-8b",
         "paper": None},
        [version("qwen3-reranker-8b@2026-09",
                 api_slug="qwen/qwen3-reranker-8b")],
        [hosted(
            "qwen3-reranker-8b.openrouter-rerank",
            "qwen3-reranker-8b@2026-09", "openrouter",
            "qwen/qwen3-reranker-8b", "openrouter.rerank", used_as="classify",
            notes="Cohere-style OpenRouter /api/v1/rerank contract "
            "({model, query, documents} -> relevance scores), used as a "
            "decision engine: score one (instruction, label) pair per "
            "label, argmax. Non-ZDR (†) - ZDR-enforcing account settings "
            "exclude every rerank endpoint. Measured $6.7e-5/q - the "
            "rerankers bill by pair, so they are the expensive end of the "
            "hosted field.")]
        ,
        ["qwen3-reranker-8b (OpenRouter)"],
        notes="used_as classify on the deployment: the native track is "
              "rerank, borrowed as a classifier via argmax (docs/plan/02). "
              "README records the served endpoint under a proprietary API "
              "with no open-weights link - open_weights false. claims."
              "languages is the v0-measured baseline (96% on the "
              "9-language suite).",
    ),

    # --------------------------------------------------------------- voyageai
    model(
        "voyageai/rerank-2.5", "rerank-hosted", "Voyage AI", "voyageai",
        "proprietary", False, None,
        claims(["rerank"], ["text"], "en"),
        {"card": "https://openrouter.ai/voyageai/rerank-2.5", "paper": None},
        [version("rerank-2.5@2026-09", api_slug="voyageai/rerank-2.5")],
        [hosted(
            "rerank-2.5.openrouter-rerank", "rerank-2.5@2026-09",
            "openrouter", "voyageai/rerank-2.5", "openrouter.rerank",
            used_as="classify",
            notes="Cohere-style /api/v1/rerank contract, used as a decision "
            "engine: (instruction, label) pair scores, argmax. Non-ZDR (†). "
            "Measured $3.5e-6/q.")]
        ,
        ["voyage-rerank-2.5 (OpenRouter)"],
        notes="Size closed - params_b null. claims.languages is the "
              "v0-measured baseline (74% on the 9-language suite).",
    ),
    model(
        "voyageai/rerank-2.5-lite", "rerank-hosted", "Voyage AI", "voyageai",
        "proprietary", False, None,
        claims(["rerank"], ["text"], "en"),
        {"card": None, "paper": None},
        [version("rerank-2.5-lite@2026-09",
                 api_slug="voyageai/rerank-2.5-lite")],
        [hosted(
            "rerank-2.5-lite.openrouter-rerank", "rerank-2.5-lite@2026-09",
            "openrouter", "voyageai/rerank-2.5-lite", "openrouter.rerank",
            used_as="classify",
            notes="The cheaper lite tier of Voyage's rerank-2.5, same "
            "(instruction, label) argmax mapping. Non-ZDR (†). Measured "
            "$1.4e-6/q.")]
        ,
        ["voyage-rerank-2.5-lite (OpenRouter)"],
        notes="The README gives no separate card link for the lite tier "
              "(links.card null). Size closed - params_b null. claims."
              "languages is the v0-measured baseline (65% on the "
              "9-language suite).",
    ),

    # ----------------------------------------------------------------- cohere
    model(
        "cohere/rerank-4-pro", "rerank-hosted", "Cohere", "cohere",
        "proprietary", False, None,
        claims(["rerank"], ["text"], "en"),
        {"card": "https://openrouter.ai/cohere/rerank-4-pro", "paper": None},
        [version("rerank-4-pro@2026-09", api_slug="cohere/rerank-4-pro")],
        [hosted(
            "rerank-4-pro.openrouter-rerank", "rerank-4-pro@2026-09",
            "openrouter", "cohere/rerank-4-pro", "openrouter.rerank",
            used_as="classify",
            notes="Cohere-style /api/v1/rerank contract, used as a decision "
            "engine: (instruction, label) pair scores, argmax. Non-ZDR (†). "
            "Bills ~2.5 search units per rerank request, so a 48-question "
            "run costs $0.12 - measured $2.5e-3/q, the expensive end of the "
            "hosted field.")]
        ,
        ["cohere-rerank-4-pro (OpenRouter)"],
        notes="Size closed - params_b null. claims.languages is the "
              "v0-measured baseline (83% on the 9-language suite).",
    ),
    model(
        "cohere/rerank-4-fast", "rerank-hosted", "Cohere", "cohere",
        "proprietary", False, None,
        claims(["rerank"], ["text"], "en"),
        {"card": None, "paper": None},
        [version("rerank-4-fast@2026-09", api_slug="cohere/rerank-4-fast")],
        [hosted(
            "rerank-4-fast.openrouter-rerank", "rerank-4-fast@2026-09",
            "openrouter", "cohere/rerank-4-fast", "openrouter.rerank",
            used_as="classify",
            notes="The fast tier of Cohere Rerank 4, same argmax mapping. "
            "Non-ZDR (†). Measured $2.0e-3/q (~2.5 search units per "
            "request).")]
        ,
        ["cohere-rerank-4-fast (OpenRouter)"],
        notes="The README gives no separate card link for this tier "
              "(links.card null). Size closed - params_b null.",
    ),
    model(
        "cohere/rerank-v3.5", "rerank-hosted", "Cohere", "cohere",
        "proprietary", False, None,
        claims(["rerank"], ["text"], "en"),
        {"card": None, "paper": None},
        [version("rerank-v3.5@2026-09", api_slug="cohere/rerank-v3.5")],
        [hosted(
            "rerank-v3.5.openrouter-rerank", "rerank-v3.5@2026-09",
            "openrouter", "cohere/rerank-v3.5", "openrouter.rerank",
            used_as="classify",
            notes="The previous-generation Cohere reranker, same argmax "
            "mapping. Non-ZDR (†). Measured $1.0e-3/q (~2.5 search units "
            "per request).")]
        ,
        ["cohere-rerank-v3.5 (OpenRouter)"],
        notes="The README gives no separate card link for this tier "
              "(links.card null). Size closed - params_b null.",
    ),
]

# ---------------------------------------------------------------------------
# Writer + verification
# ---------------------------------------------------------------------------

REQUIRED_TOP = ["id", "family", "org", "license", "open_weights", "params_b",
                "claims", "status", "links", "versions", "deployments"]
HOSTED_REQUIRED = ["id", "kind", "version", "provider", "endpoint", "adapter"]
LOCAL_REQUIRED = ["id", "kind", "version", "runner", "runtime", "requires"]


def check_entry(entry: dict, benched: set) -> list:
    """Schema-ish sanity checks; returns a list of problem strings."""
    problems = []
    mid = entry["id"]

    def fail(msg):
        problems.append(f"{mid}: {msg}")

    # params_b and license are nullable per catalog/schema/model.json; the
    # rest must be present AND truthy.
    for key in REQUIRED_TOP:
        if key not in entry:
            fail(f"missing required field {key!r}")
    for key in ("id", "family", "org", "claims", "status", "links",
                "versions", "deployments"):
        if not entry.get(key):
            fail(f"empty required field {key!r}")
    if entry["status"] not in ("active", "culled", "retired"):
        fail(f"bad status {entry['status']!r}")
    if not entry["claims"]["tracks"]:
        fail("claims.tracks is empty")
    for dep in entry["deployments"]:
        req = HOSTED_REQUIRED if dep["kind"] == "hosted" else LOCAL_REQUIRED
        for key in req:
            if dep.get(key) is None:
                fail(f"deployment {dep['id']!r} missing {key!r}")
        if dep["kind"] == "hosted" and dep["version"] not in \
                {v["id"] for v in entry["versions"]}:
            fail(f"deployment {dep['id']!r} references unknown version")
    if not entry["versions"]:
        fail("no versions")
    benched_names = [n for n in entry["_v0_names"] if n in benched]
    entry["_benched"] = bool(benched_names)
    missing = [n for n in entry["_v0_names"] if n not in benched]
    if missing:
        # A mapped v0 system name that no results file knows is a mapping
        # bug, not a false - stop rather than write benched:false silently.
        fail(f"v0 system name(s) not found in legacy/results: {missing}")
    return problems


def write_all() -> None:
    benched = load_benched_names()
    written = []
    problems = []
    for entry in MODELS:
        problems += check_entry(entry, benched)

        data = {k: v for k, v in entry.items() if not k.startswith("_")}
        data["v0"] = {"benched": entry["_benched"]}
        if data["notes"] is None:
            del data["notes"]

        org_dir = os.path.join(MODELS_DIR, entry["_org_dir"])
        os.makedirs(org_dir, exist_ok=True)
        slug = entry["id"].split("/", 1)[1]
        path = os.path.join(org_dir, f"{slug}.yaml")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True,
                           width=100)
        written.append(path)

    if problems:
        for problem in problems:
            print(f"PROBLEM: {problem}", file=sys.stderr)
        raise SystemExit("catalog extraction failed its own sanity checks")

    # Verify: every written file parses, ids are unique, and no file escaped
    # this script's assigned org dirs.
    ids = []
    allowed_dirs = {m["_org_dir"] for m in MODELS}
    for path in written:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert isinstance(data, dict) and data.get("id"), path
        ids.append(data["id"])
        rel = os.path.relpath(path, MODELS_DIR).replace("\\", "/")
        assert rel.split("/", 1)[0] in allowed_dirs, f"escaped org dir: {rel}"
        assert data.get("v0", {}).get("benched") is True or \
            data.get("v0", {}).get("benched") is False, path
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate ids: {dupes}"

    families = sorted({m["family"] for m in MODELS})
    orgs = sorted({m["_org_dir"] for m in MODELS})
    benched_files = sum(1 for m in MODELS if m["_benched"])
    print(f"wrote {len(written)} model files under catalog/models/")
    print(f"org dirs ({len(orgs)}): {', '.join(orgs)}")
    print(f"families ({len(families)}): {', '.join(families)}")
    print(f"v0.benched=true: {benched_files} / {len(written)}")
    print("verification: all files parse (yaml.safe_load), ids unique, "
          "deployments reference known versions")


if __name__ == "__main__":
    write_all()
