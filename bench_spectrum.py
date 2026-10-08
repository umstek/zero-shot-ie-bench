r"""Spectrum benchmark: one mixed pool of questions per ability.

All classification questions (sentiment + topic) form one pool of 48; NER
questions form one pool of 18. Every system answers every question; a
question's difficulty is computed afterwards as the fraction of systems
that got it wrong (continuous 0.0-1.0).

Per-question predictions are stored (not just aggregates) so the app can
plot accuracy along the difficulty spectrum.

Run from the MAIN venv for most systems, from .venv-von for von,
JevK5-Lite, LFM2.5-RLCD 350M, MoJev 0.85B, the Lumma-Fev family,
Julia 1, the Intern-Decision family, K2-Type 0.9B and the Decision 2.0
family:
    python bench_spectrum.py --system GLiNER2.5-base
    ...
    .venv-von/Scripts/python bench_spectrum.py --system von
    .venv-von/Scripts/python bench_spectrum.py --system JevK5-Lite
    .venv-von/Scripts/python bench_spectrum.py --system "LFM2.5-RLCD 350M"
    .venv-von/Scripts/python bench_spectrum.py --system "MoJev 0.85B"
    .venv-von/Scripts/python bench_spectrum.py --system "Lumma-fev 0.15B"
    .venv-von/Scripts/python bench_spectrum.py --system "Lumma-fev 0.6B"
    .venv-von/Scripts/python bench_spectrum.py --system "Lumma-fev 4B"
    .venv-von/Scripts/python bench_spectrum.py \
        --system "Julia 1 144M"
    .venv-von/Scripts/python bench_spectrum.py --system "Intern-Decision 0.8B"
    .venv-von/Scripts/python bench_spectrum.py --system "Intern-Decision 2B"
    .venv-von/Scripts/python bench_spectrum.py --system "Intern-Decision 4B"
    .venv-von/Scripts/python bench_spectrum.py --system "K2-Type 0.9B (local)"
    .venv-von/Scripts/python bench_spectrum.py --system "Decision 2.0 Kai 0.6B"
    .venv-von/Scripts/python bench_spectrum.py --system "Decision 2.0 Eos 0.8B"
    .venv-von/Scripts/python bench_spectrum.py --system "Decision 2.0 Sol 2B"

GLiNER-X (Knowledgator's mT5-encoder multilingual family) needs its own
venv — the classic gliner package with the stanza extra resolves
transformers 5.x, incompatible with the main venv's pinned 4.57.6
(see README setup for `.venv-glinerx`):
    .venv-glinerx/Scripts/python bench_spectrum.py --system GLiNER-X-small
    .venv-glinerx/Scripts/python bench_spectrum.py --system GLiNER-X-base
    .venv-glinerx/Scripts/python bench_spectrum.py --system GLiNER-X-large

Kev 0.8B needs its local server running first (System One contract):
    cd ../kev && uv run --extra serve python -m kev.serve \
        --run jaredpalmer/kev-0.8b --port 8009
AgentJev 0.6B likewise (own /api/evaluate contract):
    cd ../agent-jev && <python> -m jev_service.server \
        --checkpoint agentjev_v1.pt --model-path <Qwen3-0.6B snapshot> \
        --temperatures temperatures.json --port 8149 --device cpu
decider 0.8B and OpenThai 0.8B also serve the System One contract (both
installed in the shared agent-jev venv):
    DECIDER_MODEL=Mapika/decider-0.8b DECIDER_DEVICE=cpu <py> -m uvicorn \
        decider.serve:app --host 127.0.0.1 --port 8018
    OPENTHAI_SYSTEMONE_MODEL=iapp/OpenThai-SystemOne <py> -m uvicorn \
        openthai_systemone.server:app --host 127.0.0.1 --port 8029
Verdict 151M runs in-process from the Verdict-open-jev checkout
(VERDICT_HOME, default C:\src\verdict) under the agent-jev venv python:
    C:/venvs/agent-jev/Scripts/python bench_spectrum.py \
        --system "Verdict 151M (local)"

Ollaya systems need the local Ollaya daemon (TypeSafe System One contract
on :11435; install from https://ollaya.dev/download, then `ollaya serve`
and once `ollaya pull <tag>` per model — see engines/ollaya_client.py):
    python bench_spectrum.py --system "nli deberta-v3-large (Ollaya)"
    python bench_spectrum.py --system "winnow e4b (Ollaya)"

OpenRouter-hosted systems need OPENROUTER_API_KEY in .env; their providers
are not ZDR - they may retain request data (Solar Decide additionally has
a ZDR Upstage endpoint, but the runs here use default routing):
    python bench_spectrum.py --system "Kev 4B (OpenRouter)"
    python bench_spectrum.py --system "Solar Decide (OpenRouter)"
    python bench_spectrum.py --system "Decider V1.1 27B (OpenRouter)"
    # Decisions family: also "Decider V1 27B (OpenRouter)",
    # "D1 (OpenRouter)", "Tev1 4B (OpenRouter)",
    # "Mercury Decide (OpenRouter)" (free tier, 20 req/min)
    python bench_spectrum.py --system "Span-01 Lite"
    python bench_spectrum.py --system "cohere-rerank-v3.5 (OpenRouter)"

Cloudflare's Clef models are hosted on Workers AI and need a one-time
`cf auth login` (the CLI's session token is used and auto-refreshed;
see engines/clef_client.py) or a Workers AI API token in .env as
CLOUDFLARE_AUTH_TOKEN plus CLOUDFLARE_ACCOUNT_ID:
    python bench_spectrum.py --system "Clef (Workers AI)"
    python bench_spectrum.py --system "Clef-flash (Workers AI)"

Fastino's hosted systems need FASTINO_API_KEY in .env (all ZDR per the
model catalog): GLiDE rides Fastino's /v1/systemone (input $0.15/M
tokens, thinking tokens free) and the hosted GLiNER twins - the same
open-weight checkpoints benched locally above - ride the
chat-completions endpoint (input $0.03/M tokens, output free; the
small checkpoint is not hosted, so no twin for it):
    python bench_spectrum.py --system "GLiDE (Fastino)"
    python bench_spectrum.py --system "GLiNER2.5-base (Fastino)"
    python bench_spectrum.py --system "GLiNER2.5-multi (Fastino)"
    python bench_spectrum.py --system "GLiNER2.5-Decide (Fastino)"

TypeLLM's hosted type-safe generation API (constrained enum picks over
their typellm-latest Qwen3.8-27B deployment) needs TYPELLM_API_KEY in
.env (input $0.05/M, thinking $0.50/M, answers free); both modes run,
the thinking one bills reasoning tokens per question:
    python bench_spectrum.py --system "TypeLLM (hosted)"
    python bench_spectrum.py --system "TypeLLM thinking (hosted)"

Output: results/bench_spectrum_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

from bench import NER_LABELS, SENTIMENT_LABELS, TOPIC_LABELS, spans_of
from bench_graded import NER, SENTIMENT, TOPIC
from engines import (decision2_client, glinerx_client, intern_decision_client,
                     julia_client, k2type_client)
from engines.ollaya_client import MODELS as OLLAYA_MODELS

RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "bench_spectrum_results.json")

# fixed question order: sentiment pool then topic pool then NER pool
CLS_QUESTIONS = (
    [{"task": "sentiment", "text": t, "gold": g}
     for tier in ("easy", "medium", "hard") for t, g in SENTIMENT[tier]]
    + [{"task": "topic", "text": t, "gold": g}
       for tier in ("easy", "medium", "hard") for t, g in TOPIC[tier]]
)
NER_QUESTIONS = [
    {"text": text, "gold": sorted(spans_of(text, truth))}
    for tier in ("easy", "medium", "hard") for text, truth in NER[tier]
]

EXTRACTORS = {
    "GLiNER2.5-small": "fastino/gliner2.5-small-v1",
    "GLiNER2.5-base": "fastino/gliner2.5-base-v1",
    "GLiNER2.5-multi": "fastino/gliner2.5-multi-v1",
    "GLiNER2.5-Decide": "fastino/GLiNER2.5-Decide",
    "GLiFormer-base": "knowledgator/gliformer-base-v1",
    "GLiFormer-large": "knowledgator/gliformer-large-v1",
}
GLICLASS = {
    "gliclass-edge": "knowledgator/gliclass-edge-v3.0",
    "gliclass-modern-base": "knowledgator/gliclass-modern-base-v3.0",
    "gliclass-base": "knowledgator/gliclass-base-v3.0",
    "gliclass-large": "knowledgator/gliclass-large-v3.0",
}
# Knowledgator's GLiNER-X multilingual family (mT5 encoder backbone, the
# first non-DeBERTa GLiNER here; classic `gliner` package + stanza word
# splitter, Apache-2.0, released 2026-04-29). engines/glinerx_client.py
# holds the registry and the request-shape findings (NER is the cards'
# only task; classification rides the vendor multitask prompt as a census
# row). Runs under .venv-glinerx/Scripts/python — see README setup
GLINER_X = glinerx_client.MODELS
RERANKERS = {
    "mxbai-rerank-base-v2": "mixedbread-ai/mxbai-rerank-base-v2",
    "mxbai-rerank-large-v2": "mixedbread-ai/mxbai-rerank-large-v2",
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    "GTE-rerank-ModernBERT-base": "Alibaba-NLP/gte-reranker-modernbert-base",
}
# NVIDIA's llama-nemotron-rerank-1b-v2 (1.24B, openmdw-1.1 + Llama 3.2
# Community License, weights refreshed 2026-08-26): a bidirectional-
# attention Llama-3.2-1B cross-encoder whose head ships as REMOTE CODE
# (LlamaBidirectionalForSequenceClassification) and whose trained pair
# format is the card's "question:{q} \n \n passage:{p}" template, so it
# cannot ride the shared CrossEncoder path (which would concatenate the
# pair without the trained markers). It still loads in the MAIN venv:
# the remote code is version-adaptive (transformers >=4.44 incl. 5.x)
# and runs under the pinned 4.57.6 (one benign unrecognized-rope_theta
# warning). Request shape measured on the 48-question mixed pool
# (2026-10-05): the house (instruction, bare-label) pairs the other
# rerankers score collapse it to 9/24 sentiment, raw text + bare labels
# 26/48, raw text + the house label descriptions as the passage wins at
# 31/48 - the same call the Ollaya NLI branch made (bare-label criteria
# measurably hurt the NLI pair encoder, so criteria carry the
# descriptions there). fp32 on CPU (the card's bf16 example targets GPU)
NEMOTRON_RERANKERS = {
    "nemotron-rerank-1b-v2": "nvidia/llama-nemotron-rerank-1b-v2",
}
# hosted on OpenRouter: System One contract (kev, solar-decide, span) and
# the rerank router. Solar Decide is Upstage's structured-decision model on
# Solar Mini 4 (35B MoE / 3B active, 524K context; answers choice with
# probabilities, score rubrics, noul - but rejects an explicit
# criteria: null on noul, omit the field instead). Solar Decide Flash
# (added 2026-10-08) is its low-latency sibling on the same contract
# ($0.05/M input, output free). typesafe/jev-1.13 also lives there (and
# on the newer POST /api/alpha/decisions router - both answer this
# account's key; the old RBAC gate is gone) but is already benched
# through the TypeSafe API directly; typesafe/jev-router is a chat
# router, not a typed-decision endpoint.
#
# The Decisions family OpenRouter added since (output modality
# "decisions", same /v1/systemone contract; only input bills, decisions
# are free): Perplexity's Decider V1 / V1.1 27B (262K ctx, vision-capable,
# $0.04 / $0.02 per M input), Liquid's d1 ($0.04/M; one of its providers
# retains prompts -> non-ZDR), Together's Tev1 4B experimental (SFT of
# Qwen3.5-4B, $0.042/M) and Inception's Mercury Decide on the :free tier
# (diffusion LM, $0; the free-tier provider retains prompts -> non-ZDR).
# openai/gpt-6-luna-decisions ($0.10/M) is NOT benchmarked: OpenAI's
# endpoint retains prompts and this account enforces ZDR, so OpenRouter
# refuses to route to it.
OPENROUTER_SYSTEMONE = {
    "Kev 4B (OpenRouter)": "jaredpalmer/kev-4b",
    "Solar Decide (OpenRouter)": "upstage/solar-decide",
    "Solar Decide Flash (OpenRouter)": "upstage/solar-decide-flash",
    "Span-01": "respan/span-01",
    "Span-01 Lite": "respan/span-01-lite",
    "Decider V1 27B (OpenRouter)": "perplexity/pplx-decider-v1-27b",
    "Decider V1.1 27B (OpenRouter)": "perplexity/pplx-decider-v1.1-27b",
    "D1 (OpenRouter)": "liquid/d1",
    "Tev1 4B (OpenRouter)": "togethercomputer/tev1-4b-experimental",
    "Mercury Decide (OpenRouter)": "inception/mercury-decide:free",
}
# the choice-question subset: every OpenRouter System One model except
# the two Span behavior scorers, whose noul branch in main() runs first
OR_SYSTEMONE_CHOICE = {name for name in OPENROUTER_SYSTEMONE
                       if not name.startswith("Span-01")}
# Cloudflare's Clef decision models on Workers AI (engines/clef_client.py;
# values are the body "model" selectors). Clef rides a frozen Qwen3.8-27B
# backbone (64k ctx, vision), Clef-flash a Qwen3.5-9B one (~39 ms median,
# self-reported); same System One contract, and unlike the OpenRouter
# systems Cloudflare commits to not reading/storing/training on requests
CLEF_MODELS = {
    "Clef (Workers AI)": "clef",
    "Clef-flash (Workers AI)": "clef-flash",
}
# Fastino's hosted GLiDE decision model (engines/fastino_client.py;
# values are the body "model" selectors). One fast pass plus adaptive
# thinking when uncertain; usage.output_tokens are thinking tokens priced
# at $0, so only input bills ($0.15/M). Rides /v1/systemone like the
# local System One servers, hence the batched classify_batched path
FASTINO_MODELS = {"GLiDE (Fastino)": "glide"}
# TypeLLM's hosted type-safe generation API (engines/typellm_client.py;
# single public model typellm-latest, so the value selects the variant:
# thinking off, or per-question reasoning at $0.50/M on top of input's
# $0.05/M - answers stay free). Same names as bench_multilingual.py so
# results files line up across benchmarks
TYPELLM_MODELS = {"TypeLLM (hosted)": False,
                  "TypeLLM thinking (hosted)": True}
# Fastino's hosted GLiNER twins (same open-weight checkpoints as the
# local EXTRACTORS entries; engines/fastino_client.py mirrors the local
# AutoExtractor surface over the chat-completions endpoint, so one API
# request per question like a local one-call-per-text run). Input
# $0.03/M, output $0. The small checkpoint has no hosted twin (not in
# Fastino's catalog; the id 404s), so no entry for it. Decide is the
# decision-tuned sibling - classification primary, extraction still
# supported - mirroring its local twin's both-sections coverage
FASTINO_EXTRACTORS = {
    "GLiNER2.5-base (Fastino)": "gliner2.5-base",
    "GLiNER2.5-multi (Fastino)": "gliner2.5-multi",
    "GLiNER2.5-Decide (Fastino)": "decide",
}
OPENROUTER_RERANKERS = {
    "qwen3-reranker-8b (OpenRouter)": "qwen/qwen3-reranker-8b",
    "voyage-rerank-2.5-lite (OpenRouter)": "voyageai/rerank-2.5-lite",
    "voyage-rerank-2.5 (OpenRouter)": "voyageai/rerank-2.5",
    "nemotron-rerank-vl-1b (OpenRouter)":
        "nvidia/llama-nemotron-rerank-vl-1b-v2:free",
    "cohere-rerank-4-pro (OpenRouter)": "cohere/rerank-4-pro",
    "cohere-rerank-4-fast (OpenRouter)": "cohere/rerank-4-fast",
    "cohere-rerank-v3.5 (OpenRouter)": "cohere/rerank-v3.5",
}
# served by the local Ollaya daemon over the same System One contract
# (engines/ollaya_client.py): MoritzLaurer NLI classifiers, vLLM Semantic
# Router "decision", the full JevK5 4B GGUF and Winnow E4B GGUF
OLLAYA = OLLAYA_MODELS
# FrontiersMind's Lumma-Fev typed-decision family, in-process via the
# lumma-fev package (.venv-von: transformers >=5.4,<6); the 9b sibling is
# not benchmarked (~16 GB bf16, see README)
LUMMA = {
    "Lumma-fev 0.15B": "FrontiersMind/Lumma-fev-0.1b",
    "Lumma-fev 0.6B": "FrontiersMind/Lumma-fev-0.6b",
    "Lumma-fev 4B": "FrontiersMind/Lumma-fev-4b",
}
# SupersonicLabs' Julia 1 typed-decision model, in-process via the `julia`
# package shipped inside its HF repo (engines/julia_client.py loads the
# local snapshot clone, JULIA_HOME, default ../Julia-1); the ONNX/WebGPU
# twin is not benchmarked (redundant here, see README)
JULIA = {"Julia 1 144M": julia_client.MODEL_ID}
# internlm's Intern-Decision typed-decision family, in-process via the
# runtime shipped inside each HF snapshot (engines/intern_decision_client.py
# loads the local C:\src\Intern-Decision-* snapshots); same names as
# bench_multilingual.py so results files line up across benchmarks
INTERN_DECISION = {"Intern-Decision 0.8B": "0.8B",
                   "Intern-Decision 2B": "2B",
                   "Intern-Decision 4B": "4B"}
# IFM's K2-Type-0.9B typed-decision model, in-process via the runtime
# shipped inside its HF snapshot (engines/k2type_client.py loads the local
# C:\src\K2-Type-0.9B snapshot; the upstream jev.serve server hardcodes a
# CUDA GPU this machine lacks); same names as bench_multilingual.py so
# results files line up across benchmarks
K2TYPE = {"K2-Type 0.9B (local)": k2type_client.MODEL_ID}
# vLLM Semantic Router's Decision 2.0 family (the line behind the
# Ollaya-served `decision` model), in-process via the self-verifying
# runtime shipped inside each HF snapshot (engines/decision2_client.py
# loads the local C:\src\Decision-2.0-* snapshots); same names as
# bench_multilingual.py so results files line up across benchmarks
DECISION2 = {"Decision 2.0 Kai 0.6B": "kai-0.6b",
             "Decision 2.0 Eos 0.8B": "eos-0.8b",
             "Decision 2.0 Sol 2B": "sol-2b"}
ALL_SYSTEMS = (list(EXTRACTORS) + list(FASTINO_EXTRACTORS)
               + list(GLICLASS) + list(GLINER_X) + list(RERANKERS)
               + list(NEMOTRON_RERANKERS)
               + list(OPENROUTER_SYSTEMONE) + list(OPENROUTER_RERANKERS)
               + list(CLEF_MODELS) + list(FASTINO_MODELS)
               + list(TYPELLM_MODELS)
               + ["Certo 421M", "MoJev 0.85B", "nanodiff 350M",
                  "nanodiff 350M v2",
                  "Laya (local)", "Laya typed-decisions", "Jev",
                  "Kev 0.8B (local)", "AgentJev 0.6B (local)",
                  "decider 0.8B (local)", "OpenThai 0.8B (local)",
                  "Verdict 151M (local)", "von", "JevK5-Lite",
                  "LFM2.5-RLCD 350M", "so1 (Qwen2.5-0.5B)"]
               + list(OLLAYA) + list(LUMMA) + list(JULIA)
               + list(INTERN_DECISION) + list(K2TYPE) + list(DECISION2))

# local servers speaking the System One wire format: one JevClient pattern,
# different ports. decider and OpenThai lazy-load their weights on the first
# request, so callers fire one untimed warmup question.
SYSTEMONE_LOCAL_PORTS = {"kev": 8009, "decider": 8018, "openthai": 8029}


def classify_extractor(model_id: str, gliformer: bool):
    if gliformer:
        from gliformer import GLiFormer
        model = GLiFormer.from_pretrained(model_id,
                                          load_tokenizer=True).to("cpu").eval()
    else:
        from gliner2 import AutoExtractor
        model = AutoExtractor.from_pretrained(model_id, map_location="cpu")

    def cls_one(text: str, task: str) -> str | None:
        labels = list(SENTIMENT_LABELS if task == "sentiment"
                      else TOPIC_LABELS)
        if gliformer:
            out = model.classify(text, labels, threshold=0.5)
            return out[0]["class_name"] if out else None
        return model.classify_text(text, {"task": labels})["task"]

    def ner_one(text: str) -> list:
        if gliformer:
            ents = model.predict_entities(text, NER_LABELS, threshold=0.5)
            return sorted({(e["start"], e["end"], e["label"]) for e in ents})
        out = model.extract_entities(text, NER_LABELS, include_spans=True,
                                     include_confidence=False)
        return sorted({(i["start"], i["end"], lab)
                       for lab, items in out.get("entities", {}).items()
                       for i in items})

    return cls_one, ner_one


def classify_batched(client_kind: str, repo: str = "convaiinnovations/laya",
                     tracker=None, model_id: str | None = None):
    """Laya, Jev, AgentJev, or a local System One server (Kev, decider,
    OpenThai): one batched call per task, preds mapped back per question.
    The locals serve the same wire format as Jev on their own ports and get
    string instructions — the shape Laya and Kev both expect; AgentJev has
    its own /api/evaluate contract (port 8149) with label descriptions as
    option semantics. model_id selects the checkpoint for the hosted
    System One engines ("or-systemone": kev-4b, solar-decide; "clef":
    clef, clef-flash; "fastino": glide - single model, selector unused)
    and the thinking flag for "typellm" (single public model).
    (Ollaya systems do NOT batch here: their decision layers build the
    premise from the state, so each text is its own request - see the
    dedicated branch in main().)"""
    INSTR = {
        "sentiment": 'What is the overall sentiment of this text: "{text}"',
        "topic": 'Which topic category does this text belong to: "{text}"',
    }

    def _options(task: str) -> dict:
        # AgentJev needs a description per option; topics already carry one,
        # sentiment gets a fixed per-label phrase (no per-question leakage)
        return ({label: f"The text expresses {label} sentiment"
                 for label in SENTIMENT_LABELS} if task == "sentiment"
                else dict(TOPIC_LABELS))

    if client_kind == "laya":
        import laya

        from engines.jev_client import choice

        agent = laya.load(repo)

        def run_task(task: str, texts: list[str]) -> list:
            labels = SENTIMENT_LABELS if task == "sentiment" else TOPIC_LABELS
            questions = {
                f"t{i}": choice(INSTR[task].format(text=t),
                                {l: None for l in labels})
                for i, t in enumerate(texts)}
            out = agent.predict({"task": task}, questions)
            return [out["answers"][f"t{i}"].get("choice")
                    for i in range(len(texts))]
    elif (client_kind in SYSTEMONE_LOCAL_PORTS
          or client_kind in ("or-systemone", "clef", "fastino")):
        from engines.jev_client import JevClient, choice

        if client_kind == "or-systemone":
            # OpenRouter-hosted System One decision engines (kev-4b,
            # solar-decide): same wire format, Bearer key
            from engines.openrouter_client import systemone

            client = systemone(model_id, tracker)
        elif client_kind == "clef":
            # Cloudflare Workers AI-hosted Clef models: same wire format
            # behind the v4 REST envelope (engines/clef_client.py)
            from engines.clef_client import clef as clef_build

            client = clef_build(model_id,
                                usage_sink=tracker.add if tracker else None)
        elif client_kind == "fastino":
            # Fastino-hosted GLiDE: same wire format, flat payload,
            # Bearer key (engines/fastino_client.py)
            from engines.fastino_client import glide as glide_build

            client = glide_build(usage_sink=tracker.add if tracker else None)
        else:
            client = JevClient(
                base_url=f"http://127.0.0.1:{SYSTEMONE_LOCAL_PORTS[client_kind]}"
                         "/v1/systemone",
                model=f"{client_kind}-latest")
            # pay any lazy model loading before the timed section; OpenThai's
            # cold load runs minutes, past ask()'s 120 s default
            client.ask({"task": "warmup"},
                       {"w": choice('Sentiment of "good"?',
                                    {"positive": None, "negative": None})},
                       timeout=600)

        def run_task(task: str, texts: list[str]) -> list:
            labels = SENTIMENT_LABELS if task == "sentiment" else TOPIC_LABELS
            questions = {
                f"t{i}": choice(INSTR[task].format(text=t),
                                {l: None for l in labels})
                for i, t in enumerate(texts)}
            out = client.ask({"task": task}, questions)
            return [out["answers"][f"t{i}"].get("choice")
                    for i in range(len(texts))]
    elif client_kind == "typellm":
        # TypeLLM's hosted API (engines/typellm_client.py): one
        # /v1/generate per task packs all 24 texts as constrained enum
        # questions - the same batching the System One engines above get.
        # The house shape carries over: context is the System One state
        # JSON-serialized ({"task": ...}), each question's instructions
        # restate the text. model_id is the thinking flag (TYPELLM_MODELS).
        from engines.typellm_client import enum_question, generate

        thinking = bool(model_id)

        def run_task(task: str, texts: list[str]) -> list:
            labels = SENTIMENT_LABELS if task == "sentiment" else TOPIC_LABELS
            questions = {
                f"t{i}": enum_question(INSTR[task].format(text=t),
                                       list(labels), thinking=thinking)
                for i, t in enumerate(texts)}
            out = generate(json.dumps({"task": task}), questions,
                           timeout=90,
                           usage_sink=tracker.add if tracker else None)
            return [out["result"].get(f"t{i}") for i in range(len(texts))]
    elif client_kind == "agentjev":
        from engines.agentjev_client import ask

        def run_task(task: str, texts: list[str]) -> list:
            questions = {
                f"t{i}": {"id": f"t{i}", "type": "choice",
                          "question": INSTR[task].format(text=t),
                          "options": _options(task)}
                for i, t in enumerate(texts)}
            out = ask({"task": task}, questions)
            return [out[f"t{i}"]["value"] for i in range(len(texts))]
    elif client_kind == "jev":
        from engines.jev_client import JevClient

        client = JevClient(usage_sink=tracker.add if tracker else None)

        def run_task(task: str, texts: list[str]) -> list:
            labels = SENTIMENT_LABELS if task == "sentiment" else TOPIC_LABELS
            return client.classify(texts, dict(labels), task=task)
    else:
        raise SystemExit(f"unknown client kind {client_kind!r}")
    return run_task


def classify_fastino_hosted(selector: str, tracker=None):
    """Fastino's hosted GLiNER twins (engines/fastino_client.py): the
    same open-weight checkpoints as the local EXTRACTORS entries, served
    from Fastino's chat-completions endpoint. The client mirrors the
    local AutoExtractor surface, so the per-question decomposition and
    the parsing match classify_extractor exactly; the difference is one
    network round-trip per question instead of a local forward pass.
    Covers both sections like the local checkpoints (Decide's
    classification tuning does not remove its span support)."""
    from engines.fastino_client import gliner

    client = gliner(selector, usage_sink=tracker.add if tracker else None)

    def cls_one(text: str, task: str) -> str | None:
        labels = list(SENTIMENT_LABELS if task == "sentiment"
                      else TOPIC_LABELS)
        return client.classify_text(text, {"task": labels})["task"]

    def ner_one(text: str) -> list:
        out = client.extract_entities(text, NER_LABELS, include_spans=True,
                                      include_confidence=False)
        return sorted({(i["start"], i["end"], lab)
                       for lab, items in out.get("entities", {}).items()
                       for i in items})

    return cls_one, ner_one


def classify_glinerx(model_id: str):
    """Knowledgator's GLiNER-X (classic `gliner` package over an mT5
    encoder, stanza word splitter — runs under .venv-glinerx). NER via
    predict_entities is the card's only supported task; classification
    rides the vendor multitask prompt (engines/glinerx_client.py) and
    lands as a census row — the checkpoints are NER-only and every
    mapping we probed collapses onto one label (see the client
    docstring). Covers both sections like the GLiNER2.5 rows."""
    from engines.glinerx_client import (CLASSIFICATION_THRESHOLD,
                                        NER_THRESHOLD, classification_prompt,
                                        load_glinerx, reduce_classification)

    model = load_glinerx(model_id)

    def cls_one(text: str, task: str) -> str | None:
        labels = list(SENTIMENT_LABELS if task == "sentiment"
                      else TOPIC_LABELS)
        entities = model.predict_entities(
            classification_prompt(labels, text), labels,
            threshold=CLASSIFICATION_THRESHOLD)
        return reduce_classification(entities, labels)

    def ner_one(text: str) -> list:
        entities = model.predict_entities(text, NER_LABELS,
                                          threshold=NER_THRESHOLD)
        return sorted({(e["start"], e["end"], e["label"]) for e in entities})

    return cls_one, ner_one


def classify_gliclass(model_id: str):
    from gliclass import GLiClassModel, ZeroShotClassificationPipeline
    from transformers import AutoTokenizer

    model = GLiClassModel.from_pretrained(model_id)
    pipe = ZeroShotClassificationPipeline(
        model, AutoTokenizer.from_pretrained(model_id),
        classification_type="multi-label", device="cpu")

    def cls_one(text: str, task: str) -> str | None:
        labels = list(SENTIMENT_LABELS if task == "sentiment"
                      else TOPIC_LABELS)
        out = pipe(text, labels, threshold=0.0)[0]
        return max(out, key=lambda x: x["score"])["label"] if out else None
    return cls_one


def classify_reranker(repo: str):
    """Cross-encoder reranker as decision engine (shared by the three
    rerankers): score one (instruction, label) pair per label and pick the
    highest-scoring label. Classification only - rerankers cannot extract
    spans, so no NER answers."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(repo, device="cpu")
    instr = {"sentiment": 'What is the overall sentiment of this text: "{text}"',
             "topic": 'Which topic category does this text belong to: "{text}"'}

    def cls_one(text: str, task: str) -> str | None:
        labels = list(SENTIMENT_LABELS if task == "sentiment"
                      else TOPIC_LABELS)
        pairs = [(instr[task].format(text=text), label) for label in labels]
        scores = model.predict(pairs)
        return labels[max(range(len(scores)), key=lambda i: scores[i])]
    return cls_one


def nemotron_prompt(query: str, passage: str) -> str:
    """The card's trained pair format, verbatim (including the spaces
    around the blank line) - see the NEMOTRON_RERANKERS notes."""
    return f"question:{query} \n \n passage:{passage}"


def classify_nemotron_reranker(repo: str):
    """NVIDIA's llama-nemotron-rerank-1b-v2 as decision engine, loading
    the card verbatim: trust_remote_code (custom bidirectional-attention
    head), left padding, pad token falling back to eos. Each question
    scores one prompt per label - the raw text as the query, the house
    label description as the passage (shape probe in the
    NEMOTRON_RERANKERS notes) - in one batched forward, argmax = decision.
    Classification only - no span extraction, so no NER answers. fp32 on
    CPU (the card's bf16 example targets GPU)."""
    import torch
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)

    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True,
                                              padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        repo, trust_remote_code=True).eval()
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id
    descriptions = {**SENTIMENT_LABELS, **TOPIC_LABELS}

    def cls_one(text: str, task: str) -> str | None:
        labels = list(SENTIMENT_LABELS if task == "sentiment"
                      else TOPIC_LABELS)
        prompts = [nemotron_prompt(text, descriptions[label])
                   for label in labels]
        batch = tokenizer(prompts, padding=True, truncation=True,
                          return_tensors="pt", max_length=512)
        with torch.inference_mode():
            logits = model(**batch).logits
        scores = logits.view(-1).tolist()
        return labels[max(range(len(scores)), key=lambda i: scores[i])]
    return cls_one


def classify_certo():
    """Certo 421M: calibrated non-generative decision model (vendored
    engines/certo_engine/, card documents no PyPI package). One forward pass scores
    each label description against the state; argmax = decision. Classification
    only - fixed option lists in, one label out, no span extraction."""
    from huggingface_hub import snapshot_download

    from engines.certo_engine import DecisionModel

    model = DecisionModel.load(
        snapshot_download("altslate/certo-decision-model"), device="cpu")
    OPTIONS = {
        task: [{"id": label, "description": desc}
               for label, desc in (SENTIMENT_LABELS if task == "sentiment"
                                   else TOPIC_LABELS).items()]
        for task in ("sentiment", "topic")}

    def cls_one(text: str, task: str) -> str | None:
        return model.decide(text, OPTIONS[task])["top"]
    return cls_one


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True, choices=ALL_SYSTEMS)
    args = parser.parse_args()
    name = args.system

    print(f"{name}: 48 classification questions ...")
    t0 = time.perf_counter()
    cls_preds: list[str | None] = []
    cls_lat: list[float] = []
    ner_ok: list[bool] | None = None
    ner_lat: list[float] = []
    tracker = None  # set by the hosted branches; locals stay untracked

    if (name in EXTRACTORS or name in GLICLASS
            or name in FASTINO_EXTRACTORS or name in GLINER_X):
        if name in EXTRACTORS:
            cls_one, ner_one = classify_extractor(
                EXTRACTORS[name], name.startswith("GLiFormer"))
        elif name in GLICLASS:
            cls_one = classify_gliclass(GLICLASS[name])
            ner_one = None
        elif name in GLINER_X:
            # Knowledgator's GLiNER-X: NER-only checkpoints, so the cls
            # answers ride the vendor multitask prompt (census rows)
            cls_one, ner_one = classify_glinerx(GLINER_X[name])
        else:
            # Fastino's hosted GLiNER twins: measured provider accounting
            # like the other hosted branches (input tokens x list price)
            from engines.openrouter_client import UsageTracker

            tracker = UsageTracker()
            cls_one, ner_one = classify_fastino_hosted(
                FASTINO_EXTRACTORS[name], tracker)
        for q in CLS_QUESTIONS:
            t1 = time.perf_counter()
            cls_preds.append(cls_one(q["text"], q["task"]))
            cls_lat.append(time.perf_counter() - t1)
        if ner_one is not None:
            print(f"  + 18 NER questions ...")
            ner_ok = []
            for q in NER_QUESTIONS:
                t1 = time.perf_counter()
                ner_ok.append(ner_one(q["text"]) == q["gold"])
                ner_lat.append(time.perf_counter() - t1)
    elif name in RERANKERS:
        # in-process cross-encoder rerankers as decision engines,
        # classification only - no span extraction, so no NER answers
        cls_one = classify_reranker(RERANKERS[name])
        for q in CLS_QUESTIONS:
            t1 = time.perf_counter()
            cls_preds.append(cls_one(q["text"], q["task"]))
            cls_lat.append(time.perf_counter() - t1)
    elif name in NEMOTRON_RERANKERS:
        # NVIDIA's remote-code bidirectional cross-encoder: card-verbatim
        # load + measured (text, described-label) shape - see the
        # NEMOTRON_RERANKERS notes. Classification only, no NER answers.
        cls_one = classify_nemotron_reranker(NEMOTRON_RERANKERS[name])
        for q in CLS_QUESTIONS:
            t1 = time.perf_counter()
            cls_preds.append(cls_one(q["text"], q["task"]))
            cls_lat.append(time.perf_counter() - t1)
    elif name in OPENROUTER_RERANKERS:
        # OpenRouter-hosted rerankers: same decision-engine mapping as the
        # local cross-encoders (score one (instruction, label) pair per
        # label, argmax), one API request per question. Non-ZDR endpoints.
        from engines.openrouter_client import UsageTracker, rerank_classify

        model_id = OPENROUTER_RERANKERS[name]
        tracker = UsageTracker()
        instr = {"sentiment": 'What is the overall sentiment of this text: "{text}"',
                 "topic": 'Which topic category does this text belong to: "{text}"'}
        for q in CLS_QUESTIONS:
            labels = list(SENTIMENT_LABELS if q["task"] == "sentiment"
                          else TOPIC_LABELS)
            t1 = time.perf_counter()
            cls_preds.append(rerank_classify(
                model_id, instr[q["task"]].format(text=q["text"]), labels,
                tracker))
            cls_lat.append(time.perf_counter() - t1)
    elif name in ("Span-01", "Span-01 Lite"):
        # OpenRouter-hosted behavior scorer: one noul (yes-probability)
        # question per label in one request, argmax = classification.
        # Non-ZDR endpoint. Classification only - no span extraction.
        from engines.openrouter_client import UsageTracker, noul_classify

        model_id = OPENROUTER_SYSTEMONE[name]
        tracker = UsageTracker()
        QUESTION = {"sentiment": "Does this text express {label} sentiment?",
                    "topic": "Is this text about the {label} topic?"}
        for q in CLS_QUESTIONS:
            labels = list(SENTIMENT_LABELS if q["task"] == "sentiment"
                          else TOPIC_LABELS)
            t1 = time.perf_counter()
            cls_preds.append(noul_classify(model_id, q["text"],
                                           QUESTION[q["task"]], labels,
                                           tracker))
            cls_lat.append(time.perf_counter() - t1)
    elif name == "Certo 421M":
        # in-process decision head over a ModernBERT-large backbone
        # (vendored engines/certo_engine/); classification only - no span
        # extraction, so no NER answers
        cls_one = classify_certo()
        for q in CLS_QUESTIONS:
            t1 = time.perf_counter()
            cls_preds.append(cls_one(q["text"], q["task"]))
            cls_lat.append(time.perf_counter() - t1)
    elif (name in ("Laya (local)", "Laya typed-decisions", "Jev",
                   "Kev 0.8B (local)", "AgentJev 0.6B (local)",
                   "decider 0.8B (local)", "OpenThai 0.8B (local)")
          or name in OR_SYSTEMONE_CHOICE
          or name in CLEF_MODELS or name in FASTINO_MODELS
          or name in TYPELLM_MODELS):
        from engines.openrouter_client import UsageTracker

        repo = ("convaiinnovations/laya-typed-decisions"
                if name == "Laya typed-decisions"
                else "convaiinnovations/laya")
        kind = ("laya" if name.startswith("Laya")
                else "kev" if name.startswith("Kev 0.8B")
                else "agentjev" if name.startswith("AgentJev")
                else "decider" if name.startswith("decider")
                else "openthai" if name.startswith("OpenThai")
                else "or-systemone" if name in OR_SYSTEMONE_CHOICE
                else "clef" if name in CLEF_MODELS
                else "fastino" if name in FASTINO_MODELS
                else "typellm" if name in TYPELLM_MODELS
                else "jev" if name == "Jev" else None)
        if kind is None:   # unmapped names must never reach a cloud API
            raise SystemExit(f"unwired system {name!r} — add a kind mapping")
        tracker = (UsageTracker() if name == "Jev"
                   or name in OR_SYSTEMONE_CHOICE
                   or name in CLEF_MODELS or name in FASTINO_MODELS
                   or name in TYPELLM_MODELS
                   else None)
        run_task = classify_batched(kind, repo=repo, tracker=tracker,
                                    model_id=OPENROUTER_SYSTEMONE.get(name)
                                    or CLEF_MODELS.get(name)
                                    or FASTINO_MODELS.get(name)
                                    or TYPELLM_MODELS.get(name))
        for task in ("sentiment", "topic"):
            texts = [q["text"] for q in CLS_QUESTIONS if q["task"] == task]
            t1 = time.perf_counter()
            preds = run_task(task, texts)
            dt = time.perf_counter() - t1
            cls_preds.extend(preds)
            cls_lat.extend([dt / len(texts)] * len(texts))
    elif name == "Verdict 151M (local)":
        # in-process rlcd engine; run this entry under the agent-jev venv
        # python (torch + transformers 5), which the other entries don't need
        home = os.environ.get("VERDICT_HOME", r"C:\src\verdict")
        sys.path.insert(0, home)
        from rlcd import Choice, DecisionEngine, Option

        engine = DecisionEngine(model_name_or_path=os.path.join(
            home, "artifacts", "v2"), device="cpu")
        QUESTION = {"sentiment": "What is the overall sentiment of this text?",
                    "topic": "Which topic category does this text belong to?"}
        for q in CLS_QUESTIONS:
            desc = dict(SENTIMENT_LABELS if q["task"] == "sentiment"
                        else TOPIC_LABELS)
            query = Choice(id="q", question=QUESTION[q["task"]],
                           options=[Option(id=l, description=d)
                                    for l, d in desc.items()])
            t1 = time.perf_counter()
            res = engine.evaluate(context=q["text"],
                                  queries=[query]).results[0]
            cls_lat.append(time.perf_counter() - t1)
            # abstaining answers nothing: wrong against any gold label
            cls_preds.append(None if res.is_abstention else res.selected_id)
    elif name in OLLAYA:
        # Ollaya-served decision models (local daemon on :11435, System One
        # contract): one request per question, and the state IS the text -
        # these models' decision layers build their premise from the state,
        # so embedding the text in the instructions (the shape the kev
        # servers take) silently drops it for some of them. Criteria carry
        # the label descriptions these models score (bare-label criteria
        # measurably hurt the NLI pair encoder). Classification only - no
        # span extraction, so no NER answers.
        from engines.jev_client import choice
        from engines.ollaya_client import systemone

        client = systemone(OLLAYA[name])
        # pay the first-request model load (winnow:e4b pulls 8 GB into
        # memory) before the timed section, like the other local servers
        client.ask("good", {"w": choice(
            "What is the sentiment of this text?",
            dict(SENTIMENT_LABELS))}, timeout=600)
        QUESTION = {"sentiment": "What is the overall sentiment of this text?",
                    "topic": "Which topic category does this text belong to?"}
        for q in CLS_QUESTIONS:
            labels = dict(SENTIMENT_LABELS if q["task"] == "sentiment"
                          else TOPIC_LABELS)
            t1 = time.perf_counter()
            out = client.ask(q["text"], {"q": choice(QUESTION[q["task"]],
                                                     labels)}, timeout=600)
            cls_lat.append(time.perf_counter() - t1)
            cls_preds.append(out["answers"]["q"].get("choice"))
    elif name == "von":
        from engines.von_client import load_von_decider

        decide = load_von_decider()

        for q in CLS_QUESTIONS:
            labels = (SENTIMENT_LABELS if q["task"] == "sentiment"
                      else TOPIC_LABELS)
            t1 = time.perf_counter()
            res = decide(state=q["text"], choices=dict(labels),
                         instructions=f"What is the overall {q['task']} "
                                      "of this text?")
            cls_lat.append(time.perf_counter() - t1)
            cls_preds.append(res.choice)
    elif name == "JevK5-Lite":
        # in-process label-head classifier like von, run under the
        # .venv-von python (transformers 5.17 + jevk5); classification
        # only - no span extraction, so no NER answers
        from jevk5 import JevK5Lite

        lite = JevK5Lite.from_pretrained("alibiserikbay/JevK5-Lite",
                                         threads=16)
        for q in CLS_QUESTIONS:
            labels = list(SENTIMENT_LABELS if q["task"] == "sentiment"
                          else TOPIC_LABELS)
            t1 = time.perf_counter()
            out = lite.classify(q["text"], {"task": labels})
            cls_lat.append(time.perf_counter() - t1)
            cls_preds.append(out["task"]["labels"][0]
                             if out["task"]["labels"] else None)
    elif name == "LFM2.5-RLCD 350M":
        # in-process constrained-decision engine (vendored engines/rlcd_engine/),
        # run under the .venv-von python (transformers 5.17 + jsonschema);
        # classification only - the supported schema subset (flat
        # boolean/string-enum fields) cannot express span extraction, so
        # no NER answers
        from engines.rlcd_engine.engine import Engine

        engine = Engine(device="cpu", dtype="float32")
        FIELD_DESC = {"sentiment": "The overall sentiment of the text",
                      "topic": "The topic category of the text"}

        def schema_for(task: str) -> dict:
            return {"type": "object",
                    "properties": {task: {"type": "string",
                                          "description": FIELD_DESC[task],
                                          "enum": list(SENTIMENT_LABELS
                                                       if task == "sentiment"
                                                       else TOPIC_LABELS)}},
                    "required": [task],
                    "additionalProperties": False}

        for q in CLS_QUESTIONS:
            t1 = time.perf_counter()
            res = engine.constrained(q["text"], schema_for(q["task"]))
            cls_lat.append(time.perf_counter() - t1)
            try:
                cls_preds.append(json.loads(res["text"])[q["task"]])
            except (KeyError, ValueError):
                cls_preds.append(None)
    elif name == "MoJev 0.85B":
        # in-process packed one-pass decision scorer (vendored
        # engines/mojev_engine/), run under the .venv-von python (transformers 5.17
        # for the Qwen3.5 encoder); classification only - fixed candidate
        # menus in, one label out, no span extraction, so no NER answers
        from engines.mojev_engine import load_engine

        score, _ = load_engine("cpu")
        QUESTION = {"sentiment": "What is the overall sentiment of this "
                                 "text?",
                    "topic": "Which topic category does this text belong "
                             "to?"}
        for q in CLS_QUESTIONS:
            t1 = time.perf_counter()
            pred, _ = score(q["text"], q["task"], QUESTION[q["task"]],
                            list(SENTIMENT_LABELS if q["task"] == "sentiment"
                                 else TOPIC_LABELS))
            cls_lat.append(time.perf_counter() - t1)
            cls_preds.append(pred)
    elif name in LUMMA:
        # in-process Lumma-Fev typed-decision model via the lumma-fev
        # package (.venv-von: transformers >=5.4,<6); classification only -
        # no span extraction, so no NER answers. Request shape measured on
        # the eight easy-tier sentiment texts with 0.1b, deterministic
        # across repeats: the text goes in the state AND is restated in
        # the instructions (5/8) - bare instructions over the state (3/8)
        # and the kev shape (empty state, text only in the instructions,
        # 3/8) both trail, and the kev shape's argmax collapses onto one
        # label, i.e. the text is effectively dropped (same symptom the
        # Ollaya modernbert shows). Criteria carry the label descriptions,
        # the options these models score. One decide() call per question -
        # a decide() answers all its questions in one forward pass, but
        # each pool question is one text with one choice here.
        import lumma_fev

        model = lumma_fev.load(LUMMA[name])
        # one untimed decide() pays torch's first-pass init (kernel
        # dispatch, allocator warm-up) before the timed section
        model.decide("good", {"w": {"type": "choice",
                                    "instructions": "What is the sentiment "
                                                    "of this text?",
                                    "criteria": dict(SENTIMENT_LABELS)}})
        INSTR = {"sentiment": 'What is the overall sentiment of this text: "{text}"',
                 "topic": 'Which topic category does this text belong to: "{text}"'}
        for q in CLS_QUESTIONS:
            labels = dict(SENTIMENT_LABELS if q["task"] == "sentiment"
                          else TOPIC_LABELS)
            t1 = time.perf_counter()
            out = model.decide(q["text"], {"q": {
                "type": "choice",
                "instructions": INSTR[q["task"]].format(text=q["text"]),
                "criteria": labels}})
            cls_lat.append(time.perf_counter() - t1)
            cls_preds.append(out["q"].get("choice"))
    elif name in JULIA:
        # in-process Julia 1 via the `julia` package shipped inside its HF
        # repo (engines/julia_client.py loads the ../Julia-1 snapshot under
        # .venv-von; the transformers 5.17 note lives there); classification
        # only - no span extraction, so no NER answers. Request shape
        # measured on the eight easy-tier sentiment texts, deterministic
        # across repeats: bare instructions over the state and the text
        # restated in the instructions BOTH score 3/8 with identical picks
        # - the shape does not move this model (its noul and domain-matched
        # choice questions read the state fine; informal review sentiment
        # is simply far from its typed-decisions training domain). The
        # house shape (text restated) stays for comparability. Criteria
        # carry the label descriptions, the options the model scores
        # (Julia requires nonempty descriptions). One predict() per
        # question - one predict() scores all its questions in one batch,
        # but each pool question is one text with one choice here.
        engine = julia_client.load_engine()
        # one untimed predict() pays torch's first-pass init (kernel
        # dispatch, allocator warm-up) before the timed section
        engine.predict("good", {"w": {"type": "choice",
                                      "instructions": "What is the sentiment "
                                                      "of this text?",
                                      "criteria": dict(SENTIMENT_LABELS)}})
        INSTR = {"sentiment": 'What is the overall sentiment of this text: "{text}"',
                 "topic": 'Which topic category does this text belong to: "{text}"'}
        for q in CLS_QUESTIONS:
            labels = dict(SENTIMENT_LABELS if q["task"] == "sentiment"
                          else TOPIC_LABELS)
            t1 = time.perf_counter()
            out = engine.predict(q["text"], {"q": {
                "type": "choice",
                "instructions": INSTR[q["task"]].format(text=q["text"]),
                "criteria": labels}})
            cls_lat.append(time.perf_counter() - t1)
            cls_preds.append(out["answers"]["q"].get("choice"))
    elif name in INTERN_DECISION:
        # in-process internlm Intern-Decision (Qwen3.5 fine-tune) via the
        # runtime shipped inside each HF snapshot
        # (engines/intern_decision_client.py loads the local snapshot with
        # its own per-checkpoint calibration temperature); classification
        # only - no span extraction, so no NER answers. Request shape
        # measured on the eight easy-tier sentiment texts with the 0.8B,
        # deterministic across repeats: bare instructions over the state
        # and the text restated in the instructions BOTH score 8/8 with
        # identical picks - the shape does not move this model (its choice
        # and score schemas read the state fine). The house shape (text
        # restated) stays for comparability. Criteria carry the label
        # descriptions, the options the model scores (choice criteria must
        # be an object). One predict() per question - one predict() scores
        # ALL its questions (up to 16) in one causal forward pass at their
        # masked <decision> slots, but each pool question is one text with
        # one choice here. fp32 on CPU: bf16 runs ~7x slower with
        # identical predictions (see the client docstring).
        engine = intern_decision_client.load_engine(INTERN_DECISION[name])
        # one untimed predict() pays torch's first-pass init (kernel
        # dispatch, allocator warm-up) before the timed section
        engine.predict({"state": "good", "questions": {"w": {
            "type": "choice",
            "instructions": "What is the sentiment of this text?",
            "criteria": dict(SENTIMENT_LABELS)}}})
        INSTR = {"sentiment": 'What is the overall sentiment of this text: "{text}"',
                 "topic": 'Which topic category does this text belong to: "{text}"'}
        for q in CLS_QUESTIONS:
            labels = dict(SENTIMENT_LABELS if q["task"] == "sentiment"
                          else TOPIC_LABELS)
            t1 = time.perf_counter()
            out = engine.predict({"state": q["text"], "questions": {"q": {
                "type": "choice",
                "instructions": INSTR[q["task"]].format(text=q["text"]),
                "criteria": labels}}})
            cls_lat.append(time.perf_counter() - t1)
            cls_preds.append(out["answers"]["q"].get("choice"))
    elif name in K2TYPE:
        # IFM's K2-Type-0.9B typed-decision model, in-process via the
        # runtime shipped inside its HF snapshot (engines/k2type_client.py
        # loads C:\src\K2-Type-0.9B on the CPU; the upstream jev.serve
        # server hardcodes a CUDA GPU). One predict() per task packs all
        # 24 texts as one-request questions - the model's native
        # many-questions shape (state encoded once, questions isolated by
        # the block-causal attention mask), the same batching the other
        # System One engines get; latency = dt / n. Classification only -
        # fixed option menus in, one label out, no span extraction, so no
        # NER answers. Request shape measured on the eight easy-tier
        # sentiment texts, deterministic across repeats: the house shape
        # (state {"task": ...}, text restated in the instructions), the
        # kev shape (state = text, bare instructions) and bare-label
        # criteria ALL score 8/8 with identical picks - the shape does not
        # move this model (its choice questions read the state fine). The
        # house shape (text restated) with described criteria - the option
        # texts the pointer head scores, matching the model card's own
        # examples - stays for comparability.
        from engines.jev_client import choice

        engine = k2type_client.load_engine()
        # one untimed predict pays torch's first-pass init (kernel
        # dispatch, allocator warm-up) before the timed section
        engine.predict({"task": "warmup"},
                       {"w": choice('Sentiment of "good"?',
                                    {"positive": None, "negative": None})})
        INSTR = {"sentiment": 'What is the overall sentiment of this text: "{text}"',
                 "topic": 'Which topic category does this text belong to: "{text}"'}
        for task in ("sentiment", "topic"):
            texts = [q["text"] for q in CLS_QUESTIONS if q["task"] == task]
            labels = dict(SENTIMENT_LABELS if task == "sentiment"
                          else TOPIC_LABELS)
            questions = {
                f"t{i}": choice(INSTR[task].format(text=t), labels)
                for i, t in enumerate(texts)}
            t1 = time.perf_counter()
            out = engine.predict({"task": task}, questions)
            dt = time.perf_counter() - t1
            cls_preds.extend(out["answers"][f"t{i}"].get("choice")
                             for i in range(len(texts)))
            cls_lat.extend([dt / len(texts)] * len(texts))
    elif name in DECISION2:
        # vLLM Semantic Router's Decision 2.0 family (the line behind the
        # Ollaya-served `decision` model), in-process via the
        # self-verifying runtime shipped inside each HF snapshot
        # (engines/decision2_client.py loads the local C:\src\Decision-2.0-*
        # snapshots, fp32 on the CPU; the loader also re-binds the vendored
        # checkpoint fingerprint with POSIX path separators - on Windows
        # the backslash keying breaks the manifest identity check). One
        # predict() per task packs all 24 texts as one-request questions -
        # the family's native many-questions shape (one padded batch over
        # the shared state; share_context stays off, the exact scored
        # path), the same batching the other System One engines get;
        # latency = dt / n. Classification only - fixed option menus in,
        # one label out, no span extraction, so no NER answers. Request
        # shape measured on the eight easy-tier sentiment texts with Kai,
        # deterministic across repeats: the house shape (state
        # {"task": ...}, text restated in the instructions), bare-label
        # criteria and the kev shape (state = text, bare instructions) ALL
        # score 8/8 with identical picks - the shape does not move this
        # model. The house shape (text restated) with described criteria -
        # the option descriptions the candidate head scores, matching the
        # model card's own examples - stays for comparability.
        from engines.jev_client import choice

        engine = decision2_client.load_engine(DECISION2[name])
        # one untimed predict pays torch's first-pass init (kernel
        # dispatch, allocator warm-up) before the timed section
        engine.predict({"task": "warmup"},
                       {"w": choice('Sentiment of "good"?',
                                    {"positive": None, "negative": None})})
        INSTR = {"sentiment": 'What is the overall sentiment of this text: "{text}"',
                 "topic": 'Which topic category does this text belong to: "{text}"'}
        for task in ("sentiment", "topic"):
            texts = [q["text"] for q in CLS_QUESTIONS if q["task"] == task]
            labels = dict(SENTIMENT_LABELS if task == "sentiment"
                          else TOPIC_LABELS)
            questions = {
                f"t{i}": choice(INSTR[task].format(text=t), labels)
                for i, t in enumerate(texts)}
            t1 = time.perf_counter()
            out = engine.predict({"task": task}, questions)
            dt = time.perf_counter() - t1
            cls_preds.extend(out["answers"][f"t{i}"].get("choice")
                             for i in range(len(texts)))
            cls_lat.extend([dt / len(texts)] * len(texts))
    elif name == "nanodiff 350M":
        # diffusion-LM decision model: engines/nanodiff_engine vendors the NanoDiff
        # class (BY571/nanoDiff) and the pngwn typed-decision format; one
        # bidirectional forward, softmax restricted to option-letter token
        # ids. Classification only - a single-letter choice interface, no
        # span extraction, so no NER answers. Runs in the MAIN venv
        # (tiktoken); slow (~10 s/question).
        from engines.nanodiff_engine.runner import QUESTION, load_model, predict

        model, _ = load_model("cpu")
        for q in CLS_QUESTIONS:
            t1 = time.perf_counter()
            pred, _ = predict(model, q["text"], QUESTION[q["task"]],
                              list(SENTIMENT_LABELS if q["task"] == "sentiment"
                                   else TOPIC_LABELS), "cpu")
            cls_lat.append(time.perf_counter() - t1)
            cls_preds.append(pred)
    elif name == "nanodiff 350M v2":
        # the v2 retrain: same typed-decision format and architecture, new
        # weights (pngwn/nanodiff-350m-typed-decisions-v2 -- see
        # runner.CKPT_V2); classification only, like v1
        from engines.nanodiff_engine.runner import (CKPT_V2, QUESTION,
                                                    load_model, predict)

        model, _ = load_model("cpu", CKPT_V2)
        for q in CLS_QUESTIONS:
            t1 = time.perf_counter()
            pred, _ = predict(model, q["text"], QUESTION[q["task"]],
                              list(SENTIMENT_LABELS if q["task"] == "sentiment"
                                   else TOPIC_LABELS), "cpu")
            cls_lat.append(time.perf_counter() - t1)
            cls_preds.append(pred)
    elif name == "so1 (Qwen2.5-0.5B)":
        from so1 import Choice, Decider

        decider = Decider.from_pretrained("Qwen/Qwen2.5-0.5B", backend="hf")
        for q in CLS_QUESTIONS:
            labels = list(SENTIMENT_LABELS if q["task"] == "sentiment"
                          else TOPIC_LABELS)
            t1 = time.perf_counter()
            out = decider.decide(state=q["text"],
                                 questions=[Choice(q["task"], labels)],
                                 mode="separate")
            cls_lat.append(time.perf_counter() - t1)
            cls_preds.append(out[0].choice)
    else:
        raise SystemExit(f"unwired system {name!r} — add a dispatch "
                         "branch in main()")

    correct = [p == q["gold"] for p, q in zip(cls_preds, CLS_QUESTIONS)]
    # only the local System One servers, Ollaya's first-request model load,
    # and the first-pass init of Lumma, Julia, Intern-Decision, K2-Type and
    # the Decision 2.0 family get the untimed warm-up; the hosted endpoints
    # are stateless, so every request is timed
    warmed = ((name.startswith(("Kev", "decider", "OpenThai", "K2-Type",
                                "Decision 2.0"))
               or name in OLLAYA or name in LUMMA or name in JULIA
               or name in INTERN_DECISION)
              and name != "Kev 4B (OpenRouter)")
    entry = {
        "recorded": time.strftime("%Y-%m-%d %H:%M"),
        "cls_preds": cls_preds,
        "cls_correct": correct,
        "cls_accuracy": round(sum(correct) / len(correct), 4),
        "cls_mean_latency_s": round(statistics.mean(cls_lat), 3),
        "timing": ("Model download and loading excluded; one untimed "
                   "warm-up question pays the server's lazy weight load "
                   "first - timed latencies are warmed." if warmed else
                   "Model download and loading excluded; first forward "
                   "pass included."),
    }
    if tracker and tracker.requests:
        # measured provider accounting for the hosted run (48 questions)
        entry["usage"] = tracker.as_dict()
    if ner_ok is not None:
        entry["ner_exact"] = ner_ok
        entry["ner_exact_rate"] = round(sum(ner_ok) / len(ner_ok), 4)
        entry["ner_mean_latency_s"] = round(statistics.mean(ner_lat), 3)
    if name == "von":
        from engines.von_client import provenance

        entry["provenance"] = provenance()

    try:
        with open(RESULTS_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError
    except (OSError, ValueError):
        data = {}
    data.setdefault("meta", {
        "pools": {"classification": 48, "ner": 18},
        "difficulty": "per question: fraction of answering systems that "
                      "answered it wrong (continuous 0-1, computed by the "
                      "app from stored per-question results)",
        "notes": ["One mixed pool per ability (48 classification, 18 NER).",
                  "von, JevK5-Lite and LFM2.5-RLCD 350M run in .venv-von; "
                  "so1 uses Qwen2.5-0.5B."]})
    data["systems"] = {k: v for k, v in data.get("systems", {}).items()
                       if k != name}
    data["systems"][name] = entry
    tmp = RESULTS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, RESULTS_FILE)
    ner_note = (f", NER exact {entry['ner_exact_rate'] * 100:.0f}%"
                if ner_ok is not None else "")
    print(f"  classification {entry['cls_accuracy'] * 100:.1f}%{ner_note} "
          f"in {time.perf_counter() - t0:.0f}s -> {RESULTS_FILE}")


if __name__ == "__main__":
    main()
