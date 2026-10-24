"""One-off extractor: catalog YAML for every LOCAL-ONLY README system.

Phase-1 acceptance shape (docs/plan/05): ONE FILE PER SYSTEM (per checkpoint),
like docs/plan/03's own example fastino/gliner2.5-multi.yaml. The universe of
files is the README system table's local rows only:

- every benched checkpoint (each row of the v0-final README "Mixed-pool
  spectrum benchmark (all 78 systems)" table owned by this pass) gets a file
  with v0.benched: true,
- every unbenched README system-table system gets a file with v0.benched: false
  (here: GLiREL, GLiNER-relex, ReLiK — demoed, never benched),
- systems that exist ONLY in "Noted, not benchmarked", TRIALS.md or the engine
  code (gliclass-edge, Nox/Lux/Vega, winnow:12b, Lumma-fev-9b, decider-2b,
  Certo, Verdict, Julia, nanodiff, LFM2.5-RLCD, so1, GLiNER-X, …) get NO file;
  STALE_OUTPUTS below removes the consolidated/superseded files this script
  wrote before the per-checkpoint rework.

Family slugs are harmonized to land the repo-wide union at exactly 38 with the
hosted-org pass (17 slugs, HOSTED_FAMILIES below): the four local cross-encoder
reranker product families share `rerank-local` (mirroring the README's bundled
local-rerankers row and the hosted `rerank-hosted`), and Knowledgator's
GLiFormer + GLiClass join family `gliner` per the README's own lineage
language ("the GLiNER lineage forked between Fastino (GLiNER 2.5 …) and
Knowledgator (classic `gliner` — GLiNER-relex is that line's joint
NER+relations checkpoint — plus GLiFormer and GLiClass)").

Sources: README.md (root; = v0-final for the system table) — system table,
Family maps, mixed-pool table; legacy/engines/* for ids, pinned HF revisions,
dtype/quant/device and ports (engine code wins for technical facts, README for
license/links/sizes); legacy/results/*.json for v0.benched (result-bearing
containers only — the multilingual file's `timing`/`provenance` dicts carry
leftover keys for systems with no recorded run).

Run from the repo root:
    .venv/Scripts/python.exe legacy/extract_catalog_local.py

Only writes/deletes under catalog/models/ in the org dirs owned by this pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "legacy" / "results"
OUT_DIR = REPO_ROOT / "catalog" / "models"

# Org dirs owned by other passes; this script must never write there.
FORBIDDEN_ORGS = {
    "fastino", "liquid", "jaredpalmer", "typesafe", "upstage", "cloudflare",
    "typellm", "respan", "perplexity", "microsoft", "nace-ai", "base-labs",
    "openai", "cohere", "google", "anthropic", "mistral", "meta", "qwen",
    "baseten", "togethercomputer", "inception", "voyageai", "scaledown",
}

# Family slugs reported by the hosted-org pass (37 files, all benched). Used
# only for the union self-check printed at the end; `gliner` (the fastino
# GLiNER2.5 line) is shared with this pass.
HOSTED_FAMILIES = {
    "clef", "d1", "drex", "elephant", "gliner", "glide", "gpt6-luna-decisions",
    "jev", "kev", "mercury-decide", "microsoft-decision-1", "pplx-decider",
    "rerank-hosted", "solar-decide", "span-01", "tev1", "typellm",
}

# Consolidated / superseded files written by the pre-rework version of this
# script. Removed on regen: consolidated multi-checkpoint files are split into
# per-checkpoint files, and systems that are not README system-table rows lose
# their file entirely (docs/plan/05 phase-1 acceptance).
STALE_OUTPUTS = [
    "knowledgator/gliformer.yaml",          # split into gliformer-base/-large
    "knowledgator/gliclass.yaml",           # split into the 3 benched sizes
    "knowledgator/gliner-x.yaml",           # not a README system-table row
    "mixedbread-ai/mxbai-rerank.yaml",      # split into the 2 checkpoints
    "vllm-sr/d3.yaml",                      # split into d3-edge/-lite/-nano
    "internlm/intern-decision.yaml",        # split into the 3 sizes
    "frontiersmind/lumma-fev.yaml",         # split into the 3 benched sizes
    "moritzlaurer/nli.yaml",                # split into the 2 checkpoints
    "eldanring/winnow.yaml",                # renamed to winnow-e4b (per-checkpoint)
    "nvidia/nemotron-rerank-vl-1b.yaml",    # hosted-only sibling, not a table row
    "altslate/certo.yaml",                  # trials/engines only, not a table row
    "heman10x/verdict.yaml",                # trials/engines only
    "supersoniclabs/julia.yaml",            # trials/engines only
    "pngwn/nanodiff.yaml",                  # trials/engines only
    "notnotsamuel/lfm2.5-rlcd.yaml",        # trials/engines only
    "ikermoel/so1.yaml",                    # trials/engines only
]

# Runner islands per docs/plan/03-folder-structure.md.
RUNNER_GLINER = "gliner"              # GLiNER2/2.5, GLiFormer, GLiREL, Relex,
                                      # GLiClass, Laya — transformers 4.57.x
RUNNER_VON = "von"                    # von SDK — transformers 5.x, py3.13
RUNNER_RELIK = "relik"                # py3.11, torch 2.3.1, Windows csv shim
RUNNER_SYSTEMONE_LOCAL = "systemone-local"  # Kev/decider/OpenThai :8009/8018/8029
RUNNER_SMALL_DECISIONS = "small-decisions"  # in-process + Ollaya-served smalls


def rt(engine, dtype=None, quant=None, device="cpu"):
    return {"engine": engine, "dtype": dtype, "quant": quant, "device": device}


def req(ram_gb=None, note=None):
    """Best-effort hardware requirements (CPU-only bench machine). ram_gb is
    an estimate from README sizes unless the notes say otherwise."""
    out = {"vram_gb": 0, "ram_gb": ram_gb, "accel": "cpu"}
    if note:
        out["note"] = note
    return out


def local_dep(version_id, runner, runtime, requires, notes=None, used_as=None):
    dep = {"id": version_id.split("@", 1)[0] + ".local", "kind": "local",
           "version": version_id, "runner": runner, "runtime": runtime,
           "requires": requires}
    if used_as:
        dep["used_as"] = used_as
    if notes:
        dep["notes"] = notes
    return dep


def ollaya_dep(tag, version_id, runtime=None, requires=None, notes=None,
               dep_id=None):
    """A deployment served by the local Ollaya daemon (:11435, System One
    contract, weights pinned + sha256-verified by Ollaya; free, no key)."""
    runtime = runtime or rt(None)
    requires = requires or req()
    base = ("served by the local Ollaya daemon on http://127.0.0.1:11435/v1/systemone "
            "(TypeSafe System One contract, daemon 0.12.1 here; `ollaya pull` per tag; "
            "one request per question — the decision layers build their premise from "
            "the state)")
    return {"id": dep_id or f"{tag.replace(':', '-')}.ollaya", "kind": "local",
            "version": version_id, "runner": RUNNER_SMALL_DECISIONS,
            "runtime": runtime, "requires": requires,
            "notes": notes + " " + base if notes else base}


def ver(vid, hf_repo=None, hf_revision=None, api_slug=None, released=None,
        params_b=None, license_=None, notes=None):
    v = {"id": vid, "hf_revision": hf_revision, "api_slug": api_slug,
         "released": released}
    if params_b is not None:
        v["params_b"] = params_b
    if license_ is not None:
        v["license"] = license_
    if hf_repo is not None:
        v["hf_repo"] = hf_repo
    if notes:
        v["notes"] = notes
    return v


def m(org, org_name, slug, family, license_, params_b, tracks, modalities,
      languages, card, v0_names, version, deployments, notes=None,
      open_weights=True):
    return {
        "org": org, "org_name": org_name, "slug": slug, "family": family,
        "license": license_, "open_weights": open_weights, "params_b": params_b,
        "tracks": tracks, "modalities": modalities, "languages": languages,
        "card": card, "v0_names": v0_names, "version": version,
        "deployments": deployments, "notes": notes,
    }


# ---------------------------------------------------------------------------
# One entry per README system-table system (one per benched checkpoint).
# Facts: sizes/licenses/links from the README system table + Family maps;
# revisions/dtype/quant/device/ports from legacy/engines/*.
# ---------------------------------------------------------------------------
MODELS = [
    # -- extractors (gliner island; Knowledgator's GLiNER lineage -> family gliner)
    m("knowledgator", "Knowledgator", "gliformer-base", "gliner", "Apache 2.0",
      0.19, ["classify", "ner"], ["text"], None,
      "https://github.com/Knowledgator/GLiFormer", ["GLiFormer-base"],
      ver("gliformer-base-v1@unversioned", hf_repo="knowledgator/gliformer-base-v1",
          params_b=0.19,
          notes="~190M; benched; best NER F1 on the mixed pool (100% exact)"),
      [local_dep("gliformer-base-v1@unversioned", RUNNER_GLINER, rt("torch"),
                 req(1, "ram_gb estimated from params (fp32)"),
                 "gliner2/GLiFormer package, threshold 0.5, one call per question")],
      notes="Family gliner per the README's GLiNER-lineage language: the lineage forked "
            "between Fastino (GLiNER 2.5) and Knowledgator (classic gliner plus GLiFormer "
            "and GLiClass)."),
    m("knowledgator", "Knowledgator", "gliformer-large", "gliner", "Apache 2.0",
      0.5756, ["classify", "ner"], ["text"], None,
      "https://github.com/Knowledgator/GLiFormer", ["GLiFormer-large"],
      ver("gliformer-large-v1@unversioned", hf_repo="knowledgator/gliformer-large-v1",
          params_b=0.5756,
          notes="575.6M; benched; family is base+large only, no small"),
      [local_dep("gliformer-large-v1@unversioned", RUNNER_GLINER, rt("torch"),
                 req(3, "ram_gb estimated from params (fp32)"))],
      notes="Knowledgator's GLiNER lineage (family gliner); see gliformer-base."),
    m("knowledgator", "Knowledgator", "gliner-relex", "gliner", "Apache 2.0",
      0.319, ["ner", "relation"], ["text"], "multilingual",
      "https://huggingface.co/knowledgator/gliner-relex-multi-v1.0", [],
      ver("gliner-relex-multi-v1.0@unversioned",
          hf_repo="knowledgator/gliner-relex-multi-v1.0", params_b=0.319,
          notes="~319M; zero-shot joint NER + relations in one pass; classic gliner line "
                "(Knowledgator's branch of the GLiNER lineage)"),
      [local_dep("gliner-relex-multi-v1.0@unversioned", RUNNER_GLINER, rt("torch"),
                 req(2, "ram_gb estimated from params (fp32)"),
                 "classic gliner package")],
      notes="README system-table row that was never benched (demoed only in "
            "demos/demo_gliner_relex.py), so v0.benched is false."),
    m("knowledgator", "Knowledgator", "gliclass-modern-base", "gliner",
      "Apache 2.0", 0.151, ["classify"], ["text"], None,
      "https://github.com/knowledgator/gliclass", ["gliclass-modern-base"],
      ver("gliclass-modern-base-v3.0@unversioned",
          hf_repo="knowledgator/gliclass-modern-base-v3.0", params_b=0.151,
          notes="151M; benched"),
      [local_dep("gliclass-modern-base-v3.0@unversioned", RUNNER_GLINER, rt("torch"),
                 req(1, "ram_gb estimated from params (fp32)"),
                 "gliclass package, ZeroShotClassificationPipeline, multi-label, "
                 "threshold 0.0 + argmax")],
      notes="Knowledgator's GLiNER lineage (family gliner). The edge-v3.0 size (33M) is "
            "CULLED from the benchmark (TRIALS.md: 66.7% on the trial pool) and gets no "
            "catalog file in the per-checkpoint split."),
    m("knowledgator", "Knowledgator", "gliclass-base", "gliner", "Apache 2.0",
      0.187, ["classify"], ["text"], None,
      "https://github.com/knowledgator/gliclass", ["gliclass-base"],
      ver("gliclass-base-v3.0@unversioned", hf_repo="knowledgator/gliclass-base-v3.0",
          params_b=0.187, notes="187M; benched"),
      [local_dep("gliclass-base-v3.0@unversioned", RUNNER_GLINER, rt("torch"),
                 req(2, "ram_gb estimated from params (fp32)"),
                 "gliclass package, ZeroShotClassificationPipeline, multi-label, "
                 "threshold 0.0 + argmax")]),
    m("knowledgator", "Knowledgator", "gliclass-large", "gliner", "Apache 2.0",
      0.439, ["classify"], ["text"], None,
      "https://github.com/knowledgator/gliclass", ["gliclass-large"],
      ver("gliclass-large-v3.0@unversioned", hf_repo="knowledgator/gliclass-large-v3.0",
          params_b=0.439, notes="439M; benched"),
      [local_dep("gliclass-large-v3.0@unversioned", RUNNER_GLINER, rt("torch"),
                 req(3, "ram_gb estimated from params (fp32)"),
                 "gliclass package, ZeroShotClassificationPipeline, multi-label, "
                 "threshold 0.0 + argmax")]),

    # -- unbenched README system-table rows -----------------------------------
    m("jackboyla", "jackboyla", "glirel", "glirel", "CC BY-NC-SA 4.0", 0.467,
      ["relation"], ["text"], None, "https://github.com/jackboyla/GLiREL", [],
      ver("glirel-large-v0@unversioned", hf_repo="jackboyla/glirel-large-v0",
          params_b=0.467,
          notes="~467M; zero-shot relation extraction over entity pairs handed to it "
                "(the demo feeds it GLiNER 2.5 spans)"),
      [local_dep("glirel-large-v0@unversioned", RUNNER_GLINER, rt("torch"),
                 req(3, "ram_gb estimated from params (fp32)"),
                 "needs entity spans as input")],
      notes="README system-table row that was never benched (demoed only in "
            "demos/demo_glirel.py), so v0.benched is false."),
    m("relik-ie", "SapienzaNLP (relik-ie)", "relik", "relik",
      "HF card Apache 2.0; repo has no LICENSE file and its README footer says "
      "CC BY-NC-SA 4.0",
      0.18, ["relation", "ner"], ["text"], None,
      "https://github.com/SapienzaNLP/relik", [],
      ver("relik-relation-extraction-small@unversioned",
          hf_repo="relik-ie/relik-relation-extraction-small", params_b=0.18,
          notes="33.4M E5-small retriever + 146.5M DeBERTa-v3 reader (~180M); closed "
                "Wikidata-property vocabulary, untyped reader spans; the demoed joint "
                "NER+relation sibling (relik-ie/relik-relation-extraction-ner-small, 10 "
                "Wikipedia-trained entity types) needs two load-time fixes in "
                "engines/relik_client.py"),
      [local_dep("relik-relation-extraction-small@unversioned", RUNNER_RELIK,
                 rt("torch"), req(3, "ram_gb estimated"),
                 "relik 1.0.7, torch 2.3.1 + faiss-cpu, py3.11 venv (.venv-relik); on "
                 "Windows import needs the csv.field_size_limit shim in "
                 "engines/relik_client.py")],
      notes="README system-table row that was never benched (demoed only in "
            "demos/demo_relik.py), so v0.benched is false."),

    # -- cross-encoder rerankers (native rerank, benched used_as classify) -----
    m("mixedbread-ai", "Mixedbread AI", "mxbai-rerank-base-v2", "rerank-local",
      "Apache 2.0", 0.494, ["rerank"], ["text"], None,
      "https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v2",
      ["mxbai-rerank-base-v2"],
      ver("mxbai-rerank-base-v2@unversioned",
          hf_repo="mixedbread-ai/mxbai-rerank-base-v2", params_b=0.494,
          notes="494M cross-encoder; benched"),
      [local_dep("mxbai-rerank-base-v2@unversioned", RUNNER_GLINER, rt("torch"),
                 req(3, "ram_gb estimated from params (fp32)"),
                 "sentence-transformers CrossEncoder on CPU; scores one (instruction, "
                 "label) pair per label, argmax = decision", used_as="classify")],
      notes="Family rerank-local: the README bundles the local cross-encoder rerankers "
            "in one system-table row (mirror of the hosted pass's rerank-hosted)."),
    m("mixedbread-ai", "Mixedbread AI", "mxbai-rerank-large-v2", "rerank-local",
      "Apache 2.0", 1.54, ["rerank"], ["text"], None,
      "https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2",
      ["mxbai-rerank-large-v2"],
      ver("mxbai-rerank-large-v2@unversioned",
          hf_repo="mixedbread-ai/mxbai-rerank-large-v2", params_b=1.54,
          notes="1.54B cross-encoder; benched"),
      [local_dep("mxbai-rerank-large-v2@unversioned", RUNNER_GLINER, rt("torch"),
                 req(9, "ram_gb estimated from params (fp32)"),
                 "sentence-transformers CrossEncoder on CPU", used_as="classify")],
      notes="Family rerank-local (bundled local-rerankers row)."),
    m("baai", "BAAI", "bge-reranker-v2-m3", "rerank-local", "Apache 2.0", 0.568,
      ["rerank"], ["text"], "multilingual",
      "https://huggingface.co/BAAI/bge-reranker-v2-m3", ["bge-reranker-v2-m3"],
      ver("bge-reranker-v2-m3@unversioned", hf_repo="BAAI/bge-reranker-v2-m3",
          params_b=0.568,
          notes="568M XLM-RoBERTa-large cross-encoder; benched; best local reranker on "
                "the mixed pool (72.9%)"),
      [local_dep("bge-reranker-v2-m3@unversioned", RUNNER_GLINER, rt("torch"),
                 req(3, "ram_gb estimated from params (fp32)"),
                 "sentence-transformers CrossEncoder on CPU", used_as="classify")],
      notes="Family rerank-local (bundled local-rerankers row)."),
    m("alibaba-nlp", "Alibaba-NLP", "gte-rerank-modernbert-base", "rerank-local",
      "Apache 2.0", 0.15, ["rerank"], ["text"], None,
      "https://huggingface.co/Alibaba-NLP/gte-rerank-modernbert-base",
      ["GTE-rerank-ModernBERT-base"],
      ver("gte-rerank-modernbert-base@unversioned",
          hf_repo="Alibaba-NLP/gte-rerank-modernbert-base", params_b=0.15,
          notes="150M ModernBERT-base cross-encoder; benched"),
      [local_dep("gte-rerank-modernbert-base@unversioned", RUNNER_GLINER, rt("torch"),
                 req(1, "ram_gb estimated from params (fp32)"),
                 "sentence-transformers CrossEncoder on CPU", used_as="classify")],
      notes="Family rerank-local (bundled local-rerankers row)."),
    m("nvidia", "NVIDIA", "nemotron-rerank-1b-v2", "rerank-local",
      "openmdw-1.1 + Llama 3.2 Community License", 1.24, ["rerank"], ["text"], None,
      "https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2",
      ["nemotron-rerank-1b-v2"],
      ver("llama-nemotron-rerank-1b-v2@unversioned",
          hf_repo="nvidia/llama-nemotron-rerank-1b-v2", params_b=1.24,
          notes="1.24B bidirectional-attention Llama-3.2-1B cross-encoder; head ships as "
                "remote code (LlamaBidirectionalForSequenceClassification); weights "
                "refreshed 2026-08-26 per bench notes; benched with the card's verbatim "
                "pair format 'question:{q} \\n \\n passage:{p}' (raw text + described "
                "labels, 31/48)"),
      [local_dep("llama-nemotron-rerank-1b-v2@unversioned", RUNNER_GLINER,
                 rt("torch", dtype="fp32"),
                 req(7, "ram_gb estimated from params (fp32)"),
                 "trust_remote_code load, left padding, fp32 on CPU (the card's bf16 "
                 "example targets GPU)", used_as="classify")],
      notes="Family rerank-local (bundled local-rerankers row). The hosted-only vl "
            "sibling (nvidia/llama-nemotron-rerank-vl-1b-v2:free, culled) is not a "
            "README system-table row and gets no file."),

    # -- typed-decision engines ------------------------------------------------
    m("convaiinnovations", "ConvaiInnovations", "laya", "laya", "Apache 2.0",
      0.421, ["decision", "score"], ["text"], None,
      "https://huggingface.co/convaiinnovations/laya", ["Laya (local)"],
      ver("laya@unversioned", hf_repo="convaiinnovations/laya", params_b=0.421,
          notes="421M English root; benched as 'Laya (local)' (the multilingual pool's "
                "'Laya Router' mode routes between this and the typed-decisions "
                "checkpoint); the repo's 322M multilingual subfolder is not separately "
                "benched"),
      [local_dep("laya@unversioned", RUNNER_GLINER, rt("torch"),
                 req(3, "ram_gb estimated from params (fp32)"),
                 "in-process via the `laya` package (laya.load + agent.predict, one "
                 "batched call per task)")]),
    m("convaiinnovations", "ConvaiInnovations", "laya-typed-decisions", "laya",
      "Apache 2.0", None, ["decision", "score"], ["text"], None,
      "https://huggingface.co/convaiinnovations/laya-typed-decisions",
      ["Laya typed-decisions"],
      ver("laya-typed-decisions@unversioned",
          hf_repo="convaiinnovations/laya-typed-decisions",
          notes="typed-decisions variant; benched as 'Laya typed-decisions'; its param "
                "count is not documented (the table row's 322M figure is the "
                "multilingual subfolder)"),
      [local_dep("laya-typed-decisions@unversioned", RUNNER_GLINER, rt("torch"),
                 req(3, "ram_gb estimated"),
                 "same in-process `laya` package path")]),
    m("wfzyx", "wfzyx", "von", "von", "Apache 2.0", 0.396,
      ["decision", "score"], ["text"], None,
      "https://huggingface.co/wfzyx/von-1.0", ["von"],
      ver("von-1.0@fbc0196efa4fa1825557a8d5adad464fa1e872bb",
          hf_repo="wfzyx/von-1.0",
          hf_revision="fbc0196efa4fa1825557a8d5adad464fa1e872bb", params_b=0.396,
          notes="396M complete trained option-marker checkpoint (config.json + "
                "model.safetensors + option_marker.pt + marker_calibration.json); "
                "loaded with the pinned Git von SDK revision "
                "a94aa368ff4dcc58cdfd569cf0b4b1ae7d592d54 (the PyPI SDK's default NLI "
                "loader invents a random classifier — the loader refuses to run without "
                "option_marker.pt); the HF card now describes von-1.2, this repo pins "
                "the von-1.0 weights"),
      [local_dep("von-1.0@fbc0196efa4fa1825557a8d5adad464fa1e872bb", RUNNER_VON,
                 rt("torch"), req(3, "ram_gb estimated"),
                 "OptionMarkerBackend, device cpu, temperature from "
                 "marker_calibration.json; runs in .venv-von (requirements-von.txt, "
                 "transformers 5.x py3.13)")]),
    m("malevrigns", "malevrigns", "agentjev", "agentjev", "Apache 2.0", 0.6,
      ["decision", "score"], ["text"], None,
      "https://github.com/malevrigns/agent-jev", ["AgentJev 0.6B (local)"],
      ver("agentjev-v1@unversioned", hf_repo="aimeigaoshou/agent-jev", params_b=0.6,
          notes="0.6B candidate head over Qwen3-0.6B (LM head swapped); weights convert "
                "to agentjev_v1.pt once per its model card, served with "
                "temperatures.json; single checkpoint; benched"),
      [local_dep("agentjev-v1@unversioned", RUNNER_SYSTEMONE_LOCAL,
                 rt("torch", dtype="fp32"), req(3, "ram_gb estimated"),
                 "own HTTP contract POST /api/evaluate (NOT System One) on "
                 "http://127.0.0.1:8149 — jev_service.server --checkpoint "
                 "agentjev_v1.pt --device cpu, own venv; fp32 (autocast is CUDA-only, "
                 "engines/agentjev_client.py)")]),
    m("mapika", "Mapika", "decider", "decider", "Apache 2.0", 0.8,
      ["decision", "score"], ["text"], None,
      "https://huggingface.co/Mapika/decider-0.8b", ["decider 0.8B (local)"],
      ver("decider-0.8b@unversioned", hf_repo="Mapika/decider-0.8b", params_b=0.8,
          notes="0.8B (752M) Qwen3.5-0.8B-Base fine-tune; served by the PyPI "
                "decider-ai[serve] System One server; benched; the larger "
                "decider-2b / decider-35b-a3b siblings are 'Noted, not benchmarked' "
                "rows (GPU-priced) and get no catalog file"),
      [local_dep("decider-0.8b@unversioned", RUNNER_SYSTEMONE_LOCAL, rt("torch"),
                 req(5, "ram_gb estimated"),
                 "local System One server on http://127.0.0.1:8018/v1/systemone "
                 "(uvicorn decider.serve:app, DECIDER_MODEL=Mapika/decider-0.8b "
                 "DECIDER_DEVICE=cpu); lazy-loads on first request so the bench fires "
                 "an untimed warm-up; the shared server venv installs onnxruntime — "
                 "engine not documented per model")]),
    m("iapp", "iApp", "openthai", "openthai",
      "Apache 2.0 package; weights gated on HF", 0.8,
      ["decision", "score"], ["text"], "th, en",
      "https://huggingface.co/iapp/OpenThai-SystemOne", ["OpenThai 0.8B (local)"],
      ver("openthai-systemone@unversioned", hf_repo="iapp/OpenThai-SystemOne",
          params_b=0.8,
          notes="0.8B Qwen3.5 Gated DeltaNet hybrid + 256-way slot head, Thai/English; "
                "served by the PyPI openthai-systemone[server] package; weights gated "
                "on HF; benched"),
      [local_dep("openthai-systemone@unversioned", RUNNER_SYSTEMONE_LOCAL, rt("torch"),
                 req(5, "ram_gb estimated"),
                 "local System One server on http://127.0.0.1:8029/v1/systemone "
                 "(uvicorn openthai_systemone.server:app, "
                 "OPENTHAI_SYSTEMONE_MODEL=iapp/OpenThai-SystemOne); lazy cold load "
                 "runs minutes (warm-up timeout 600 s); the shared server venv "
                 "installs onnxruntime — engine not documented per model")]),
    m("alibiserikbay", "alibiserikbay", "jevk5-lite", "jevk5", "Apache 2.0",
      0.437, ["decision"], ["text"], None,
      "https://huggingface.co/alibiserikbay/JevK5-Lite", ["JevK5-Lite"],
      ver("jevk5-lite@unversioned", hf_repo="alibiserikbay/JevK5-Lite",
          params_b=0.437,
          notes="437M label-head DeBERTa-v3-large encoder, the lite build of JevK5 "
                "(JevBench #3); `jevk5` package (jevk5[lite]==0.3.1 in .venv-von), "
                "threads=16; benched"),
      [local_dep("jevk5-lite@unversioned", RUNNER_SMALL_DECISIONS, rt("torch"),
                 req(3, "ram_gb estimated from params (fp32)"),
                 "in-process JevK5Lite.from_pretrained in .venv-von "
                 "(transformers 5.17)")]),
    m("alibiserikbay", "alibiserikbay", "jevk5-4b", "jevk5", "Apache 2.0", 4,
      ["decision"], ["text"], None,
      "https://huggingface.co/alibiserikbay/JevK5-GGUF", ["jevk5 4B (Ollaya)"],
      ver("jevk5-4b@unversioned", hf_repo="alibiserikbay/JevK5-GGUF", api_slug="jevk5",
          params_b=4,
          notes="the full JevK5 v0.3 (Qwen3.5 fine-tune), Q8_0 GGUF on llama.cpp; "
                "labels A-P read from next-token probabilities; Ollaya pins the "
                "authors' HF files to a commit and sha256-verifies them; benched"),
      [ollaya_dep("jevk5", "jevk5-4b@unversioned",
                  runtime=rt("llama.cpp", quant="Q8_0"),
                  requires=req(5, "ram_gb estimated from params (Q8_0 ~1 byte/param "
                                  "+ overhead)"))]),
    m("molemo-lab", "MoLeMo-Lab", "mojev", "mojev",
      "engine MIT; checkpoint card MIT (Qwen base-model license on the encoder "
      "weights)",
      0.85, ["decision", "score"], ["text"], None,
      "https://huggingface.co/MoLeMo-Lab/mojev", ["MoJev 0.85B"],
      ver("mojev@0c8695b6252f4205907433d4e196a94f032e60c3",
          hf_repo="MoLeMo-Lab/mojev",
          hf_revision="0c8695b6252f4205907433d4e196a94f032e60c3", params_b=0.85,
          notes="0.85B Qwen3.5 + fla packed one-pass scorer (choice/score/noul in one "
                "forward); scorer class loads from the checkpoint via trust_remote_code; "
                "request path adapted in legacy/engines/mojev_engine/ from the GitHub "
                "runtime (revision a74d58cd19ec573e83e8e27f9fecd837b8d830fb); benched"),
      [local_dep("mojev@0c8695b6252f4205907433d4e196a94f032e60c3",
                 RUNNER_SMALL_DECISIONS, rt("torch"), req(5, "ram_gb estimated"),
                 "in-process in .venv-von (transformers 5.17 for the Qwen3.5 encoder; "
                 "fla reference kernels on CPU)")]),
    m("frontiersmind", "FrontiersMind", "lumma-fev-0.1b", "lumma", "Apache 2.0",
      0.154, ["decision", "score"], ["text"], None,
      "https://huggingface.co/FrontiersMind/Lumma-fev-0.1b", ["Lumma-fev 0.15B"],
      ver("lumma-fev-0.1b@unversioned", hf_repo="FrontiersMind/Lumma-fev-0.1b",
          params_b=0.154,
          notes="154M (card name 'Lumma-Fev-0.15B'); causal backbone + pointer head "
                "over choice/score/noul rows; benched"),
      [local_dep("lumma-fev-0.1b@unversioned", RUNNER_SMALL_DECISIONS, rt("torch"),
                 req(2, "ram_gb estimated from params (fp32)"),
                 "in-process via the lumma-fev package (.venv-von, transformers "
                 ">=5.4,<6); one decide() call per question")]),
    m("frontiersmind", "FrontiersMind", "lumma-fev-0.6b", "lumma", "Apache 2.0",
      0.649, ["decision", "score"], ["text"], None,
      "https://huggingface.co/FrontiersMind/Lumma-fev-0.6b", ["Lumma-fev 0.6B"],
      ver("lumma-fev-0.6b@unversioned", hf_repo="FrontiersMind/Lumma-fev-0.6b",
          params_b=0.649, notes="649M same pointer-head architecture; benched"),
      [local_dep("lumma-fev-0.6b@unversioned", RUNNER_SMALL_DECISIONS, rt("torch"),
                 req(4, "ram_gb estimated from params (fp32)"),
                 "same lumma-fev in-process path")]),
    m("frontiersmind", "FrontiersMind", "lumma-fev-4b", "lumma", "Apache 2.0",
      4.2, ["decision", "score"], ["text"], None,
      "https://huggingface.co/FrontiersMind/Lumma-fev-4b", ["Lumma-fev 4B"],
      ver("lumma-fev-4b@unversioned", hf_repo="FrontiersMind/Lumma-fev-4b",
          params_b=4.2,
          notes="4.2B, largest checkpoint practical on the 32 GB CPU box; benched; the "
                "7.9B sibling (Lumma-fev-9b) is a 'Noted, not benchmarked' row (~16 GB "
                "bf16) and gets no catalog file"),
      [local_dep("lumma-fev-4b@unversioned", RUNNER_SMALL_DECISIONS, rt("torch"),
                 req(17, "ram_gb estimated from params (fp32 residency on the 32 GB "
                         "box)"),
                 "same lumma-fev in-process path")]),
    m("internlm", "internlm", "intern-decision-0.8b", "intern-decision",
      "Apache 2.0 (Qwen3.5 upstream weights also Apache 2.0)", 0.853,
      ["decision", "score"], ["text", "image"], None,
      "https://huggingface.co/collections/internlm/intern-decision",
      ["Intern-Decision 0.8B"],
      ver("intern-decision-0.8b@unversioned",
          hf_repo="internlm/Intern-Decision-0.8B", params_b=0.853,
          notes="853M Qwen3.5-0.8B decision fine-tune; one causal forward over a masked "
                "assistant JSON skeleton, option-symbol logits at the <decision> slots; "
                "runtime ships inside the snapshot (standalone inference.py, "
                "per-checkpoint calibration temperature); vision tower accepts 1-8 "
                "images per request; benched"),
      [local_dep("intern-decision-0.8b@unversioned", RUNNER_SMALL_DECISIONS,
                 rt("torch", dtype="fp32"),
                 req(5, "ram_gb estimated from params (fp32)"),
                 "in-process in .venv-von (torch 2.14.0+cpu, transformers 5.17; "
                 "snapshot pins torch 2.9.1/transformers 5.14.1 — satisfied); fp32 CPU "
                 "family-wide (bf16 runs ~7x slower with identical predictions, "
                 "engines/intern_decision_client.py); INTERN_DECISION_HOME overrides "
                 "the snapshot prefix")]),
    m("internlm", "internlm", "intern-decision-2b", "intern-decision",
      "Apache 2.0 (Qwen3.5 upstream weights also Apache 2.0)", 2.2,
      ["decision", "score"], ["text", "image"], None,
      "https://huggingface.co/collections/internlm/intern-decision",
      ["Intern-Decision 2B"],
      ver("intern-decision-2b@unversioned", hf_repo="internlm/Intern-Decision-2B",
          params_b=2.2,
          notes="2.2B, same masked-slot mechanism, own calibration temperature; benched"),
      [local_dep("intern-decision-2b@unversioned", RUNNER_SMALL_DECISIONS,
                 rt("torch", dtype="fp32"),
                 req(11, "ram_gb estimated from params (fp32)"),
                 "same in-process snapshot-runtime path")]),
    m("internlm", "internlm", "intern-decision-4b", "intern-decision",
      "Apache 2.0 (Qwen3.5 upstream weights also Apache 2.0)", 4.5,
      ["decision", "score"], ["text", "image"], None,
      "https://huggingface.co/collections/internlm/intern-decision",
      ["Intern-Decision 4B"],
      ver("intern-decision-4b@unversioned", hf_repo="internlm/Intern-Decision-4B",
          params_b=4.5,
          notes="4.5B, largest of the three, own calibration temperature; benched"),
      [local_dep("intern-decision-4b@unversioned", RUNNER_SMALL_DECISIONS,
                 rt("torch", dtype="fp32"),
                 req(22, "ram_gb estimated from params (fp32 residency on the 32 GB "
                         "box)"),
                 "same in-process snapshot-runtime path")]),
    m("ifm", "IFM", "k2-type", "k2-type", "Apache 2.0", 0.9,
      ["decision", "score"], ["text"], None,
      "https://huggingface.co/IFM/K2-Type-0.9B", ["K2-Type 0.9B (local)"],
      ver("k2-type-0.9b@unversioned", hf_repo="IFM/K2-Type-0.9B", params_b=0.9,
          notes="0.9B active (1.08B stored incl. the unused LM head) on the "
                "K2-Horizon-0.9B backbone; state encoded once, block-causal mask "
                "isolates each question, pointer head scores every option in one "
                "forward, softmax at fitted temperature 1.478; `jev/` runtime ships in "
                "the snapshot; 0.762 JevBench public per the card; benched"),
      [local_dep("k2-type-0.9b@unversioned", RUNNER_SMALL_DECISIONS,
                 rt("torch", dtype="fp32"), req(6, "ram_gb estimated from params (fp32)"),
                 "in-process in .venv-von (the snapshot's own jev.serve server "
                 "hardcodes CUDA); fp32 weights on CPU — the snapshot stores bf16 "
                 "(engines/k2type_client.py); K2TYPE_HOME overrides the snapshot dir")]),

    # -- vLLM Semantic Router Decision line (family decision) -------------------
    m("vllm-sr", "vLLM Semantic Router contributors", "decision", "decision",
      "Apache 2.0", 0.75, ["decision", "score"], ["text"], None,
      "https://huggingface.co/collections/vllm-sr/decision-20",
      ["decision 0.75B (Ollaya)"],
      ver("decision-1.0@unversioned", api_slug="decision", params_b=0.75,
          notes="Decision 1.0 — fully fine-tuned Qwen3.5-0.8B + endpoint head scoring "
                "each option's last token, one forward per question, never generates; "
                "the tag behind Ollaya's `decision`; larger Nox 4B / Lux 9B compute "
                "fp32 (17/32 GB) and are not in the Ollaya library; benched"),
      [ollaya_dep("decision", "decision-1.0@unversioned",
                  requires=req(4, "ram_gb estimated from params (fp32, per the Nox/Lux "
                                  "17/32 GB figures)"))]),
    m("vllm-sr", "vLLM Semantic Router contributors", "decision-2.0-kai",
      "decision", "Apache 2.0", 0.597, ["decision", "score"], ["text"], None,
      "https://huggingface.co/collections/vllm-sr/decision-20",
      ["Decision 2.0 Kai 0.6B"],
      ver("decision-2.0-kai-0.6b@unversioned",
          hf_repo="vllm-sr/Decision-2.0-Kai-0.6B", params_b=0.597,
          notes="597M on Qwen3-0.6B-Base + trained candidate head scoring option "
                "descriptions, one batched forward, no generation; score answers carry "
                "per-level logit offsets (score_bias.json); 8,192-token budget; "
                "self-verifying snapshot; benched"),
      [local_dep("decision-2.0-kai-0.6b@unversioned", RUNNER_SMALL_DECISIONS,
                 rt("torch", dtype="fp32"), req(4, "ram_gb from the documented load "
                                                 "size; fp32 residency is higher"),
                 "in-process in .venv-von via the snapshot's decision2 package; "
                 "self-verifying (SHA-256 manifest + packed parameter counts), load "
                 "~1.5 GB; one batched predict per task (share_context off); Windows "
                 "loader re-keys the vendored fingerprint's path separators "
                 "(engines/decision2_client.py); DECISION2_HOME overrides the snapshot "
                 "prefix")]),
    m("vllm-sr", "vLLM Semantic Router contributors", "decision-2.0-eos",
      "decision", "Apache 2.0", 0.753, ["decision", "score"], ["text"], None,
      "https://huggingface.co/collections/vllm-sr/decision-20",
      ["Decision 2.0 Eos 0.8B"],
      ver("decision-2.0-eos-0.8b@unversioned",
          hf_repo="vllm-sr/Decision-2.0-Eos-0.8B", params_b=0.753,
          notes="753M, a Decision-1.0-Eos fine-tune, 16,384-token budget; benched"),
      [local_dep("decision-2.0-eos-0.8b@unversioned", RUNNER_SMALL_DECISIONS,
                 rt("torch", dtype="fp32"), req(5, "ram_gb estimated"),
                 "same decision2 in-process path; load size not documented")]),
    m("vllm-sr", "vLLM Semantic Router contributors", "decision-2.0-sol",
      "decision", "Apache 2.0", 1.88, ["decision", "score"], ["text"], None,
      "https://huggingface.co/collections/vllm-sr/decision-20",
      ["Decision 2.0 Sol 2B"],
      ver("decision-2.0-sol-2b@unversioned", hf_repo="vllm-sr/Decision-2.0-Sol-2B",
          params_b=1.88,
          notes="1.88B, largest benched size of the family, 16,384-token budget; "
                "benched; the Nox-4B / Lux-9B / Vega-27B siblings are 'Noted, not "
                "benchmarked' rows (fp32 CPU residency) and get no catalog file"),
      [local_dep("decision-2.0-sol-2b@unversioned", RUNNER_SMALL_DECISIONS,
                 rt("torch", dtype="fp32"), req(9, "ram_gb from the documented load "
                                                 "size; fp32 residency is higher"),
                 "same decision2 in-process path; load ~4.8 GB per engine docs")]),

    # -- Decision 3.0 (family d3) ------------------------------------------------
    m("vllm-sr", "vLLM Semantic Router contributors", "d3-edge", "d3",
      "Apache 2.0", 0.594, ["decision", "score"], ["text", "image", "video"], None,
      "https://huggingface.co/collections/vllm-sr/decision-30",
      ["Decision 3.0 Edge 0.6B"],
      ver("d3-edge@unversioned", hf_repo="vllm-sr/d3-edge", params_b=0.594,
          released="2026-10-10",
          notes="594M (0.49B text + 0.10B vision encoder), Qwen3.5-0.8B backbone "
                "fine-tune, the collection's floor; 255-way code-readout head scores "
                "single-token answer codes at the last prompt position, one forward per "
                "question; images + videos ride the same system_one call (benches "
                "exercise the text path); self-verifying snapshot; released 2026-10-10 "
                "('Towards Open Multimodal Foundation Decision Models'); benched"),
      [local_dep("d3-edge@unversioned", RUNNER_SMALL_DECISIONS,
                 rt("torch", dtype="bf16"), req(3, "ram_gb from the documented weight "
                                                  "read"),
                 "in-process in .venv-von via the snapshot's d3_runtime; bf16 backbone "
                 "+ SDPA (hard-coded), fp32 readout + softmax; weight read ~1.2 GB; "
                 "Gated DeltaNet on the transformers reference kernels (the fla kernels "
                 "are GPU-only); D3_HOME overrides the snapshot prefix")]),
    m("vllm-sr", "vLLM Semantic Router contributors", "d3-lite", "d3",
      "Apache 2.0", 0.853, ["decision", "score"], ["text", "image", "video"], None,
      "https://huggingface.co/collections/vllm-sr/decision-30",
      ["Decision 3.0 Lite 0.85B"],
      ver("d3-lite@unversioned", hf_repo="vllm-sr/d3-lite", params_b=0.853,
          released="2026-10-10",
          notes="853M, the full Qwen3.5-0.8B fine-tune (Edge is its pruned sibling); "
                "card self-reports 36.9 public-suite vs Decision 2.0 0.8B's 20.6; "
                "benched"),
      [local_dep("d3-lite@unversioned", RUNNER_SMALL_DECISIONS,
                 rt("torch", dtype="bf16"), req(4, "ram_gb estimated"),
                 "same d3_runtime in-process path; load size not documented")]),
    m("vllm-sr", "vLLM Semantic Router contributors", "d3-nano", "d3",
      "Apache 2.0", 2.21, ["decision", "score"], ["text", "image", "video"], None,
      "https://huggingface.co/collections/vllm-sr/decision-30",
      ["Decision 3.0 Nano 2B"],
      ver("d3-nano@unversioned", hf_repo="vllm-sr/d3-nano", params_b=2.21,
          released="2026-10-10",
          notes="2.21B Qwen3.5-2B fine-tune, largest benched size; card self-reports "
                "42.2 public-suite vs Decision 2.0 2B's 30.1; benched; the d3-mini "
                "4.54B / d3-flash 8.39B / d3 26.1B siblings are 'Noted, not "
                "benchmarked' rows and get no catalog file"),
      [local_dep("d3-nano@unversioned", RUNNER_SMALL_DECISIONS,
                 rt("torch", dtype="bf16"), req(7, "ram_gb from the documented weight "
                                                  "read"),
                 "same d3_runtime in-process path; weight read ~4.4 GB per engine "
                 "docs")]),

    # -- Ollaya-served models ------------------------------------------------
    m("madani", "A. M. Madani", "decima", "decima",
      "Apache-2.0 runtime; per-model weight licenses on the Ollaya catalog pages",
      0.321, ["decision", "score"], ["text"], "multilingual",
      "https://ollaya.dev/library/decima", ["decima 321M (Ollaya)"],
      ver("decima-base-2.0@unversioned", api_slug="decima", params_b=0.321,
          released="2026-10-05",
          notes="base 2.0: 321M mmBERT (100+ languages, 512 ctx) late-interaction "
                "scorer — state and each option encode separately so option order never "
                "changes the answer; ordinal head for scores; benched"),
      [ollaya_dep("decima", "decima-base-2.0@unversioned")]),
    m("madani", "A. M. Madani", "decima-small", "decima",
      "Apache-2.0 runtime; per-model weight licenses on the Ollaya catalog pages",
      0.122, ["decision", "score"], ["text"], "multilingual",
      "https://ollaya.dev/library/decima", ["decima small 122M (Ollaya)"],
      ver("decima-small-1.1@unversioned", api_slug="decima:small", params_b=0.122,
          notes="small 1.1: 122M multilingual-e5-small, the fastest model Ollaya runs "
                "on a CPU; benched"),
      [ollaya_dep("decima:small", "decima-small-1.1@unversioned",
                  dep_id="decima-small.ollaya")]),
    m("logitlab", "logitlab", "snap", "snap",
      "Apache-2.0 runtime; per-model weight licenses on the Ollaya catalog pages",
      2, ["decision"], ["text"], "en, it",
      "https://ollaya.dev/library/snap", ["snap 2B (Ollaya)"],
      ver("snap1-2b@unversioned", api_slug="snap", params_b=2,
          notes="snap1-2b: MiniCPM5-2B fine-tuned on emnlmn's snap-engine prompt, Q8_0 "
                "GGUF on llama.cpp, 8k ctx, English+Italian; one option letter per "
                "option read from next-token probabilities, no calibration; needs the "
                "0.10+ daemon; a q4_k_m variant scores 0.660 vs 0.648 on Ollaya's suite "
                "(README family map); benched"),
      [ollaya_dep("snap", "snap1-2b@unversioned", runtime=rt("llama.cpp", quant="Q8_0"),
                  requires=req(4, "ram_gb estimated from params (Q8_0)"))]),
    m("codekins", "Codekins (Zyot Lab)", "arbiter", "arbiter",
      "Apache-2.0 runtime; per-model weight licenses on the Ollaya catalog pages",
      4.3, ["decision", "score"], ["text"], "multilingual",
      "https://ollaya.dev/library/arbiter", ["arbiter 4B (Ollaya)"],
      ver("arbiter-4b@unversioned", api_slug="arbiter", params_b=4.3,
          notes="Gemma 3 4B IT + LoRA behind a fixed 24-slot head (noul + choice of up "
                "to 16 options + scores of exactly 6 levels), 4.3B Q8_0, 8.8 GB "
                "residency; benched; new best local classifier on the mixed pool at "
                "bench time"),
      [ollaya_dep("arbiter", "arbiter-4b@unversioned",
                  runtime=rt("llama.cpp", quant="Q8_0"),
                  requires=req(9, "8.8 GB Q8_0 per README family map"))]),
    m("txoka", "Txoka", "credence", "credence",
      "Apache-2.0 runtime; per-model weight licenses on the Ollaya catalog pages",
      7.5, ["decision"], ["text"], None,
      "https://ollaya.dev/library/credence", ["credence 7.5B (Ollaya)"],
      ver("credence-v1@unversioned", api_slug="credence", params_b=7.5,
          notes="Credence v1: MiCA refinements of Winnow-E4B, 7.5B Q8_0 (8.7 GB), two "
                "checkpoints trading public accuracy vs calibration (checkpoint split "
                "not named in the sources — the `credence` tag row is the benched one); "
                "the vision tags add Winnow's unchanged projector (benches exercised "
                "text only); neither beats the original Winnow on every metric; benched "
                "for the family tree"),
      [ollaya_dep("credence", "credence-v1@unversioned",
                  runtime=rt("llama.cpp", quant="Q8_0"),
                  requires=req(9, "8.7 GB Q8_0 per README family map"))]),
    m("eldanring", "EldanRing", "winnow-e4b", "winnow", "Apache 2.0", None,
      ["decision"], ["text"], None,
      "https://huggingface.co/EldanRing/Winnow-E4B", ["winnow e4b (Ollaya)"],
      ver("winnow-e4b@unversioned", hf_repo="EldanRing/Winnow-E4B",
          api_slug="winnow:e4b",
          notes="fine-tuned from Gemma 4 E4B IT ('E4B' is an efficiency class, so "
                "params_b stays null here; 8 GB Q8_0 residency), Q8_0 GGUF on "
                "llama.cpp; Ollaya builds the author's prompt and reads the option "
                "letters' logits; first request pulls 8 GB into memory (warm-up before "
                "the timed section); benched; the winnow:12b sibling is a 'Noted, not "
                "benchmarked' row (12.7 GB Q8_0) and gets no catalog file"),
      [ollaya_dep("winnow:e4b", "winnow-e4b@unversioned",
                  runtime=rt("llama.cpp", quant="Q8_0"),
                  requires=req(9, "8 GB Q8_0 per README"))]),
    m("moritzlaurer", "MoritzLaurer", "nli-deberta-v3-large", "nli",
      "MIT (card notes non-commercial training-data parts)", 0.435,
      ["classify"], ["text"], None,
      "https://ollaya.dev/library/nli", ["nli deberta-v3-large (Ollaya)"],
      ver("deberta-v3-large@unversioned", api_slug="nli", params_b=0.435,
          notes="DeBERTa-v3-large 435M zero-shot NLI classifier; every option becomes a "
                "hypothesis scored for entailment, one sequence pair per option, "
                "batched per request; criteria must carry label descriptions (bare "
                "labels measurably hurt the pair encoder); benched"),
      [ollaya_dep("nli", "deberta-v3-large@unversioned")]),
    m("moritzlaurer", "MoritzLaurer", "nli-modernbert-large", "nli", "Apache 2.0",
      0.396, ["classify"], ["text"], None,
      "https://ollaya.dev/library/nli", ["nli modernbert-large (Ollaya)"],
      ver("modernbert-large@unversioned", api_slug="nli:modernbert-large",
          params_b=0.396, notes="ModernBERT-large 396M; benched"),
      [ollaya_dep("nli:modernbert-large", "modernbert-large@unversioned",
                  dep_id="nli-modernbert-large.ollaya")]),
]


def benched_names() -> set[str]:
    """Every v0 system name with recorded results in legacy/results/*.json.

    Only the result-bearing containers keyed by system name are read
    (systems, ner, classification.*, by_language/by_tier/latency/usage).
    The `timing` and `provenance` containers are deliberately excluded: the
    multilingual file carries leftover timing-method keys there for systems
    with no recorded run (LFM2.5-RLCD, the nanodiff pair).
    """
    name_containers = ("systems", "ner", "by_language", "by_tier", "latency",
                       "usage")
    names: set[str] = set()
    for path in sorted(RESULTS_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for container in name_containers:
            node = data.get(container)
            if isinstance(node, dict):
                names.update(k for k in node if isinstance(k, str))
        cls = data.get("classification")
        if isinstance(cls, dict):
            for sub in cls.values():
                if isinstance(sub, dict):
                    names.update(k for k in sub if isinstance(k, str))
    return names


def remove_stale() -> list[str]:
    removed = []
    for rel in STALE_OUTPUTS:
        if rel.endswith(".keep"):  # placeholder guard entry, never written
            continue
        path = OUT_DIR / rel
        if OUT_DIR not in path.parents:
            raise SystemExit(f"refusing to remove outside {OUT_DIR}: {rel}")
        if path.is_file():
            path.unlink()
            removed.append(rel)
        org_dir = path.parent
        if org_dir.is_dir() and not any(org_dir.iterdir()):
            org_dir.rmdir()
    return removed


def write_models(benched: set[str]) -> list[Path]:
    written: list[Path] = []
    seen_ids: dict[str, str] = {}
    seen_deployment_ids: set[str] = set()
    for entry in MODELS:
        if entry["org"] in FORBIDDEN_ORGS:
            raise SystemExit(f"refusing to write into forbidden org dir {entry['org']!r}")
        matched = [n for n in entry["v0_names"] if n in benched]
        missing = [n for n in entry["v0_names"] if n not in benched]
        doc = {
            "id": f"{entry['org']}/{entry['slug']}",
            "family": entry["family"],
            "org": entry["org_name"],
            "license": entry["license"],
            "open_weights": entry["open_weights"],
            "params_b": entry["params_b"],
            "claims": {
                "tracks": entry["tracks"],
                "modalities": entry["modalities"],
                "languages": entry["languages"],
            },
            "status": "active",
            "links": {"card": entry["card"], "paper": None},
            "versions": [entry["version"]],
            "deployments": entry["deployments"],
            "v0": {"benched": bool(matched)},
        }
        if entry["notes"]:
            doc["notes"] = entry["notes"]
        if entry["v0_names"] and not matched:
            doc["v0"]["note"] = ("none of the model's v0 system names appear in "
                                 "legacy/results/*.json")
        if missing and matched:
            doc["v0"]["note"] = ("v0 names not present in legacy/results/*.json: "
                                 + ", ".join(missing))

        if doc["id"] in seen_ids:
            raise SystemExit(f"duplicate model id {doc['id']!r}")
        seen_ids[doc["id"]] = entry["org"]
        for dep in doc["deployments"]:
            if dep["version"] != entry["version"]["id"]:
                raise SystemExit(f"{doc['id']}: deployment {dep['id']} references "
                                 f"{dep['version']!r}, not the file's version")
            if dep["id"] in seen_deployment_ids:
                raise SystemExit(f"duplicate deployment id {dep['id']!r}")
            seen_deployment_ids.add(dep["id"])

        out_dir = OUT_DIR / entry["org"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{entry['slug']}.yaml"
        header = ("# Generated by legacy/extract_catalog_local.py (one-off v0 extraction;\n"
                  "# docs/plan/04-data-model.md shape + transitional v0 bookkeeping).\n"
                  "# Sources: README.md system table + Family maps, legacy/engines/*,\n"
                  "# legacy/results/*.json (v0.benched), TRIALS.md (culled systems).\n")
        text = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=98)
        out_path.write_text(header + text, encoding="utf-8")
        written.append(out_path)
    return written


def main() -> None:
    removed = remove_stale()
    benched = benched_names()
    written = write_models(benched)

    # verify: every file parses, ids are unique, families are as intended
    ids, families, benched_count = [], set(), 0
    for path in written:
        with open(path, encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        if not isinstance(doc, dict) or "id" not in doc:
            raise SystemExit(f"{path}: did not parse to a catalog model dict")
        ids.append(doc["id"])
        families.add(doc["family"])
        if doc["v0"]["benched"]:
            benched_count += 1
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate ids after round-trip")

    orgs = sorted({p.parent.name for p in written})
    union = families | HOSTED_FAMILIES
    print(f"removed {len(removed)} stale file(s): {', '.join(removed) or '-'}")
    print(f"wrote {len(written)} model files under {OUT_DIR}")
    print(f"org dirs ({len(orgs)}): {', '.join(orgs)}")
    print(f"local families ({len(families)}): {', '.join(sorted(families))}")
    print(f"benched: {benched_count}/{len(written)}")
    print(f"repo-wide family union with hosted pass: {len(union)} "
          f"(hosted {len(HOSTED_FAMILIES)} + local {len(families)} "
          f"- shared {len(families & HOSTED_FAMILIES)})")
    if len(union) != 38:
        print("WARNING: union != 38")


if __name__ == "__main__":
    main()
