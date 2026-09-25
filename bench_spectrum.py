r"""Spectrum benchmark: one mixed pool of questions per ability.

All classification questions (sentiment + topic) form one pool of 48; NER
questions form one pool of 18. Every system answers every question; a
question's difficulty is computed afterwards as the fraction of systems
that got it wrong (continuous 0.0-1.0).

Per-question predictions are stored (not just aggregates) so the app can
plot accuracy along the difficulty spectrum.

Run from the MAIN venv for most systems, from .venv-von for von,
JevK5-Lite, LFM2.5-RLCD 350M and MoJev 0.85B:
    python bench_spectrum.py --system GLiNER2.5-base
    ...
    .venv-von/Scripts/python bench_spectrum.py --system von
    .venv-von/Scripts/python bench_spectrum.py --system JevK5-Lite
    .venv-von/Scripts/python bench_spectrum.py --system "LFM2.5-RLCD 350M"
    .venv-von/Scripts/python bench_spectrum.py --system "MoJev 0.85B"

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

Output: bench_spectrum_results.json
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

RESULTS_FILE = "bench_spectrum_results.json"

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
RERANKERS = {
    "mxbai-rerank-base-v2": "mixedbread-ai/mxbai-rerank-base-v2",
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    "GTE-rerank-ModernBERT-base": "Alibaba-NLP/gte-reranker-modernbert-base",
}
ALL_SYSTEMS = (list(EXTRACTORS) + list(GLICLASS) + list(RERANKERS)
               + ["Certo 421M", "MoJev 0.85B", "nanodiff 350M",
                  "Laya (local)", "Laya typed-decisions", "Jev",
                  "Kev 0.8B (local)", "AgentJev 0.6B (local)",
                  "decider 0.8B (local)", "OpenThai 0.8B (local)",
                  "Verdict 151M (local)", "von", "JevK5-Lite",
                  "LFM2.5-RLCD 350M", "so1 (Qwen2.5-0.5B)"])

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


def classify_batched(client_kind: str, repo: str = "convaiinnovations/laya"):
    """Laya, Jev, AgentJev, or a local System One server (Kev, decider,
    OpenThai): one batched call per task, preds mapped back per question.
    The locals serve the same wire format as Jev on their own ports and get
    string instructions — the shape Laya and Kev both expect; AgentJev has
    its own /api/evaluate contract (port 8149) with label descriptions as
    option semantics."""
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

        from jev_client import choice

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
    elif client_kind in SYSTEMONE_LOCAL_PORTS:
        from jev_client import JevClient, choice

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
    elif client_kind == "agentjev":
        from agentjev_client import ask

        def run_task(task: str, texts: list[str]) -> list:
            questions = {
                f"t{i}": {"id": f"t{i}", "type": "choice",
                          "question": INSTR[task].format(text=t),
                          "options": _options(task)}
                for i, t in enumerate(texts)}
            out = ask({"task": task}, questions)
            return [out[f"t{i}"]["value"] for i in range(len(texts))]
    elif client_kind == "jev":
        from jev_client import JevClient

        client = JevClient()

        def run_task(task: str, texts: list[str]) -> list:
            labels = SENTIMENT_LABELS if task == "sentiment" else TOPIC_LABELS
            return client.classify(texts, dict(labels), task=task)
    else:
        raise SystemExit(f"unknown client kind {client_kind!r}")
    return run_task


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


def classify_certo():
    """Certo 421M: calibrated non-generative decision model (vendored
    certo_engine/, card documents no PyPI package). One forward pass scores
    each label description against the state; argmax = decision. Classification
    only - fixed option lists in, one label out, no span extraction."""
    from huggingface_hub import snapshot_download

    from certo_engine import DecisionModel

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

    if name in EXTRACTORS or name in GLICLASS:
        if name in EXTRACTORS:
            cls_one, ner_one = classify_extractor(
                EXTRACTORS[name], name.startswith("GLiFormer"))
        else:
            cls_one = classify_gliclass(GLICLASS[name])
            ner_one = None
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
    elif name == "Certo 421M":
        # in-process decision head over a ModernBERT-large backbone
        # (vendored certo_engine/); classification only - no span
        # extraction, so no NER answers
        cls_one = classify_certo()
        for q in CLS_QUESTIONS:
            t1 = time.perf_counter()
            cls_preds.append(cls_one(q["text"], q["task"]))
            cls_lat.append(time.perf_counter() - t1)
    elif name in ("Laya (local)", "Laya typed-decisions", "Jev",
                  "Kev 0.8B (local)", "AgentJev 0.6B (local)",
                  "decider 0.8B (local)", "OpenThai 0.8B (local)"):
        repo = ("convaiinnovations/laya-typed-decisions"
                if name == "Laya typed-decisions"
                else "convaiinnovations/laya")
        kind = ("laya" if name.startswith("Laya")
                else "kev" if name.startswith("Kev")
                else "agentjev" if name.startswith("AgentJev")
                else "decider" if name.startswith("decider")
                else "openthai" if name.startswith("OpenThai")
                else "jev" if name == "Jev" else None)
        if kind is None:   # unmapped names must never reach the cloud API
            raise SystemExit(f"unwired system {name!r} — add a kind mapping")
        run_task = classify_batched(kind, repo=repo)
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
    elif name == "von":
        from von_client import load_von_decider

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
        # in-process constrained-decision engine (vendored rlcd_engine/),
        # run under the .venv-von python (transformers 5.17 + jsonschema);
        # classification only - the supported schema subset (flat
        # boolean/string-enum fields) cannot express span extraction, so
        # no NER answers
        from rlcd_engine.engine import Engine

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
        # mojev_engine/), run under the .venv-von python (transformers 5.17
        # for the Qwen3.5 encoder); classification only - fixed candidate
        # menus in, one label out, no span extraction, so no NER answers
        from mojev_engine import load_engine

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
    elif name == "nanodiff 350M":
        # diffusion-LM decision model: nanodiff_engine vendors the NanoDiff
        # class (BY571/nanoDiff) and the pngwn typed-decision format; one
        # bidirectional forward, softmax restricted to option-letter token
        # ids. Classification only - a single-letter choice interface, no
        # span extraction, so no NER answers. Runs in the MAIN venv
        # (tiktoken); slow (~10 s/question).
        from nanodiff_engine.runner import QUESTION, load_model, predict

        model, _ = load_model("cpu")
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
    warmed = name.startswith(("Kev", "decider", "OpenThai"))
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
    if ner_ok is not None:
        entry["ner_exact"] = ner_ok
        entry["ner_exact_rate"] = round(sum(ner_ok) / len(ner_ok), 4)
        entry["ner_mean_latency_s"] = round(statistics.mean(ner_lat), 3)
    if name == "von":
        from von_client import provenance

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
