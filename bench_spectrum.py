"""Spectrum benchmark: one mixed pool of questions, difficulty measured.

No easy/medium/hard buckets. All classification questions (sentiment +
topic from bench_graded's pool) form ONE pool of 48; NER questions form one
pool of 18. Every system answers every question; a question's difficulty is
computed afterwards as the fraction of answering systems that got it wrong
(continuous 0.0-1.0 spectrum).

Per-question predictions are stored (not just aggregates) so the app can
plot accuracy along the difficulty spectrum.

Run from the MAIN venv for most systems, from .venv-von for von:
    python bench_spectrum.py --system GLiNER2.5-base
    ...
    .venv-von/Scripts/python bench_spectrum.py --system von

Output: bench_spectrum_results.json
"""

from __future__ import annotations

import argparse
import json
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
    "GLiFormer-base": "knowledgator/gliformer-base-v1",
    "GLiFormer-large": "knowledgator/gliformer-large-v1",
}
GLICLASS = {
    "gliclass-edge": "knowledgator/gliclass-edge-v3.0",
    "gliclass-modern-base": "knowledgator/gliclass-modern-base-v3.0",
    "gliclass-base": "knowledgator/gliclass-base-v3.0",
    "gliclass-large": "knowledgator/gliclass-large-v3.0",
}
ALL_SYSTEMS = (list(EXTRACTORS) + list(GLICLASS)
               + ["Laya (local)", "Jev", "von", "so1 (Qwen2.5-0.5B)"])


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


def classify_batched(client_kind: str):
    """Laya or Jev: one batched call per task, preds mapped back per question."""
    if client_kind == "laya":
        import laya

        from jev_client import choice

        agent = laya.load("convaiinnovations/laya")
        INSTR = {
            "sentiment": 'What is the overall sentiment of this text: "{text}"',
            "topic": 'Which topic category does this text belong to: "{text}"',
        }

        def run_task(task: str, texts: list[str]) -> list:
            labels = SENTIMENT_LABELS if task == "sentiment" else TOPIC_LABELS
            questions = {
                f"t{i}": choice(INSTR[task].format(text=t),
                                {l: None for l in labels})
                for i, t in enumerate(texts)}
            out = agent.predict({"task": task}, questions)
            return [out["answers"][f"t{i}"].get("choice")
                    for i in range(len(texts))]
    else:
        from jev_client import JevClient

        client = JevClient()

        def run_task(task: str, texts: list[str]) -> list:
            labels = SENTIMENT_LABELS if task == "sentiment" else TOPIC_LABELS
            return client.classify(texts, dict(labels), task=task)
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
    elif name in ("Laya (local)", "Jev"):
        run_task = classify_batched("laya" if name.startswith("Laya") else "jev")
        for task in ("sentiment", "topic"):
            texts = [q["text"] for q in CLS_QUESTIONS if q["task"] == task]
            t1 = time.perf_counter()
            preds = run_task(task, texts)
            dt = time.perf_counter() - t1
            cls_preds.extend(preds)
            cls_lat.extend([dt / len(texts)] * len(texts))
    elif name == "von":
        import von

        for q in CLS_QUESTIONS:
            labels = (SENTIMENT_LABELS if q["task"] == "sentiment"
                      else TOPIC_LABELS)
            t1 = time.perf_counter()
            res = von.decide(state=q["text"], choices=dict(labels),
                             instructions=f"What is the overall {q['task']} "
                                           "of this text?")
            cls_lat.append(time.perf_counter() - t1)
            cls_preds.append(res.choice)
    else:  # so1
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

    correct = [p == q["gold"] for p, q in zip(cls_preds, CLS_QUESTIONS)]
    entry = {
        "recorded": time.strftime("%Y-%m-%d %H:%M"),
        "cls_preds": cls_preds,
        "cls_correct": correct,
        "cls_accuracy": round(sum(correct) / len(correct), 4),
        "cls_mean_latency_s": round(statistics.mean(cls_lat), 3),
    }
    if ner_ok is not None:
        entry["ner_exact"] = ner_ok
        entry["ner_exact_rate"] = round(sum(ner_ok) / len(ner_ok), 4)
        entry["ner_mean_latency_s"] = round(statistics.mean(ner_lat), 3)

    try:
        with open(RESULTS_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
    except OSError:
        data = {}
    data.setdefault("meta", {
        "pools": {"classification": 48, "ner": 18},
        "difficulty": "per question: fraction of answering systems that "
                      "answered it wrong (continuous 0-1, computed by the "
                      "app from stored per-question results)",
        "notes": ["One mixed pool per ability - no difficulty buckets.",
                  "von runs in .venv-von; so1 uses Qwen2.5-0.5B."]})
    data["systems"] = {k: v for k, v in data.get("systems", {}).items()
                       if k != name}
    data["systems"][name] = entry
    with open(RESULTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    ner_note = (f", NER exact {entry['ner_exact_rate'] * 100:.0f}%"
                if ner_ok is not None else "")
    print(f"  classification {entry['cls_accuracy'] * 100:.1f}%{ner_note} "
          f"in {time.perf_counter() - t0:.0f}s -> {RESULTS_FILE}")


if __name__ == "__main__":
    main()
