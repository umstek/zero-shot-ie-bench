"""Newcomers benchmark: GLiClass v3.0 family, von-1.0, open-alternative-jev.

Same graded data as bench_graded.py (sentiment + topic, easy/medium/hard,
8 texts per tier per task). Classification only - none of these systems
produce spans.

Run from the MAIN venv (transformers 4.57):
    python bench_newcomers.py --system gliclass-edge
    python bench_newcomers.py --system gliclass-modern-base
    python bench_newcomers.py --system gliclass-base
    python bench_newcomers.py --system gliclass-large
    python bench_newcomers.py --system so1            # Qwen2.5-0.5B, CPU

Run from .venv-von (von-sdk needs transformers 5):
    .venv-von/Scripts/python bench_newcomers.py --system von

Results merge into bench_newcomers_results.json (consumed by the app's
"Benchmarks v2" tab alongside the incumbents).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

from bench import SENTIMENT_LABELS, TOPIC_LABELS
from bench_graded import SENTIMENT, TOPIC, TIERS

RESULTS_FILE = "bench_newcomers_results.json"

GLICASS_MODELS = {
    "gliclass-edge": "knowledgator/gliclass-edge-v3.0",
    "gliclass-modern-base": "knowledgator/gliclass-modern-base-v3.0",
    "gliclass-base": "knowledgator/gliclass-base-v3.0",
    "gliclass-large": "knowledgator/gliclass-large-v3.0",
}

TASK_INSTRUCTIONS = {
    "sentiment": "What is the overall sentiment of this text?",
    "topic": "Which topic category does this text belong to?",
}


def classify_gliclass(model_key: str):
    from gliclass import GLiClassModel, ZeroShotClassificationPipeline
    from transformers import AutoTokenizer

    model_id = GLICASS_MODELS[model_key]
    model = GLiClassModel.from_pretrained(model_id)
    pipe = ZeroShotClassificationPipeline(
        model, AutoTokenizer.from_pretrained(model_id),
        classification_type="multi-label", device="cpu")

    def run(tier: str, task: str):
        data = SENTIMENT[tier] if task == "sentiment" else TOPIC[tier]
        texts = [t for t, _ in data]
        gold = [g for _, g in data]
        labels = list(SENTIMENT_LABELS if task == "sentiment"
                      else TOPIC_LABELS)
        correct, lat = 0, []
        for text, truth in zip(texts, gold):
            t0 = time.perf_counter()
            out = pipe(text, labels, threshold=0.0)[0]
            lat.append(time.perf_counter() - t0)
            pred = max(out, key=lambda x: x["score"])["label"] if out else None
            correct += pred == truth
        return correct, len(texts), statistics.mean(lat)

    return run


def classify_von():
    import von

    def run(tier: str, task: str):
        data = SENTIMENT[tier] if task == "sentiment" else TOPIC[tier]
        labels = SENTIMENT_LABELS if task == "sentiment" else TOPIC_LABELS
        correct, lat = 0, []
        for text, truth in data:
            t0 = time.perf_counter()
            res = von.decide(state=text, choices=dict(labels),
                             instructions=TASK_INSTRUCTIONS[task])
            lat.append(time.perf_counter() - t0)
            correct += res.choice == truth
        return correct, len(data), statistics.mean(lat)

    return run


def classify_so1():
    from so1 import Choice, Decider

    decider = Decider.from_pretrained("Qwen/Qwen2.5-0.5B", backend="hf")

    def run(tier: str, task: str):
        data = SENTIMENT[tier] if task == "sentiment" else TOPIC[tier]
        labels = list(SENTIMENT_LABELS if task == "sentiment"
                      else TOPIC_LABELS)
        correct, lat = 0, []
        for text, truth in data:
            t0 = time.perf_counter()
            out = decider.decide(
                state=text,
                questions=[Choice(task, labels)],
                mode="separate")
            lat.append(time.perf_counter() - t0)
            correct += out[0].choice == truth
        return correct, len(data), statistics.mean(lat)

    return run


SYSTEMS = {
    **{k: classify_gliclass for k in GLICASS_MODELS},
    "von": classify_von,
    "so1 (Qwen2.5-0.5B)": classify_so1,
}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True, choices=SYSTEMS)
    args = parser.parse_args()

    factory = SYSTEMS[args.system]
    run = (factory(args.system) if args.system in GLICASS_MODELS
           else factory())

    try:
        with open(RESULTS_FILE, encoding="utf-8") as fh:
            results = json.load(fh).get("systems", {})
    except OSError:
        results = {}

    print(f"{args.system}: sentiment + topic x 3 tiers ...")
    t0 = time.perf_counter()
    entry = {"recorded": time.strftime("%Y-%m-%d %H:%M"),
             "classification": {}}
    for task in ("sentiment", "topic"):
        entry["classification"][task] = {}
        for tier in TIERS:
            correct, n, lat = run(tier, task)
            entry["classification"][task][tier] = {
                "accuracy": round(correct / n, 4),
                "mean_latency_s": round(lat, 3)}
            print(f"  {task:<10} {tier:<7} {correct}/{n} "
                  f"({correct / n * 100:.1f}%)  {lat:.3f}s/text")

    results[args.system] = entry
    with open(RESULTS_FILE, "w", encoding="utf-8") as fh:
        json.dump({"meta": {"date": time.strftime("%Y-%m-%d %H:%M"),
                            "device": "CPU",
                            "notes": [
                                "Same graded data as bench_graded.py; "
                                "classification only (no span output).",
                                "von runs in .venv-von (needs transformers "
                                "5.x); everything else in the main venv.",
                            ]},
                   "systems": results}, fh, indent=2, ensure_ascii=False)
    print(f"done in {time.perf_counter() - t0:.0f}s -> {RESULTS_FILE}")


if __name__ == "__main__":
    main()
