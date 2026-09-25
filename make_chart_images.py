"""Render the benchmark tab charts to PNGs for the README.

Imports the chart helpers and data preparation from app.py, so the
images are exactly what the web UI renders. Re-run after re-running
the benchmarks:

    python make_chart_images.py          # writes docs/charts/*.png
"""

from __future__ import annotations

import json
import os

import pandas as pd

import app
from app import (accuracy_heatmap, hbar_chart, hbar_chart_labeled,
                 spectrum_line, tradeoff_scatter)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "docs", "charts")


def load(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: not a JSON object")
    return data


def spectrum_charts(bench):
    """Classification + NER charts, mirroring app.py's benchmark tabs."""
    systems = bench["systems"]
    n_q = len(next(iter(systems.values()))["cls_correct"])
    difficulty = [1 - sum(s["cls_correct"][i] for s in systems.values())
                  / len(systems) for i in range(n_q)]

    summary = pd.DataFrame([
        {"System": s, "Accuracy %": round(v["cls_accuracy"] * 100, 1),
         "s per question": v["cls_mean_latency_s"]}
        for s, v in systems.items()])
    charts = {
        "cls_accuracy": hbar_chart_labeled(
            summary, "Accuracy %", "Classification accuracy, mixed pool",
            "Accuracy %"),
        "cls_latency": hbar_chart_labeled(
            summary, "s per question",
            "Mean latency per question (CPU; Jev/Laya batch ÷ n)",
            "seconds (log)", fmt=".3f", log=True, descending=False),
        "cls_tradeoff": tradeoff_scatter(
            summary, "Speed vs accuracy — up and left is better"),
    }

    thresholds = sorted({round(t / 20, 2) for t in range(21)})
    spec_rows = []
    for sys_, s in systems.items():
        for th in thresholds:
            idx = [i for i in range(n_q) if difficulty[i] <= th]
            if len(idx) < 4:
                continue
            acc = sum(s["cls_correct"][i] for i in idx) / len(idx)
            spec_rows.append({"Question difficulty ≤": th, "System": sys_,
                              "Accuracy %": round(acc * 100, 1)})
    if spec_rows:
        charts["cls_spectrum"] = spectrum_line(
            pd.DataFrame(spec_rows), "Accuracy %",
            "Accuracy vs question difficulty — each point is accuracy "
            "on questions at most this hard")

    extractors = {s: v for s, v in systems.items() if "ner_exact" in v}
    if extractors:
        n_ner = len(next(iter(extractors.values()))["ner_exact"])
        ner_difficulty = [
            1 - sum(s["ner_exact"][i] for s in extractors.values())
            / len(extractors) for i in range(n_ner)]
        ner_summary = pd.DataFrame([
            {"System": s,
             "Exact match %": round(v["ner_exact_rate"] * 100, 1),
             "s per text": v["ner_mean_latency_s"]}
            for s, v in extractors.items()])
        charts["ner_exact"] = hbar_chart_labeled(
            ner_summary, "Exact match %", "NER exact-match rate, mixed pool",
            "Exact match %")
        ner_rows = []
        for sys_, s in extractors.items():
            for th in thresholds:
                idx = [i for i in range(n_ner) if ner_difficulty[i] <= th]
                if len(idx) < 3:
                    continue
                rate = sum(s["ner_exact"][i] for i in idx) / len(idx)
                ner_rows.append({"Question difficulty ≤": th,
                                 "System": sys_,
                                 "Exact match %": round(rate * 100, 1)})
        if ner_rows:
            charts["ner_spectrum"] = spectrum_line(
                pd.DataFrame(ner_rows), "Exact match %",
                "Exact match vs question difficulty")
    return charts


def multilingual_charts(ml):
    by_language = ml["by_language"]
    langs = list(next(iter(by_language.values())).keys())
    systems = list(by_language.keys())
    heat = pd.DataFrame([
        {"System": s, "Language": lang,
         "Accuracy %": round(by_language[s][lang] * 100, 1)}
        for s in systems for lang in langs])
    overall = pd.DataFrame([
        {"System": s,
         "Accuracy %": round(sum(by_language[s][l] for l in langs)
                             / len(langs) * 100, 1)}
        for s in systems])
    return {
        "ml_heatmap": accuracy_heatmap(heat,
                                       "Accuracy by system and language (%)"),
        "ml_overall": hbar_chart_labeled(
            overall, "Accuracy %", "Overall multilingual accuracy (54 texts)",
            "Accuracy %"),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    charts = {}
    charts.update(spectrum_charts(
        load(os.path.join(os.path.dirname(app.BENCH_FILE),
                          "bench_spectrum_results.json"))))
    charts.update(multilingual_charts(
        load(os.path.join(os.path.dirname(app.BENCH_FILE),
                          "bench_multilingual_results.json"))))
    for name, chart in charts.items():
        path = os.path.join(OUT, f"{name}.png")
        chart.save(path, scale=2)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
