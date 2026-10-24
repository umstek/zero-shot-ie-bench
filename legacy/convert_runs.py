r"""One-off converter: the v0 bench results -> runs/<track>/<yyyy-mm>/.

Reads (all read-only):
  - legacy/results/bench_spectrum_results.json  per-system 48-item mixed-pool
    classification (cls_preds/cls_correct/cls_accuracy/cls_mean_latency_s) and,
    for extractor systems, 18-item NER (ner_exact list of bools,
    ner_exact_rate, ner_mean_latency_s); per-system `recorded` timestamps.
  - legacy/results/bench_multilingual_results.json  by_language holds
    PER-LANGUAGE RATES (floats), not per-item data: per-item truth is
    unavailable, so ml items are written with correct:null and the derived
    count goes into totals.correct. `latency` is s/question; `usage` has
    cost_usd for the hosted systems; meta.notes carry per-system
    "<name> recorded <ts>, <N>s" lines (the LAST line per system is the
    current row - later sessions superseded earlier ones, cf. the von note).
  - suites/decision/mixed-pool, suites/decision/multilingual-pool,
    suites/ner/graded-pool  item ids + gold (files are the v0 pools).
  - catalog/models/**/*.yaml  deployments[].display is the exact v0
    result-name string; it maps every result system name to
    (deployment id, model file, model_version = deployment.version).

bench_results.json (the older flat bench.py run) is NOT converted: every
system it contains is already covered by the spectrum/multilingual files
(verified below before skipping), and extract_suites.py explicitly does not
extract its 10-text NER set.

Writes, per (system x suite):
  runs/<track>/<yyyy-mm>/<run-id>.meta.json   (pretty JSON, LF)
  runs/<run-id>.items.jsonl                   (one object per line, LF, no BOM)
run_id = "v0-<mixed|ml|ner>-<deployment id with ./ replaced by ->".
yyyy-mm comes from the run's recorded timestamp. All timestamps are local
wall-clock (+05:30): the bench ran on this machine (catalog machine `seoul`,
fingerprint hostname SEOUL == this host, TZ +05:30). started_at = the recorded
moment; finished_at stays null (v0 kept a single timestamp per run - the
multilingual notes' durations are not synthesized into a finish time).

totals.cost_usd: spectrum usage.cost_usd (0 when absent); NER and multilingual
usage.cost_usd (0 when absent). Per-item cost_usd is 0 everywhere (v0 recorded
no per-item costs). latency_ms is null on every item (v0 recorded only means,
which land in totals.p50_ms); p95_ms is null everywhere.

Derived multilingual truth (documented decision): by_language gives a rate per
language at exactly 6 items per language (meta.texts_per_language == 6), so
per-language correct = round(rate * 6) and totals.correct = the sum. This
reproduces by_tier exactly (sum of round(tier_rate * 18) over
popular/medium/rare) for every system that has by_tier - asserted below. Two
late systems (mxbai-rerank-large-v2, nemotron-rerank-1b-v2) have neither
by_tier nor a latency entry; their totals.p50_ms is null.

Unresolved system names (skipped, listed in the summary): "Laya Router" - the
multilingual pool's routing mode over laya.local + laya-typed-decisions.local
(per catalog/models/convaiinnovations/laya.yaml notes); it has no single
deployment to attribute a run to.

Multilingual metas additionally carry `v0_rank`: the system's 1-based row
position in the v0-final README's multilingual table (## Multilingual
benchmark). v0 sorted accuracy ties by repo arrival history, which is not
derivable from the metrics, so the row order is recorded as legacy
bookkeeping (the lake threads meta.v0_rank through as the sort tiebreak).
Row names are matched to result system names after stripping the " †"
non-ZDR marker, then via an explicit alias table (the README decorates a few
names with "(local)"/"(mDeBERTa)"/"(mmBERT)"/"(cloud, jev-1.14.0)"/
"(option-marker)" suffixes the results file does not use); the assert below
fails loudly if the table and the results file ever drift apart. Mixed-pool
and NER runs do not carry v0_rank (their tie order is latency-derivable).

suite_hash is recomputed with the extract_suites.py spec (sha256 over
suite.yaml bytes + items/*.jsonl bytes in sorted-filename order) and asserted
against the committed suites' known values.

Run from the repo root:
    .venv/Scripts/python.exe legacy/convert_runs.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TZ = "+05:30"  # bench machine `seoul` == this host (hostname SEOUL), UTC+05:30
BENCH = {"version": "0", "git_sha": "20e4825"}
SOURCE = "legacy-v0"

EXPECTED_HASHES = {
    "decision/mixed-pool":
        "sha256:900aba60ecb6c914ca0a04f86a4ac7e8f5c9e828799196ee02ad3b38c32cc711",
    "decision/multilingual-pool":
        "sha256:ec4631c6905b8114c066431a83d75ab8e5644b94428ac86d383d7a4dd5d112ae",
    "ner/graded-pool":
        "sha256:35af2f255ac13427a43c7d44aca9eed9b5b10f8e3b9ee714238df497ab9f1fd2",
}

# multilingual system name with no single deployment (see module docstring)
UNRESOLVED = {"Laya Router"}

# v0-final README multilingual-table row name -> result system name, for the
# rows the README decorates with a suffix the results file does not carry
# ("D1 3B (local)" needs no alias: it is the literal result name).
README_ROW_ALIASES = {
    "Jev (cloud, jev-1.14.0)": "Jev",
    "GLiNER2.5-multi (mDeBERTa)": "GLiNER2.5-multi",
    "Laya Router (mmBERT)": "Laya Router",
    "von-1.0 (option-marker)": "von",
    "Decision 2.0 Kai 0.6B (local)": "Decision 2.0 Kai 0.6B",
    "Decision 2.0 Eos 0.8B (local)": "Decision 2.0 Eos 0.8B",
    "Decision 2.0 Sol 2B (local)": "Decision 2.0 Sol 2B",
    "Decision 3.0 Edge 0.6B (local)": "Decision 3.0 Edge 0.6B",
    "Decision 3.0 Lite 0.85B (local)": "Decision 3.0 Lite 0.85B",
    "Decision 3.0 Nano 2B (local)": "Decision 3.0 Nano 2B",
}


def v0_ml_ranks() -> dict[str, int]:
    """Result system name -> 1-based row position in the v0-final README
    multilingual table (## Multilingual benchmark), the v0 tiebreak order."""
    readme = subprocess.run(
        ["git", "show", "v0-final:README.md"], check=True, cwd=ROOT,
        capture_output=True, text=True, encoding="utf-8").stdout
    lines = readme.splitlines()
    start = next(i for i, line in enumerate(lines)
                 if line.startswith("## Multilingual benchmark"))
    rows: list[str] = []
    for line in lines[start:]:
        if not line.startswith("|"):
            if rows:
                break  # table ended
            continue
        cells = [c.strip() for c in line.split("|")]
        if cells[1] == "System" or set(cells[1]) <= {"-"}:
            continue  # header / separator
        name = re.sub(r"\s+[†‡]\s*$", "", cells[1])  # strip the non-ZDR marker
        rows.append(name)
    ranks: dict[str, int] = {}
    for position, row in enumerate(rows, 1):
        # alias table first (exact README names it decorates), else the row
        # name is expected to be the literal result name
        system = README_ROW_ALIASES.get(row, row)
        assert system not in ranks, f"{system!r} appears twice in the table"
        ranks[system] = position
    return ranks


def suite_hash(base: str) -> str:
    """sha256 over suite.yaml bytes + each items/*.jsonl sorted by filename
    (same spec as legacy/extract_suites.py:suite_hash)."""
    h = hashlib.sha256()
    with open(os.path.join(base, "suite.yaml"), "rb") as fh:
        h.update(fh.read())
    items_dir = os.path.join(base, "items")
    for name in sorted(os.listdir(items_dir)):
        with open(os.path.join(items_dir, name), "rb") as fh:
            h.update(fh.read())
    return "sha256:" + h.hexdigest()


def load_json(rel: str):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return json.load(fh)


def load_suite(track: str, suite_id: str) -> dict:
    """{suite_id, files: {lang: [rows]}} with the recomputed hash asserted."""
    base = os.path.join(ROOT, "suites", track, suite_id)
    meta = yaml.safe_load(open(os.path.join(base, "suite.yaml"), encoding="utf-8"))
    digest = suite_hash(base)
    assert digest == EXPECTED_HASHES[f"{track}/{suite_id}"], digest
    files = {}
    items_dir = os.path.join(base, "items")
    for name in sorted(os.listdir(items_dir)):
        lang = name.removesuffix(".jsonl")
        with open(os.path.join(items_dir, name), encoding="utf-8") as fh:
            files[lang] = [json.loads(line) for line in fh if line.strip()]
    return {"id": meta["id"], "version": meta["version"], "files": files,
            "hash": digest}


def iso(ts: str) -> str:
    """'YYYY-MM-DD HH:MM' (or with :SS) -> ISO 8601 with the local offset."""
    parts = ts.split(" ")
    date, clock = parts[0], parts[1]
    hh, mm = clock.split(":")[:2]
    return f"{date}T{hh}:{mm}:00{TZ}"


def slug(deployment_id: str) -> str:
    return deployment_id.replace("/", "-").replace(".", "-")


def display_map() -> dict[str, dict]:
    """v0 result-name (deployments[].display) -> {model_id, deployment}."""
    out = {}
    models_dir = os.path.join(ROOT, "catalog", "models")
    for root, _dirs, files in os.walk(models_dir):
        for fname in sorted(files):
            if not fname.endswith(".yaml"):
                continue
            path = os.path.join(root, fname)
            model = yaml.safe_load(open(path, encoding="utf-8"))
            for dep in model.get("deployments") or []:
                display = dep.get("display")
                if display is None:
                    continue
                assert display not in out, f"duplicate display {display!r}"
                assert "kind" in dep and "version" in dep, (display, dep)
                if dep["kind"] == "hosted":
                    assert "provider" in dep, (display, dep)
                out[display] = {"model_id": model["id"], "deployment": dep}
    return out


def where_of(dep: dict) -> dict:
    if dep["kind"] == "hosted":
        return {"kind": "hosted", "provider": dep["provider"]}
    return {"kind": "local", "machine": "seoul",
            "runtime": dep.get("runtime") or None}


def write_run(track: str, run_id: str, started_at: str, meta: dict,
              items: list[dict]) -> str:
    """Write <run-id>.meta.json + <run-id>.items.jsonl under
    runs/<track>/<yyyy-mm>/ (yyyy-mm from started_at); return the directory."""
    month_dir = os.path.join(ROOT, "runs", track, started_at[:7])
    os.makedirs(month_dir, exist_ok=True)
    meta = dict(meta, run_id=run_id)
    meta_bytes = (json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    items_bytes = b"".join(
        (json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8")
        for item in items)
    with open(os.path.join(month_dir, f"{run_id}.meta.json"), "wb") as fh:
        fh.write(meta_bytes)
    with open(os.path.join(month_dir, f"{run_id}.items.jsonl"), "wb") as fh:
        fh.write(items_bytes)
    return month_dir


def base_meta(dep: dict, suite: dict, started_at: str) -> dict:
    return {
        "status": "ok",
        "deferred_reason": None,
        "deployment": dep["id"],
        "model_version": dep["version"],
        "served_version": None,
        "where": where_of(dep),
        "suite": suite["id"],
        "suite_version": suite["version"],
        "suite_hash": suite["hash"],
        "params": {},
        "params_hash": None,
        "bench": dict(BENCH),
        "started_at": started_at,
        "finished_at": None,
        "totals": None,  # filled per suite kind
        "source": SOURCE,
    }


def item_row(row: dict, *, prediction, correct) -> dict:
    return {
        "item_id": row["item_id"],
        "lang": row["lang"],
        "prediction": prediction,
        "gold": row["gold"],
        "correct": correct,
        "score": None,
        "prob": None,
        "latency_ms": None,
        "compute_ms": None,
        "cost_usd": 0,
        "usage": None,
        "error": None,
    }


def main() -> None:
    spectrum = load_json("legacy/results/bench_spectrum_results.json")
    spectrum_systems = spectrum["systems"]
    multilingual = load_json("legacy/results/bench_multilingual_results.json")
    old_flat = load_json("legacy/results/bench_results.json")

    mixed_suite = load_suite("decision", "mixed-pool")
    ml_suite = load_suite("decision", "multilingual-pool")
    ner_suite = load_suite("ner", "graded-pool")
    mixed_rows = mixed_suite["files"]["en"]
    ner_rows = ner_suite["files"]["en"]
    assert len(mixed_rows) == 48 and len(ner_rows) == 18
    assert sum(len(rows) for rows in ml_suite["files"].values()) == 54

    # bench_results.json is fully superseded: every system it scores appears in
    # the spectrum file (its 10-text NER pool was never part of the suites).
    old_systems = (set(old_flat["classification"]["sentiment"])
                   | set(old_flat["classification"]["topic"])
                   | set(old_flat["ner"]))
    assert old_systems <= set(spectrum_systems), sorted(old_systems - set(spectrum_systems))
    print(f"bench_results.json: {len(old_systems)} systems, all covered by "
          f"bench_spectrum_results.json -> superseded, skipped")

    catalog_langs = yaml.safe_load(
        open(os.path.join(ROOT, "catalog", "languages.yaml"), encoding="utf-8"))
    lang_code = {e["name"]: e["code"] for e in catalog_langs}

    disp = display_map()
    unresolved = sorted({n for n in list(spectrum_systems) + list(multilingual["by_language"])
                         if n not in disp})
    assert set(unresolved) == UNRESOLVED, unresolved

    # per-system recorded notes from the multilingual meta (last one wins:
    # later sessions superseded earlier rows, cf. the von supersession note)
    ml_recorded: dict[str, str] = {}
    for note in multilingual["meta"]["notes"]:
        m = re.match(r"^(.*?) recorded (\d{4}-\d{2}-\d{2} \d{2}:\d{2})(?::\d{2})?,", note)
        if m:
            ml_recorded[m.group(1)] = m.group(2)

    assert multilingual["meta"]["texts_per_language"] == 6
    texts_per_lang = multilingual["meta"]["texts_per_language"]

    # v0-final README row order -> legacy bookkeeping ranks (see docstring);
    # must cover every multilingual system exactly once, Laya Router included
    v0_ranks = v0_ml_ranks()
    assert set(v0_ranks) == set(multilingual["by_language"]), sorted(
        set(v0_ranks) ^ set(multilingual["by_language"]))
    assert sorted(v0_ranks.values()) == list(range(1, len(v0_ranks) + 1))

    counts: dict[str, int] = {}
    written: list[str] = []

    def emit(system: str, suite: dict, short: str, started_at: str,
             totals: dict, items: list[dict],
             extra: dict | None = None) -> None:
        entry = disp[system]
        dep = entry["deployment"]
        run_id = f"v0-{short}-{slug(dep['id'])}"
        meta = base_meta(dep, suite, started_at)
        if extra:
            meta.update(extra)
        meta["totals"] = totals
        month_dir = write_run(suite["id"].split("/")[0], run_id, started_at,
                              meta, items)
        counts[suite["id"]] = counts.get(suite["id"], 0) + 1
        written.append(os.path.relpath(
            os.path.join(month_dir, run_id), ROOT).replace("\\", "/"))

    # ------------------------------------------------------------------
    # spectrum: mixed-pool (48 items) + graded-pool NER (18, extractors)
    # ------------------------------------------------------------------
    for system in sorted(spectrum_systems):
        e = spectrum_systems[system]
        preds, correct = e["cls_preds"], e["cls_correct"]
        assert len(preds) == len(correct) == 48, system
        assert sum(correct) == round(e["cls_accuracy"] * 48), system
        for i, (pred, ok) in enumerate(zip(preds, correct)):
            if ok:
                assert pred == mixed_rows[i]["gold"], (system, i)
            else:
                assert pred != mixed_rows[i]["gold"] or pred is None, (system, i)
        started_at = iso(e["recorded"])
        usage_cost = (e.get("usage") or {}).get("cost_usd", 0)
        totals = {
            "items": 48,
            "correct": sum(correct),
            "cost_usd": usage_cost,
            "p50_ms": int(round(e["cls_mean_latency_s"] * 1000)),
            "p95_ms": None,
            "errors": 0,
        }
        items = [item_row(row, prediction=pred, correct=ok)
                 for row, pred, ok in zip(mixed_rows, preds, correct)]
        emit(system, mixed_suite, "mixed", started_at, totals, items)

        if "ner_exact" not in e:
            continue
        ner_exact = e["ner_exact"]
        assert isinstance(ner_exact, list) and len(ner_exact) == 18, system
        assert round(sum(ner_exact) / 18, 4) == e["ner_exact_rate"], system
        ner_totals = {
            "items": 18,
            "correct": sum(ner_exact),
            "cost_usd": 0,  # v0 recorded no separate NER cost
            "p50_ms": int(round(e["ner_mean_latency_s"] * 1000)),
            "p95_ms": None,
            "errors": 0,
        }
        ner_items = [item_row(row, prediction=None, correct=None)
                     for row in ner_rows]
        emit(system, ner_suite, "ner", started_at, ner_totals, ner_items)

    # ------------------------------------------------------------------
    # multilingual: 54 items (9 languages x 6), per-item truth unavailable
    # ------------------------------------------------------------------
    skipped = []
    for system in sorted(multilingual["by_language"]):
        if system in UNRESOLVED:
            skipped.append(system)
            continue
        per_lang = multilingual["by_language"][system]
        assert set(per_lang) == set(lang_code) - {"English"}, system
        derived = sum(round(rate * texts_per_lang) for rate in per_lang.values())
        if system in multilingual["by_tier"]:
            tier_total = sum(round(v * 3 * texts_per_lang)
                             for v in multilingual["by_tier"][system].values())
            assert tier_total == derived, (system, derived, tier_total)
        assert system in ml_recorded, system
        started_at = iso(ml_recorded[system])
        latency = multilingual["latency"].get(system)
        usage_cost = (multilingual["usage"].get(system) or {}).get("cost_usd", 0)
        totals = {
            "items": 54,
            "correct": derived,
            "cost_usd": usage_cost,
            "p50_ms": int(round(latency * 1000)) if latency is not None else None,
            "p95_ms": None,
            "errors": 0,
        }
        items = []
        for lang in sorted(ml_suite["files"]):
            for row in ml_suite["files"][lang]:
                assert row["lang"] == lang
                items.append(item_row(row, prediction=None, correct=None))
        assert len(items) == 54
        emit(system, ml_suite, "ml", started_at, totals, items,
             extra={"v0_rank": v0_ranks[system]})

    # ------------------------------------------------------------------
    # round-trip check: every written file parses back
    # ------------------------------------------------------------------
    for rel in written:
        with open(os.path.join(ROOT, rel + ".meta.json"), encoding="utf-8") as fh:
            json.load(fh)
        with open(os.path.join(ROOT, rel + ".items.jsonl"), encoding="utf-8") as fh:
            lines = [line for line in fh.read().split("\n") if line]
            for line in lines:
                json.loads(line)

    print(f"\nwrote {len(written)} runs:")
    for suite_id in sorted(counts):
        print(f"  {suite_id}: {counts[suite_id]} runs")
    if skipped:
        print(f"skipped (no single deployment): {skipped}")


if __name__ == "__main__":
    main()
