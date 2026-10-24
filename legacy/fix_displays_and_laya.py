"""One-off post-pass after convert_runs.py, in order:
(1) rewrite deployment display names to the exact v0 README *table row* names
    (the tables decorate a few rows — "von-1.0 (option-marker)", "(local)"
    suffixes — while the results files used bare names);
(2) flag the per-deployment † (non-ZDR) set from the v0 tables — the marker
    is per endpoint, not per provider (Perplexity deciders: no †);
(3) write the one multilingual run convert_runs.py skipped (Laya Router),
    now that its deployment exists;
(4) add v0_per_language rates to every multilingual run meta — v0 stored
    per-language RATES only, so items carry correct:null and the tier
    rollups need the rates from the meta.
Idempotent. Run: .venv/Scripts/python.exe legacy/fix_displays_and_laya.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "legacy"))
from convert_runs import README_ROW_ALIASES, v0_ml_ranks  # noqa: E402

readme = subprocess.run(
    ["git", "show", "v0-final:README.md"], check=True, cwd=ROOT,
    capture_output=True, text=True, encoding="utf-8").stdout

ml = json.loads((ROOT / "legacy/results/bench_multilingual_results.json").read_text(encoding="utf-8"))

L2C = {"Spanish": "es", "French": "fr", "Chinese": "zh", "Vietnamese": "vi",
       "Turkish": "tr", "Ukrainian": "uk", "Sinhala": "si", "Icelandic": "is",
       "Welsh": "cy"}



def table_name(display):
    """display -> the bare result-name form used by the mixed table rows"""
    if display is None:
        return None
    return README_ROW_ALIASES.get(display, display)


def save_yaml(path: Path, doc: dict) -> None:
    path.write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8")


# (1) display names -> decorated README row names
decorated = {v: k for k, v in README_ROW_ALIASES.items()}
for row in re.findall(r"^\| (.+?) \| .*%.* \|\s*$", readme, re.M):
    name = row.strip().rstrip("†").strip()
    decorated.setdefault(name, name)

# deployment id -> display (for the per-language pass below)
DEP_DISPLAY: dict[str, str] = {}

changed = 0
for path in sorted((ROOT / "catalog" / "models").rglob("*.yaml")):
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    touched = False
    for dep in doc.get("deployments") or []:
        disp = dep.get("display")
        if disp:
            DEP_DISPLAY[dep["id"]] = disp
        if disp and disp in decorated and decorated[disp] != disp:
            dep["display"] = decorated[disp]
            DEP_DISPLAY[dep["id"]] = decorated[disp]
            touched = True
    if touched:
        save_yaml(path, doc)
        changed += 1
print(f"displays rewritten in {changed} model files")

# (2) per-deployment † flags
dagger_names = set()
plain_names = set()
for line in readme.splitlines():
    stripped = line.strip()
    if not (stripped.startswith("| ") and stripped.endswith(" |")) or "%" not in stripped:
        continue
    name = stripped[2:].split(" |")[0].strip()
    if name.endswith("†"):
        dagger_names.add(name.rstrip("†").strip())
    else:
        plain_names.add(name)
plain_names -= dagger_names
flagged = 0
for path in sorted((ROOT / "catalog" / "models").rglob("*.yaml")):
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    touched = False
    for dep in doc.get("deployments") or []:
        disp = dep.get("display")
        if disp in dagger_names and dep.get("zdr_safe") is not False:
            dep["zdr_safe"] = False
            touched = True
        elif disp in plain_names and dep.get("zdr_safe") is None:
            dep["zdr_safe"] = True
            touched = True
    if touched:
        save_yaml(path, doc)
        flagged += 1
print(f"zdr flags set on deployments in {flagged} files "
      f"({len(dagger_names)} dagger rows, {len(plain_names)} plain rows in v0)")

# (3) the skipped Laya Router multilingual run
SUITE_HASH = "sha256:ec4631c6905b8114c066431a83d75ab8e5644b94428ac86d383d7a4dd5d112ae"
laya_meta = ROOT / "runs/decision/2026-09/v0-ml-laya-router.meta.json"
if not laya_meta.exists():
    langs = ml["by_language"]["Laya Router"]
    items = [
        {"item_id": f"ml-{i:03d}", "lang": L2C[lang], "prediction": None, "gold": None,
         "correct": None, "score": None, "prob": None, "latency_ms": None,
         "compute_ms": None, "cost_usd": 0, "usage": None, "error": None}
        for lang, _rate in langs.items() for i in range(1, 7)
    ]
    correct = sum(round(r * 6) for r in langs.values())
    rank = v0_ml_ranks()["Laya Router"]
    meta = {
        "run_id": "v0-ml-laya-router",
        "status": "ok", "deferred_reason": None,
        "deployment": "laya.router", "model_version": "laya@unversioned",
        "served_version": None,
        "where": {"kind": "local", "machine": "seoul",
                  "runtime": {"engine": "torch", "dtype": None, "quant": None, "device": "cpu"}},
        "suite": "decision/multilingual-pool", "suite_version": 1,
        "suite_hash": SUITE_HASH,
        "params": {}, "params_hash": None, "bench": {"version": "0", "git_sha": "20e4825"},
        "started_at": "2026-09-20T11:24:00+05:30", "finished_at": None,
        "totals": {"items": 54, "correct": correct, "cost_usd": 0, "p50_ms": None,
                   "p95_ms": None, "errors": 0},
        "source": "legacy-v0", "v0_rank": rank,
    }
    laya_meta.parent.mkdir(parents=True, exist_ok=True)
    laya_meta.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")
    (laya_meta.parent / "v0-ml-laya-router.items.jsonl").write_text(
        "\n".join(json.dumps(i) for i in items) + "\n", encoding="utf-8", newline="\n")
    print(f"laya.router multilingual run written: {correct}/54, rank {rank}")

# (4) v0_per_language rates on every multilingual meta
rates_added = 0
for mf in sorted((ROOT / "runs/decision").rglob("v0-ml-*.meta.json")):
    meta = json.loads(mf.read_text(encoding="utf-8"))
    display = DEP_DISPLAY.get(meta["deployment"])
    # displays are the decorated table names; the results file keys are the
    # bare result names — invert the alias table
    result_name = README_ROW_ALIASES.get(display, display) if display else None
    if result_name is None or result_name not in ml["by_language"]:
        print(f"  no per-language rates for {meta['deployment']} ({display!r})")
        continue
    rates = {L2C[lang]: rate for lang, rate in ml["by_language"][result_name].items()
             if lang in L2C}
    if rates:
        meta["v0_per_language"] = rates
        mf.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")
        rates_added += 1
print(f"v0_per_language rates added to {rates_added} multilingual metas")

# (5) v0_rank on the mixed-pool runs (same mechanism as the multilingual
# pass): v0's within-accuracy row order follows the repo's arrival history,
# not any metric — record the row position as legacy bookkeeping
mixed_section = readme.split("## Mixed-pool spectrum benchmark")[1].split("\n## ", 1)[0]
mixed_lines = [ln for ln in mixed_section.splitlines() if ln.startswith("| ")]
# keep the main table: the longest run of lines sharing the
# "| System | Classification |" header shape
header_idx = [i for i, ln in enumerate(mixed_lines) if ln.startswith("| System | Classification |")]
best_start, best_len = 0, 0
for i in header_idx:
    j = i + 1  # separator lines don't match "| " and are already excluded
    while j < len(mixed_lines) and "%" in mixed_lines[j]:
        j += 1
    if j - i > best_len:
        best_start, best_len = i, j - i
mixed_rank: dict[str, int] = {}
for pos, ln in enumerate(mixed_lines[best_start + 1 : best_start + best_len], 1):
    name = ln.split(" |")[0].strip().lstrip("| ").rstrip("†").strip()
    mixed_rank[name] = pos
ranked = 0
for mf in sorted((ROOT / "runs/decision").rglob("v0-mixed-*.meta.json")):
    meta = json.loads(mf.read_text(encoding="utf-8"))
    display = table_name(DEP_DISPLAY.get(meta["deployment"]))
    if display in mixed_rank:
        meta["v0_rank"] = mixed_rank[display]
        mf.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")
        ranked += 1
print(f"v0_rank added to {ranked} mixed metas ({len(mixed_rank)} table rows)")

# v0's s/question cell formatting is irregular (hand-edited: "0.080" keeps a
# trailing zero, "8.0" drops two) — import the latency cells verbatim too
mixed_cells: dict[str, dict[str, str]] = {}
for ln in mixed_lines[best_start + 1 : best_start + best_len]:
    parts = [p.strip() for p in ln.strip().strip("|").split("|")]
    name = parts[0].rstrip("†").strip()
    mixed_cells[name] = {"name": parts[0],
                         "cls_s": parts[2] if len(parts) > 2 else "",
                         "ner_s": parts[4] if len(parts) > 4 else ""}
cells_stored = 0
for mf in sorted((ROOT / "runs/decision").rglob("v0-mixed-*.meta.json")):
    meta = json.loads(mf.read_text(encoding="utf-8"))
    display = table_name(DEP_DISPLAY.get(meta["deployment"]))
    if display in mixed_cells:
        meta["v0_cells"] = mixed_cells[display]
        mf.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")
        cells_stored += 1
        # the NER-latency cell belongs to the ner run of the same deployment
        for nf in sorted((ROOT / "runs/ner").rglob("v0-ner-*.meta.json")):
            nmeta = json.loads(nf.read_text(encoding="utf-8"))
            if nmeta["deployment"] == meta["deployment"]:
                nmeta["v0_cells"] = {"ner_s": mixed_cells[display]["ner_s"]}
                nf.write_text(
                    json.dumps(nmeta, indent=1) + "\n", encoding="utf-8", newline="\n")
print(f"v0_cells stored on {cells_stored} mixed metas")

# (6) v0 tier rates on the multilingual metas — by_tier is authoritative for
# the tier cells (it can disagree with by_language rounding by one item)
tier_ranked = 0
for mf in sorted((ROOT / "runs/decision").rglob("v0-ml-*.meta.json")):
    meta = json.loads(mf.read_text(encoding="utf-8"))
    display = DEP_DISPLAY.get(meta["deployment"])
    result_name = README_ROW_ALIASES.get(display, display) if display else None
    if result_name in ml["by_tier"]:
        meta["v0_tier_rates"] = ml["by_tier"][result_name]
        mf.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")
        tier_ranked += 1
print(f"v0_tier_rates added to {tier_ranked} multilingual metas")

# the published v0 table disagrees with by_tier for one cell (hand-edit
# drift in the README): cohere-rerank-v3.5 medium — table 67%, by_tier 61%.
# The acceptance gate reproduces the published table, so the table's value
# wins here, with this note as the audit trail.
V0_TIER_OVERRIDES = {
    "cohere-rerank-v3.5 (OpenRouter)": {"medium": 0.6667},
}
for mf in sorted((ROOT / "runs/decision").rglob("v0-ml-*.meta.json")):
    meta = json.loads(mf.read_text(encoding="utf-8"))
    display = DEP_DISPLAY.get(meta["deployment"])
    if display in V0_TIER_OVERRIDES:
        meta.setdefault("v0_tier_rates", {}).update(V0_TIER_OVERRIDES[display])
        meta["v0_tier_overrides"] = V0_TIER_OVERRIDES[display]
        mf.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"  tier override applied: {display}")
# (7) v0 cost table imported verbatim. The table is its own naming universe
# (different row names than the other tables, a ‡ on Jev, no † on
# Kev, grouped same-priced siblings, hand row order) and four of its
# providers' $ values were list-price DERIVED at v0 time — so each row is
# stored on its deployment's mixed run as v0_cost bookkeeping and rendered
# verbatim; c48/c54 also land in the run totals.
cost_section = readme.split("### Measured cost per hosted run")[1].split(
    "\n### ", 1)[0]
dollar = r"\$[0-9.]+(?: \(free tier\))?"
cost_rows = re.findall(
    rf"^\| (.+?) \| (\$[0-9.e-]+|n/a) \| (\$[0-9.e-]+|n/a) \| ({dollar}) \|\s*$",
    cost_section, re.M)

COST_ALIASES = {"Jev (cloud)": "Jev (cloud, jev-1.14.0)"}


def table_name(display: str | None) -> str | None:
    """display -> the bare result-name form used by the mixed table rows"""
    if display is None:
        return None
    return README_ROW_ALIASES.get(display, display)

def cost_normalize(name: str) -> str:
    n = name.strip().rstrip("†‡").strip()
    n = re.sub(r" \(OpenRouter\)$", "", n)
    return COST_ALIASES.get(n, n)

DEP_BY_DISPLAY: dict[str, list[str]] = {}
for dep_id, disp in DEP_DISPLAY.items():
    DEP_BY_DISPLAY.setdefault(disp, []).append(dep_id)
NORM_DEP: dict[str, list[str]] = {}
for dep_id, disp in DEP_DISPLAY.items():
    NORM_DEP.setdefault(cost_normalize(disp), []).append(dep_id)

def dep_ids_for(name: str) -> list[str]:
    """resolve one cost-table row name to deployment ids (grouped rows -> members)"""
    names = [name]
    if " / " in name:
        m = re.match(r"^(.*?)( \([^)]+\))?$", name.strip())
        base, suffix = (m.group(1) if m else name).strip(), (m.group(2) or "") if m else ""
        parts = [p.strip() for p in base.split(" / ")]
        stem = parts[0].rsplit("-", 1)[0]
        names = [parts[0]] + [stem + p for p in parts[1:]]
        names = [n + suffix for n in names]
    ids: list[str] = []
    for n in names:
        key = cost_normalize(n)
        ids += NORM_DEP.get(key, DEP_BY_DISPLAY.get(key, []))
    return ids

# clear stale keys from earlier passes, then apply fresh
patched = 0
for meta_file in sorted((ROOT / "runs/decision").rglob("*.meta.json")):
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    dirty = False
    for key in ("v0_cost", "v0_cost_derived", "v0_per_question", "v0_cost_suppressed"):
        if key in meta:
            del meta[key]
            dirty = True
    if dirty:
        meta_file.write_text(
            json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")

for row_pos, (row_name, c48, c54, perq) in enumerate(cost_rows, 1):
    ids = dep_ids_for(row_name)
    if not ids:
        print(f"  UNRESOLVED cost row: {row_name!r}")
        continue
    for idx, dep_id in enumerate(ids):
        for meta_file in sorted((ROOT / "runs/decision").rglob("*.meta.json")):
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if meta["deployment"] != dep_id:
                continue
            if meta["suite"] == "decision/mixed-pool":
                meta["v0_cost"] = {
                    "row": row_pos, "name": row_name if idx == 0 else None,
                    "c48": c48, "c54": c54, "perq": perq,
                    "suppressed": idx > 0,
                }
                if c48 != "n/a":
                    meta["totals"]["cost_usd"] = float(c48.lstrip("$"))
                if perq.startswith("$") and "(free tier)" not in perq:
                    meta["v0_per_question"] = float(perq.lstrip("$"))
            elif meta["suite"] == "decision/multilingual-pool" and c54 != "n/a":
                meta["totals"]["cost_usd"] = float(c54.lstrip("$"))
            else:
                continue
            meta_file.write_text(
                json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")
            patched += 1
print(f"v0_cost rows stored on {patched} run metas ({len(cost_rows)} cost rows)")
