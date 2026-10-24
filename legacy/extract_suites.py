r"""One-off extractor: the v0 question pools -> suites/<track>/<suite-id>/ v1.

Creates (and recreates, byte-identically) six suites:

  decision/mixed-pool        48 en choice items   mp-001..mp-048
  decision/gate-12           12 en choice items   mp-001..mp-012 (subset view)
  decision/multilingual-pool 6 parallel items x 9 languages (54 rows) ml-001..ml-006
  decision/canary            12 en + 12 multilingual instance subset rows
  ner/graded-pool            18 en ner-labels items  np-001..np-018
  ner/canary                 all 18 (subset view)

Sources (all read-only imports / reads):
  - legacy/bench_graded.py  SENTIMENT/TOPIC (18-82) and NER (84-143) pools,
    keyed easy/medium/hard; validate_ner_pools guards the span texts.
  - legacy/bench.py         SENTIMENT_LABELS (47-51), TOPIC_LABELS (80-85),
    NER_LABELS (102), spans_of (133-135).
  - legacy/bench_spectrum.py CLS_QUESTIONS (113-118) and NER_QUESTIONS
    (119-122) define the flattening (tier-major) and the gold span form
    (sorted (start, end, label) triples); NOT imported (it pulls engines).
  - legacy/bench_multilingual.py LANGUAGES (119-195), SENTIMENT_LABELS
    (105-109), TIERS (197-201).
  - _trials/trial_common.py CLS_POOL (24-29): the 12-question trial gate is
    literally the first 12 of the mixed pool (sentiment easy 8 + sentiment
    medium first 4); called "the first-12 gate" at bench_spectrum.py:256.
  - catalog/languages.yaml  language name -> code + tier.

The bench_graded easy/medium/hard groups are question-authoring tiers, not
the catalog popular/medium/rare language tiers, so mixed-pool item tier is
null. No license is stated anywhere for the question items (the repo MIT
LICENSE covers the software), so items_license is null.

Rerank: the v0 rerankers answered the SAME 48 classification questions via
(instruction, label) argmax (bench_spectrum.py reranker branch, line ~1004;
catalog/tracks.yaml: "benched as a classifier") - there is no distinct
rerank question pool, so no rerank suite is created.

suite_hash (printed per suite at the end, never stored): "sha256:" + hex of
sha256 over suite.yaml bytes followed by each items/*.jsonl sorted by
filename. suite.yaml contains no timestamps, so re-running reproduces the
same bytes and the same hashes. JSONL is UTF-8 (no BOM), LF, one JSON
object per line; every line matches packages/schema SuiteItem.

Run from the repo root:
    .venv/Scripts/python.exe legacy/extract_suites.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)  # import the legacy pool modules read-only

from bench import NER_LABELS, SENTIMENT_LABELS, TOPIC_LABELS, spans_of  # noqa: E402
from bench_graded import NER, SENTIMENT, TOPIC, validate_ner_pools  # noqa: E402
import bench_multilingual as mlang  # noqa: E402

VERSION = 1
SCORER_CHOICE = "exact-choice"
SCORER_NER = "span-micro-f1"
ITEMS_LICENSE = None  # no license stated for the question items; do not invent

TIERS_SOURCE = ("easy", "medium", "hard")  # bench_graded authoring tiers


# --------------------------------------------------------------------------
# catalog languages: name -> (code, tier), in file order (en excluded later)
# --------------------------------------------------------------------------
with open(os.path.join(ROOT, "catalog", "languages.yaml"),
          encoding="utf-8") as fh:
    _CATALOG_LANGS = yaml.safe_load(fh)
LANG_CODE = {e["name"]: e["code"] for e in _CATALOG_LANGS}
LANG_TIER = {e["name"]: e["tier"] for e in _CATALOG_LANGS}
CATALOG_CODES = [e["code"] for e in _CATALOG_LANGS if e["code"] != "en"]


# --------------------------------------------------------------------------
# row builders (field names exactly per packages/schema/src/suite.ts)
# --------------------------------------------------------------------------
def choice_row(item_id: str, lang: str, tier: str | None, text: str,
               options: list[str], gold: str) -> dict:
    return {"item_id": item_id, "lang": lang, "tier": tier,
            "input": {"text": text, "images": []},
            "question": {"type": "choice", "options": options},
            "gold": gold}


def ner_row(item_id: str, text: str,
            gold: list[tuple[int, int, str]]) -> dict:
    return {"item_id": item_id, "lang": "en", "tier": None,
            "input": {"text": text, "images": []},
            "question": {"type": "ner-labels", "options": list(NER_LABELS)},
            "gold": [[s, e, lab] for s, e, lab in gold]}


def ner_gold(text: str, truth: list[tuple[str, str]]) -> list[tuple[int, int, str]]:
    """Exactly what bench_spectrum.py NER_QUESTIONS stores: sorted
    spans_of(text, truth) - (start, end, label) via text.index."""
    return sorted(spans_of(text, truth))


# --------------------------------------------------------------------------
# pools
# --------------------------------------------------------------------------
SENT_OPTIONS = list(SENTIMENT_LABELS)   # positive, negative, neutral
TOPIC_OPTIONS = list(TOPIC_LABELS)      # technology, business, sports, politics

# same flattening expression as bench_spectrum.py CLS_QUESTIONS
CLS_QUESTIONS = (
    [{"task": "sentiment", "text": t, "gold": g}
     for tier in TIERS_SOURCE for t, g in SENTIMENT[tier]]
    + [{"task": "topic", "text": t, "gold": g}
       for tier in TIERS_SOURCE for t, g in TOPIC[tier]]
)
# same flattening expression as bench_spectrum.py NER_QUESTIONS
NER_QUESTIONS = [
    {"text": text, "gold": ner_gold(text, truth)}
    for tier in TIERS_SOURCE for text, truth in NER[tier]
]

assert len(CLS_QUESTIONS) == 48, len(CLS_QUESTIONS)
assert len(NER_QUESTIONS) == 18, len(NER_QUESTIONS)
validate_ner_pools()

MIXED_ROWS = [choice_row(f"mp-{i:03d}", "en", None, q["text"],
                         SENT_OPTIONS if q["task"] == "sentiment"
                         else TOPIC_OPTIONS, q["gold"])
              for i, q in enumerate(CLS_QUESTIONS, 1)]
GATE_ROWS = MIXED_ROWS[:12]  # trial_common.CLS_POOL: first 12 of the mixed pool
assert all(q["task"] == "sentiment" for q in CLS_QUESTIONS[:12])

# multilingual pool: 6 parallel sentence positions per language
ML_POSITIONS = 6
ML_FILES: dict[str, list[dict]] = {}
for name, items in mlang.LANGUAGES.items():
    code = LANG_CODE[name]
    tier = LANG_TIER[name]
    assert tier == mlang.TIERS[name], (name, tier, mlang.TIERS[name])
    assert len(items) == ML_POSITIONS, (name, len(items))
    # positions 1-2 positive, 3-4 negative, 5-6 neutral (per the module's
    # English template comment P1 P2 N1 N2 T1 T2) - the canary rule relies
    # on this mapping
    assert [g for _, g in items] == ["positive", "positive", "negative",
                                     "negative", "neutral", "neutral"], name
    rows = [choice_row(f"ml-{p:03d}", code, tier, text,
                       list(mlang.SENTIMENT_LABELS), gold)
            for p, (text, gold) in enumerate(items, 1)]
    ML_FILES[code] = rows

# canary multilingual instances: languages in catalog order round-robin over
# the six parallel positions in the fixed cycle (1, 3, 5, 2, 4, 6)
# (positive, negative, neutral, positive, negative, neutral) until 12
# instances - covers all 9 languages with a balanced 4/4/4 label mix
POSITION_CYCLE = (1, 3, 5, 2, 4, 6)
CANARY_ML = [(CATALOG_CODES[i % len(CATALOG_CODES)],
              POSITION_CYCLE[i % len(POSITION_CYCLE)])
             for i in range(12)]

NER_ROWS = [ner_row(f"np-{i:03d}", q["text"], q["gold"])
            for i, q in enumerate(NER_QUESTIONS, 1)]


# --------------------------------------------------------------------------
# suite writer + suite_hash
# --------------------------------------------------------------------------
def write_suite(track: str, suite_id: str, languages: list[str],
                scorer: str, notes: str,
                files: dict[str, list[dict]]) -> str:
    """Write suites/<track>/<suite-id>/{suite.yaml,items/<lang>.jsonl} and
    return the suite_hash computed over the written bytes."""
    base = os.path.join(ROOT, "suites", track, suite_id)
    items_dir = os.path.join(base, "items")
    os.makedirs(items_dir, exist_ok=True)

    meta = {
        "id": f"{track}/{suite_id}",
        "version": VERSION,
        "track": track,
        "languages": languages,
        "scorer": scorer,
        "items_license": ITEMS_LICENSE,
        "notes": notes,
    }
    meta_bytes = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True,
                                default_flow_style=False).encode("utf-8")
    with open(os.path.join(base, "suite.yaml"), "wb") as fh:
        fh.write(meta_bytes)

    for lang in sorted(files):  # deterministic order; also the hash order
        data = "".join(json.dumps(row, ensure_ascii=False) + "\n"
                       for row in files[lang]).encode("utf-8")
        assert b"\r" not in data
        with open(os.path.join(items_dir, f"{lang}.jsonl"), "wb") as fh:
            fh.write(data)

    return suite_hash(base)


def suite_hash(base: str) -> str:
    """sha256 over suite.yaml bytes + each items/*.jsonl sorted by filename."""
    h = hashlib.sha256()
    with open(os.path.join(base, "suite.yaml"), "rb") as fh:
        h.update(fh.read())
    items_dir = os.path.join(base, "items")
    for name in sorted(os.listdir(items_dir)):
        with open(os.path.join(items_dir, name), "rb") as fh:
            h.update(fh.read())
    return "sha256:" + h.hexdigest()


# --------------------------------------------------------------------------
# suites
# --------------------------------------------------------------------------
SUITE_NOTES_MIXED = (
    "v0 mixed classification pool: 24 sentiment + 24 topic questions, "
    "flattened tier-major (easy, medium, hard) from bench_graded.SENTIMENT "
    "and bench_graded.TOPIC exactly as legacy/bench_spectrum.py "
    "CLS_QUESTIONS (lines 113-118); item ids mp-001..mp-048 follow that "
    "order. Option lists are the label sets from legacy/bench.py "
    "(SENTIMENT_LABELS lines 47-51, TOPIC_LABELS lines 80-85); gold is the "
    "label string the bench scored against. bench_graded's easy/medium/hard "
    "groups are question-authoring tiers, not catalog language tiers, so "
    "item tier is null. No license is stated for the question items "
    "anywhere in the repo, so items_license is null.")

SUITE_NOTES_GATE = (
    "The v0 trial gate: literally the first 12 questions of the mixed pool "
    "(sentiment easy 8 + sentiment medium first 4) - the CLS_POOL of "
    "_trials/trial_common.py, called the first-12 gate at "
    "legacy/bench_spectrum.py:256 and in the README trial entries. Items "
    "are the same mp-001..mp-012 rows as decision/mixed-pool v1, identical "
    "content (a subset view, not new questions).")

SUITE_NOTES_ML = (
    "v0 multilingual pool: 6 parallel sentiment texts (2 positive, 2 "
    "negative, 2 neutral) in each of 9 languages, from "
    "legacy/bench_multilingual.py LANGUAGES (lines 119-195) with gold "
    "labels from its tuples and options from its SENTIMENT_LABELS (lines "
    "105-109). Parallel items: ml-001..ml-006 identify the six sentence "
    "positions and the same item_id appears in every language file (54 "
    "rows total). Language codes and tiers come from "
    "catalog/languages.yaml; English is the template source and was not "
    "benched in v0. Ground truth was back-translation verified per the "
    "module docstring.")

SUITE_NOTES_CANARY_DECISION = (
    "Canary subset view of decision/mixed-pool + "
    "decision/multilingual-pool; every row is identical to its parent "
    "suite (no new questions). English: the 12 gate items mp-001..mp-012 "
    "(same as decision/gate-12). Multilingual: 12 instances covering all 9 "
    "languages with a balanced 4/4/4 label mix - languages in catalog "
    "order round-robin over the six parallel positions in the fixed cycle "
    "1,3,5,2,4,6 (pos, neg, neu, pos, neg, neu) until 12 instances: es "
    "ml-001+ml-002, fr ml-003+ml-004, zh ml-005+ml-006, vi ml-002, tr "
    "ml-004, uk ml-006, si ml-001, is ml-003, cy ml-005.")

SUITE_NOTES_NER = (
    "v0 NER pool: 18 English texts with gold (start, end, label) span "
    "triples, from bench_graded.NER (6 per tier, tier-major "
    "easy/medium/hard) as flattened by legacy/bench_spectrum.py "
    "NER_QUESTIONS (lines 119-122); offsets computed with bench.spans_of "
    "(legacy/bench.py:133) exactly as the bench did. Options are "
    "legacy/bench.py NER_LABELS (person, company, product, location). "
    "Suite id from the source module bench_graded.py; the README reports "
    "this pool as NER exact-match (18 questions, extractors only). The "
    "older flat bench.py 10-text NER set (30 gold entities, "
    "results/bench_results.json) is not extracted.")

SUITE_NOTES_CANARY_NER = (
    "Canary subset view of ner/graded-pool: the pool has 18 items (fewer "
    "than the ~24 canary target), so the whole parent suite is included "
    "verbatim - identical item_ids np-001..np-018 and identical rows.")


def canary_decision_files() -> dict[str, list[dict]]:
    files: dict[str, list[dict]] = {"en": GATE_ROWS}
    by_code_pos = {(row["lang"], int(row["item_id"][3:])): row
                   for rows in ML_FILES.values() for row in rows}
    for code, pos in CANARY_ML:
        files.setdefault(code, []).append(by_code_pos[(code, pos)])
    for code in files:
        files[code] = sorted(files[code], key=lambda r: r["item_id"])
    return files


def main() -> None:
    results: list[tuple[str, dict[str, int], str]] = []

    results.append((
        "decision/mixed-pool", {"en": len(MIXED_ROWS)},
        write_suite("decision", "mixed-pool", ["en"], SCORER_CHOICE,
                    SUITE_NOTES_MIXED, {"en": MIXED_ROWS})))
    results.append((
        "decision/gate-12", {"en": len(GATE_ROWS)},
        write_suite("decision", "gate-12", ["en"], SCORER_CHOICE,
                    SUITE_NOTES_GATE, {"en": GATE_ROWS})))
    results.append((
        "decision/multilingual-pool",
        {c: len(ML_FILES[c]) for c in CATALOG_CODES},
        write_suite("decision", "multilingual-pool", CATALOG_CODES,
                    SCORER_CHOICE, SUITE_NOTES_ML, ML_FILES)))
    canary_files = canary_decision_files()
    results.append((
        "decision/canary",
        {c: len(canary_files[c]) for c in CATALOG_CODES},
        write_suite("decision", "canary",
                    ["en"] + CATALOG_CODES, SCORER_CHOICE,
                    SUITE_NOTES_CANARY_DECISION, canary_files)))
    results.append((
        "ner/graded-pool", {"en": len(NER_ROWS)},
        write_suite("ner", "graded-pool", ["en"], SCORER_NER,
                    SUITE_NOTES_NER, {"en": NER_ROWS})))
    results.append((
        "ner/canary", {"en": len(NER_ROWS)},
        write_suite("ner", "canary", ["en"], SCORER_NER,
                    SUITE_NOTES_CANARY_NER, {"en": NER_ROWS})))

    total = 0
    print("suite                       items per language                            suite_hash")
    for suite_id, counts, digest in results:
        n = sum(counts.values())
        total += n
        per_lang = " ".join(f"{c}={k}" for c, k in counts.items())
        print(f"{suite_id:<27} {n:>4}  {per_lang:<45} {digest}")
    print(f"\n{len(results)} suites, {total} item rows written under suites/ "
          f"(version {VERSION}, scorer exact-choice/span-micro-f1, "
          f"items_license {ITEMS_LICENSE}).")
    print("suite_hash = sha256(suite.yaml bytes + items/*.jsonl sorted by "
          "filename); deterministic, not stored in suite.yaml.")


if __name__ == "__main__":
    main()
