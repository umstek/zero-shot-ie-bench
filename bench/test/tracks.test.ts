// Track layer tests: suite loader (legacy byte-exact hashes), decision
// exact-choice scoring (Brier/ECE hand-computed), NER span scoring
// (strict/partial/per-label), and the track registry.

import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  aggregateNer,
  type CalibrationSample,
  computeSuiteHash,
  expectedCalibrationError,
  getScorer,
  getTrack,
  loadSuite,
  reliabilityBins,
  runScorer,
  scoreDecision,
  scoreNer,
  tracks,
} from "../src/tracks/index.ts";

// known values printed by legacy/extract_suites.py when it wrote the suites
const KNOWN_HASHES: Record<string, string> = {
  "decision/mixed-pool": "sha256:900aba60ecb6c914ca0a04f86a4ac7e8f5c9e828799196ee02ad3b38c32cc711",
  "decision/multilingual-pool":
    "sha256:ec4631c6905b8114c066431a83d75ab8e5644b94428ac86d383d7a4dd5d112ae",
  "ner/graded-pool": "sha256:35af2f255ac13427a43c7d44aca9eed9b5b10f8e3b9ee714238df497ab9f1fd2",
};

// sha256("test\n" + "1\n" + "2\n") — suite.yaml then a.jsonl then b.jsonl
const SYNTHETIC_HASH = "4ec8ebcbcc8bb156e94dedeae5153ede8930c179cfd041558895dd5978110f06";

function knownHash(id: string): string {
  const hash = KNOWN_HASHES[id];
  if (!hash) throw new Error(`no known hash recorded for ${id}`);
  return hash;
}

describe("suite loader", () => {
  test("decision/mixed-pool: meta, items, legacy hash", () => {
    const suite = loadSuite("decision/mixed-pool");
    expect(suite.id).toBe("decision/mixed-pool");
    expect(suite.meta.version).toBe(1);
    expect(suite.meta.track).toBe("decision");
    expect(suite.meta.languages).toEqual(["en"]);
    expect(suite.meta.scorer).toBe("exact-choice");
    expect(suite.meta.items_license).toBeNull();
    expect(suite.items).toHaveLength(48);
    const first = suite.items[0];
    if (!first) throw new Error("mixed-pool has no items");
    expect(first.item_id).toBe("mp-001");
    expect(first.gold).toBe("positive");
    expect(first.question.options).toEqual(["positive", "negative", "neutral"]);
    expect(suite.suite_hash).toBe(knownHash("decision/mixed-pool"));
    expect(suite.hash).toBe(suite.suite_hash); // cli.ts alias
  });

  test("decision/multilingual-pool: 54 rows, parallel item_ids across 9 files", () => {
    const suite = loadSuite("decision/multilingual-pool");
    expect(suite.items).toHaveLength(54);
    expect(suite.files).toHaveLength(9);
    expect(new Set(suite.files.map((f) => f.lang))).toEqual(new Set(suite.meta.languages));
    const first = suite.files[0];
    if (!first) throw new Error("multilingual-pool has no files");
    const expectedIds = [...new Set(first.items.map((i) => i.item_id))].sort();
    expect(expectedIds).toEqual(["ml-001", "ml-002", "ml-003", "ml-004", "ml-005", "ml-006"]);
    for (const file of suite.files) {
      expect([...new Set(file.items.map((i) => i.item_id))].sort()).toEqual(expectedIds);
    }
    expect(suite.suite_hash).toBe(knownHash("decision/multilingual-pool"));
  });

  test("ner/graded-pool: 18 items with span-triple gold, legacy hash", () => {
    const suite = loadSuite("ner/graded-pool");
    expect(suite.meta.scorer).toBe("span-micro-f1");
    expect(suite.items).toHaveLength(18);
    const first = suite.items[0];
    if (!first) throw new Error("graded-pool has no items");
    expect(first.item_id).toBe("np-001");
    expect(first.question.type).toBe("ner-labels");
    expect(first.gold).toEqual([
      [0, 15, "person"],
      [33, 42, "location"],
    ]);
    expect(suite.suite_hash).toBe(knownHash("ner/graded-pool"));
  });

  test("decision/gate-12 is the first 12 of the mixed pool", () => {
    const gate = loadSuite("decision/gate-12");
    const mixed = loadSuite("decision/mixed-pool");
    expect(gate.items).toHaveLength(12);
    expect(gate.items.map((i) => i.item_id)).toEqual(
      mixed.items.slice(0, 12).map((i) => i.item_id),
    );
  });

  test("hash covers every items/ entry in sorted filename order, raw bytes", () => {
    const dir = mkdtempSync(join(tmpdir(), "zsib-tracks-"));
    try {
      mkdirSync(join(dir, "items"));
      writeFileSync(join(dir, "suite.yaml"), "test\n");
      writeFileSync(join(dir, "items", "b.jsonl"), "2\n");
      writeFileSync(join(dir, "items", "a.jsonl"), "1\n");
      expect(computeSuiteHash(dir)).toBe(`sha256:${SYNTHETIC_HASH}`);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  test("loadSuite accepts a custom suites root", () => {
    const root = mkdtempSync(join(tmpdir(), "zsib-tracks-"));
    try {
      const dir = join(root, "decision", "fake");
      mkdirSync(join(dir, "items"), { recursive: true });
      writeFileSync(
        join(dir, "suite.yaml"),
        [
          "id: decision/fake",
          "version: 1",
          "track: decision",
          "languages: [en]",
          "scorer: exact-choice",
          "items_license: null",
          "",
        ].join("\n"),
      );
      writeFileSync(
        join(dir, "items", "en.jsonl"),
        `${JSON.stringify({
          item_id: "f-001",
          lang: "en",
          tier: null,
          input: { text: "hi", images: [] },
          question: { type: "choice", options: ["a", "b"] },
          gold: "a",
        })}\n`,
      );
      const suite = loadSuite("decision/fake", { root });
      expect(suite.items).toHaveLength(1);
      expect(suite.suite_hash).toBe(computeSuiteHash(dir));
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("loadSuite rejects missing suites, malformed ids, and traversal", () => {
    expect(() => loadSuite("decision/does-not-exist")).toThrow(/suite not found/);
    expect(() => loadSuite("nopart")).toThrow(/<track>\/<suite-id>/);
    expect(() => loadSuite("decision/")).toThrow(/<track>\/<suite-id>/);
    expect(() => loadSuite("../escape")).toThrow(/invalid suite id/);
  });
});

describe("decision scorer (exact-choice)", () => {
  test("exact match, mismatch, and null prediction (the v0 rule)", () => {
    expect(scoreDecision("positive", "positive")).toEqual({
      correct: true,
      score: 1,
      prob: null,
      brier: null,
    });
    const wrong = scoreDecision("negative", "positive");
    expect(wrong.correct).toBe(false);
    expect(wrong.score).toBe(0);
    expect(scoreDecision(null, "positive").correct).toBe(false);
    // no normalization: the compare is exactly ===
    expect(scoreDecision(" Positive", "positive").correct).toBe(false);
  });

  test("prob surfaces only with a selected choice", () => {
    expect(scoreDecision("positive", "positive", { prob: 0.93 }).prob).toBe(0.93);
    expect(scoreDecision(null, "positive", { prob: 0.93 }).prob).toBeNull();
  });

  test("brier: sum over options of (p_i - y_i)^2", () => {
    // hand-computed: (0.8-1)^2 + (0.2-0)^2 = 0.04 + 0.04
    const hit = scoreDecision("a", "a", { options: ["a", "b"], probs: { a: 0.8, b: 0.2 } });
    expect(hit.brier).toBeCloseTo(0.08, 12);
    // same vector, wrong gold: (0.8-0)^2 + (0.2-1)^2 = 0.64 + 0.64
    const miss = scoreDecision("a", "b", { options: ["a", "b"], probs: { a: 0.8, b: 0.2 } });
    expect(miss.brier).toBeCloseTo(1.28, 12);
    // three options: 0.6^2 + 0.3^2 + (0.1-1)^2 = 0.36 + 0.09 + 0.81
    const three = scoreDecision("positive", "neutral", {
      options: ["positive", "negative", "neutral"],
      probs: { positive: 0.6, negative: 0.3, neutral: 0.1 },
    });
    expect(three.brier).toBeCloseTo(1.26, 12);
    // missing option probability counts as 0 → the [0,2] maximum
    const worst = scoreDecision("a", "c", { options: ["a", "b", "c"], probs: { a: 1 } });
    expect(worst.brier).toBeCloseTo(2, 12);
    // without an option list, the sorted prob keys bound the sum
    const keyless = scoreDecision("a", "a", { probs: { b: 0.2, a: 0.8 } });
    expect(keyless.brier).toBeCloseTo(0.08, 12);
  });

  test("brier is null without a full vector or with non-finite probabilities", () => {
    expect(scoreDecision("a", "a", { prob: 0.9 }).brier).toBeNull();
    expect(scoreDecision("a", "a").brier).toBeNull();
    expect(
      scoreDecision("a", "a", { options: ["a", "b"], probs: { a: 0.5, b: Number.NaN } }).brier,
    ).toBeNull();
  });

  test("object predictions: {choice, prob, probs}", () => {
    const r = scoreDecision(
      { choice: "positive", prob: 0.9, probs: { positive: 0.9, negative: 0.1 } },
      "positive",
      { options: ["positive", "negative"] },
    );
    expect(r.correct).toBe(true);
    expect(r.prob).toBe(0.9);
    expect(r.brier).toBeCloseTo(0.02, 12); // 0.1^2 + 0.1^2
  });

  test("ece: 10-bin equal-width reliability, hand-computed", () => {
    // bins [0.9,1.0]=conf .95 correct, [0.6,0.7)=conf .65 wrong,
    // [0.8,0.9)=conf .85 correct, [0.3,0.4)=conf .35 wrong
    // ECE = (0.05 + 0.65 + 0.15 + 0.35) / 4 = 0.3
    const samples: CalibrationSample[] = [
      { prob: 0.95, correct: true },
      { prob: 0.65, correct: false },
      { prob: 0.85, correct: true },
      { prob: 0.35, correct: false },
    ];
    expect(expectedCalibrationError(samples)).toBeCloseTo(0.3, 12);
  });

  test("ece: perfectly calibrated bin contributes 0; empty input is null", () => {
    const perfect: CalibrationSample[] = [
      { prob: 0.5, correct: true },
      { prob: 0.5, correct: false },
    ];
    expect(expectedCalibrationError(perfect)).toBe(0);
    expect(expectedCalibrationError([])).toBeNull();
  });

  test("reliability bins: 10 equal-width bins, prob 1.0 lands in the last", () => {
    const bins = reliabilityBins([
      { prob: 1.0, correct: true },
      { prob: 1.0, correct: false },
    ]);
    expect(bins).toHaveLength(10);
    expect(bins[0]?.lower).toBe(0);
    expect(bins[0]?.upper).toBeCloseTo(0.1, 12);
    const last = bins[9];
    if (!last) throw new Error("missing last bin");
    expect(last.n).toBe(2);
    expect(last.accuracy).toBe(0.5);
    expect(last.confidence).toBe(1);
    // empty bins carry n=0 and null statistics
    expect(bins[5]?.n).toBe(0);
    expect(bins[5]?.accuracy).toBeNull();
  });
});

describe("ner scorer (span-micro-f1)", () => {
  const GOLD = [
    [0, 15, "person"],
    [33, 42, "location"],
  ] as const;

  test("strict set match (the v0 rule), order-insensitive", () => {
    const hit = scoreNer([...GOLD], [...GOLD].reverse());
    expect(hit.correct).toBe(true);
    expect(hit.score).toBe(1);
    expect(hit.spans.strict).toEqual({ tp: 2, fp: 0, fn: 0, precision: 1, recall: 1, f1: 1 });
    expect(hit.spans.partial.tp).toBe(2);
  });

  test("boundary error: strict miss, partial credit", () => {
    const r = scoreNer(
      [[0, 5, "person"]],
      [
        [0, 3, "person"],
        [8, 12, "location"],
      ],
    );
    expect(r.correct).toBe(false);
    expect(r.score).toBe(0);
    // strict: 0 exact hits, 1 extra prediction, 2 missed golds
    expect(r.spans.strict.tp).toBe(0);
    expect(r.spans.strict.fp).toBe(1);
    expect(r.spans.strict.fn).toBe(2);
    // partial: [0,5] overlaps [0,3] (IoU 3/5) with the same label
    expect(r.spans.partial.tp).toBe(1);
    expect(r.spans.partial.fp).toBe(0);
    expect(r.spans.partial.fn).toBe(1);
    expect(r.spans.partial.precision).toBe(1);
    expect(r.spans.partial.recall).toBeCloseTo(0.5, 12);
    expect(r.spans.partial.f1).toBeCloseTo(2 / 3, 12);
  });

  test("label mismatch never earns partial credit", () => {
    const r = scoreNer([[0, 5, "location"]], [[0, 5, "person"]]);
    expect(r.spans.strict.tp).toBe(0);
    expect(r.spans.partial.tp).toBe(0);
    expect(r.spans.partial.fp).toBe(1);
    expect(r.spans.partial.fn).toBe(1);
  });

  test("empty predictions on empty gold are vacuously correct; fn otherwise", () => {
    const emptyBoth = scoreNer([], []);
    expect(emptyBoth.correct).toBe(true);
    expect(emptyBoth.spans.strict.f1).toBe(1); // all-zero → vacuous-perfect convention
    const missed = scoreNer([], GOLD);
    expect(missed.correct).toBe(false);
    expect(missed.spans.strict.fn).toBe(2);
    expect(missed.spans.strict.f1).toBe(0);
  });

  test("duplicate predictions collapse; object spans accepted", () => {
    const r = scoreNer(
      [[0, 4, "person"], [0, 4, "person"], { start: 0, end: 4, label: "person" }],
      [[0, 4, "person"]],
    );
    expect(r.spans.strict.tp).toBe(1);
    expect(r.spans.strict.fp).toBe(0);
    expect(r.correct).toBe(true);
  });

  test("aggregateNer: corpus micro strict/partial and per-label F1", () => {
    const a = scoreNer([[0, 4, "person"]], [[0, 4, "person"]]);
    const b = scoreNer(
      [[10, 18, "location"]],
      [
        [0, 4, "person"],
        [10, 18, "location"],
      ],
    );
    const agg = aggregateNer([a, b]);
    expect(agg.items).toBe(2);
    // strict: person tp+fn, location tp → tp2 fp0 fn1
    expect(agg.strict.tp).toBe(2);
    expect(agg.strict.fp).toBe(0);
    expect(agg.strict.fn).toBe(1);
    expect(agg.strict.precision).toBe(1);
    expect(agg.strict.recall).toBeCloseTo(2 / 3, 12);
    expect(agg.strict.f1).toBeCloseTo(0.8, 12); // 2·2 / (2·2+0+1)
    const person = agg.per_label.person;
    const location = agg.per_label.location;
    if (!person || !location) throw new Error("missing per-label aggregates");
    expect(person.tp).toBe(1);
    expect(person.fn).toBe(1);
    expect(person.f1).toBeCloseTo(2 / 3, 12);
    expect(location.f1).toBe(1);
  });
});

describe("track registry", () => {
  test("tracks expose scorer + suites namespace", () => {
    expect(tracks.decision?.id).toBe("decision");
    expect(tracks.ner?.id).toBe("ner");
    expect(tracks.decision?.suites.list()).toEqual([
      "canary",
      "gate-12",
      "mixed-pool",
      "multilingual-pool",
    ]);
    expect(tracks.ner?.suites.list()).toEqual(["canary", "graded-pool"]);
    const loaded = tracks.ner?.suites.load("graded-pool");
    expect(loaded?.items).toHaveLength(18);
  });

  test("getTrack throws on unknown ids", () => {
    expect(getTrack("decision").id).toBe("decision");
    expect(() => getTrack("rerank")).toThrow(/unknown track/);
  });

  test("getScorer: the run loop's scoreItem handle", () => {
    const decision = getScorer("decision");
    expect(decision.track.id).toBe("decision");
    const item = loadSuite("decision/mixed-pool").items[0];
    if (!item) throw new Error("no item");
    expect(decision.scoreItem(item, "positive")).toEqual({ correct: true, score: 1 });
    expect(decision.scoreItem(item, null)).toEqual({ correct: false, score: 0 });
    const ner = getScorer("ner");
    const np1 = loadSuite("ner/graded-pool").items[0];
    if (!np1) throw new Error("no item");
    expect(ner.scoreItem(np1, np1.gold)).toEqual({ correct: true, score: 1 });
    expect(() => getScorer("rerank")).toThrow(/unknown track/);
  });

  test("runScorer scores a real decision item end-to-end", () => {
    const item = loadSuite("decision/mixed-pool").items[0];
    if (!item) throw new Error("no item");
    const hit = runScorer("decision", { prediction: "positive" }, item);
    expect(hit.correct).toBe(true);
    expect(hit.score).toBe(1);
    expect(hit.metrics).toEqual({}); // no probability vector → no brier
    // gold is "positive": (0.6-1)^2 + 0.2^2 + 0.2^2 = 0.16 + 0.04 + 0.04
    const unsure = runScorer(
      "decision",
      {
        prediction: "positive",
        prob: 0.6,
        probs: { positive: 0.6, negative: 0.2, neutral: 0.2 },
      },
      item,
    );
    expect(unsure.correct).toBe(true);
    expect(unsure.prob).toBe(0.6);
    expect(unsure.metrics.brier).toBeCloseTo(0.24, 12);
  });

  test("runScorer scores a real ner item end-to-end", () => {
    const item = loadSuite("ner/graded-pool").items[0];
    if (!item) throw new Error("no item");
    const hit = runScorer(
      "ner",
      {
        prediction: [
          [0, 15, "person"],
          [33, 42, "location"],
        ],
      },
      item,
    );
    expect(hit.correct).toBe(true);
    expect(hit.prob).toBeNull();
    expect(hit.metrics.strict_f1).toBe(1);
    expect(hit.metrics.partial_f1).toBe(1);
    expect(hit.per_label?.person?.tp).toBe(1);
    const miss = runScorer("ner", { prediction: [[0, 15, "person"]] }, item);
    expect(miss.metrics.strict_fn).toBe(1);
    expect(miss.metrics.partial_fn).toBe(1);
  });
});
