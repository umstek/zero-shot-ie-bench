// Span scoring — the v0 NER rule is strict span-set match: the prediction's
// (start, end, label) spans must equal the gold set. On top of that this
// module computes the docs/plan/02 metric set for ner/relation tracks:
// span-level micro P/R/F1 strict + partial, and per-label F1.
//
// Definitions:
//   strict match  — exact (start, end, label) triple, compared as sets
//                   (order-insensitive, duplicates collapse).
//   partial match — a predicted span matches a gold span when both carry the
//                   same label and their character ranges overlap
//                   (intersection > 0, which for non-degenerate spans is
//                   exactly IoU > 0). Matching is one-to-one, greedy by best
//                   IoU. Label mismatches are never partial credit.
//   P/R/F1        — micro: P = tp/(tp+fp), R = tp/(tp+fn),
//                   F1 = 2·tp/(2·tp+fp+fn). Denominator 0 → 0, except the
//                   all-zero case (no gold, no prediction) which scores a
//                   vacuous-perfect F1 of 1. `correct` (the RunItem boolean,
//                   the v0 exact-match rule) is independent of F1: true iff
//                   the predicted span set equals the gold span set.
export type SpanTriple = { start: number; end: number; label: string };

export type Prf = {
  tp: number;
  fp: number;
  fn: number;
  precision: number;
  recall: number;
  f1: number;
};

export type NerItemScore = {
  correct: boolean; // strict set equality — the v0 NER rule
  score: 0 | 1; // 1 when correct (RunItem.score for this track)
  spans: { strict: Prf; partial: Prf };
  per_label: Record<string, Prf>; // strict, this item only
};

export type NerAggregate = {
  items: number;
  strict: Prf;
  partial: Prf;
  per_label: Record<string, Prf>;
};

/** Accept [[start, end, label], …] arrays (the suite/run JSONL form) and
 *  {start, end, label} objects; drop malformed entries; dedupe. */
export function normalizeSpans(raw: unknown): SpanTriple[] {
  if (!Array.isArray(raw)) return [];
  const spans: SpanTriple[] = [];
  for (const entry of raw) {
    let span: SpanTriple | null = null;
    if (Array.isArray(entry) && entry.length >= 3) {
      const [start, end, label] = entry as unknown[];
      if (typeof start === "number" && typeof end === "number" && typeof label === "string") {
        span = { start, end, label };
      }
    } else if (typeof entry === "object" && entry !== null) {
      const obj = entry as { start?: unknown; end?: unknown; label?: unknown };
      if (
        typeof obj.start === "number" &&
        typeof obj.end === "number" &&
        typeof obj.label === "string"
      ) {
        span = { start: obj.start, end: obj.end, label: obj.label };
      }
    }
    if (span !== null) spans.push(span);
  }
  return dedupe(spans);
}

function spanKey(s: SpanTriple): string {
  return `${s.start}:${s.end}:${s.label}`;
}

function dedupe(spans: SpanTriple[]): SpanTriple[] {
  const seen = new Set<string>();
  const out: SpanTriple[] = [];
  for (const s of spans) {
    const key = spanKey(s);
    if (!seen.has(key)) {
      seen.add(key);
      out.push(s);
    }
  }
  return out;
}

/** Micro P/R/F1 with the all-zero → vacuous-perfect convention. */
export function prf(tp: number, fp: number, fn: number): Prf {
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = tp + fp + fn === 0 ? 1 : (2 * tp) / (2 * tp + fp + fn);
  return { tp, fp, fn, precision, recall, f1 };
}

function sumPrf(parts: Iterable<Prf>): Prf {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  for (const p of parts) {
    tp += p.tp;
    fp += p.fp;
    fn += p.fn;
  }
  return prf(tp, fp, fn);
}

/** One-to-one greedy matching by best IoU over same-label overlapping
 *  pairs; returns the number of matched (tp) pairs. */
function matchPartial(pred: SpanTriple[], gold: SpanTriple[]): number {
  const candidates: { pi: number; gi: number; iou: number }[] = [];
  for (let pi = 0; pi < pred.length; pi++) {
    const p = pred[pi];
    if (!p) continue;
    for (let gi = 0; gi < gold.length; gi++) {
      const g = gold[gi];
      if (!g || g.label !== p.label) continue;
      const overlap = Math.min(p.end, g.end) - Math.max(p.start, g.start);
      if (overlap <= 0) continue;
      const union = p.end - p.start + (g.end - g.start) - overlap;
      // union > 0 whenever overlap > 0, but guard anyway
      const iou = union > 0 ? overlap / union : 0;
      candidates.push({ pi, gi, iou });
    }
  }
  // Deterministic: best IoU first, ties by pred then gold index.
  candidates.sort((a, b) => b.iou - a.iou || a.pi - b.pi || a.gi - b.gi);
  const usedPred = new Set<number>();
  const usedGold = new Set<number>();
  let tp = 0;
  for (const c of candidates) {
    if (usedPred.has(c.pi) || usedGold.has(c.gi)) continue;
    usedPred.add(c.pi);
    usedGold.add(c.gi);
    tp += 1;
  }
  return tp;
}

function perLabelPrf(pred: SpanTriple[], gold: SpanTriple[]): Record<string, Prf> {
  const counts = new Map<string, { tp: number; fp: number; fn: number }>();
  const bump = (label: string, field: "tp" | "fp" | "fn") => {
    const c = counts.get(label) ?? { tp: 0, fp: 0, fn: 0 };
    c[field] += 1;
    counts.set(label, c);
  };
  const goldKeys = new Set(gold.map(spanKey));
  const predKeys = new Set(pred.map(spanKey));
  for (const g of gold) if (predKeys.has(spanKey(g))) bump(g.label, "tp");
  for (const g of gold) if (!predKeys.has(spanKey(g))) bump(g.label, "fn");
  for (const p of pred) if (!goldKeys.has(spanKey(p))) bump(p.label, "fp");
  const out: Record<string, Prf> = {};
  for (const label of [...counts.keys()].sort()) {
    const c = counts.get(label);
    if (c) out[label] = prf(c.tp, c.fp, c.fn);
  }
  return out;
}

/** Score one NER item. `prediction` and `gold` are span lists in the
 *  [[start, end, label], …] form (objects accepted too). */
export function scoreNer(prediction: unknown, gold: unknown): NerItemScore {
  const pred = normalizeSpans(prediction);
  const goldSpans = normalizeSpans(gold);

  const predKeys = new Set(pred.map(spanKey));
  const goldKeys = new Set(goldSpans.map(spanKey));
  let strictTp = 0;
  for (const key of goldKeys) if (predKeys.has(key)) strictTp += 1;
  const strict = prf(strictTp, pred.length - strictTp, goldSpans.length - strictTp);

  const partialTp = matchPartial(pred, goldSpans);
  const partial = prf(partialTp, pred.length - partialTp, goldSpans.length - partialTp);

  return {
    correct: strictTp === goldSpans.length && pred.length === goldSpans.length,
    score: strictTp === goldSpans.length && pred.length === goldSpans.length ? 1 : 0,
    spans: { strict, partial },
    per_label: perLabelPrf(pred, goldSpans),
  };
}

/** Micro-aggregate item scores into corpus-level strict/partial/per-label
 *  P/R/F1 (docs/plan/02: span-level micro-F1 is the ner track primary). */
export function aggregateNer(scores: readonly NerItemScore[]): NerAggregate {
  const perLabelCounts = new Map<string, { tp: number; fp: number; fn: number }>();
  for (const s of scores) {
    for (const [label, p] of Object.entries(s.per_label)) {
      const c = perLabelCounts.get(label) ?? { tp: 0, fp: 0, fn: 0 };
      c.tp += p.tp;
      c.fp += p.fp;
      c.fn += p.fn;
      perLabelCounts.set(label, c);
    }
  }
  const perLabel: Record<string, Prf> = {};
  for (const label of [...perLabelCounts.keys()].sort()) {
    const c = perLabelCounts.get(label);
    if (c) perLabel[label] = prf(c.tp, c.fp, c.fn);
  }
  return {
    items: scores.length,
    strict: sumPrf(scores.map((s) => s.spans.strict)),
    partial: sumPrf(scores.map((s) => s.spans.partial)),
    per_label: perLabel,
  };
}
