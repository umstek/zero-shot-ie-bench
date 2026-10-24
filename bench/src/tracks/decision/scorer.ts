// Exact-choice scoring — the v0 decision rule: prediction === gold, a plain
// string comparison with no normalization (docs/plan/02: decision primary
// metric is accuracy; secondary Brier + ECE).
//
// Formulas (kept here so the board and the tests cite the same definitions):
//
//   Correctness (v0 rule)
//     correct := (choice === gold)  — exact string compare, untrimmed.
//     A null prediction (no parseable choice, e.g. transport/parse error)
//     is simply incorrect.
//
//   Brier score, multiclass sum form (Brier 1950; Murphy 1973 decomposes it
//   into reliability + resolution + uncertainty):
//     BS = Σ_i (p_i − y_i)²
//     summed over the item's option list, p_i = probability the adapter
//     reports for option i (missing keys count as 0), y_i = 1 iff option i
//     is gold. Range [0, 2] for k ≥ 2 options; 0 is perfect. Computed only
//     when the adapter supplies a FULL probability vector — a single
//     chosen-class probability cannot reconstruct the other options.
//
//   Expected Calibration Error (Guo et al. 2017, "On Calibration of Modern
//   Deep Neural Networks", eq. 3, M = 10):
//     ECE = Σ_b (n_b / N) · |acc(b) − conf(b)|
//     over 10 equal-width confidence bins [0,.1) … [.9,1.0] on the
//     probability of the selected answer; acc(b) = mean correctness and
//     conf(b) = mean confidence of the samples in bin b; empty bins
//     contribute 0. Range [0, 1].
export type DecisionScored = {
  correct: boolean;
  score: 0 | 1; // 1 = exact match (RunItem.score for this track)
  prob: number | null; // probability of the selected option, when reported
  brier: number | null; // Σ_i (p_i − y_i)² when a full prob vector exists
};

export type DecisionScorerOptions = {
  /** the item's option list (question.options) — bounds the Brier sum */
  options?: readonly string[] | null;
  /** adapter-reported probability of the selected option */
  prob?: number | null;
  /** adapter-reported full option → probability vector */
  probs?: Record<string, number> | null;
};

/** Accept the shapes a run can carry: a bare label string, null, or a
 *  {choice, prob?, probs?} object. The v0 rule never normalizes the label. */
export function normalizeDecisionChoice(
  prediction: unknown,
  opts: DecisionScorerOptions = {},
): { choice: string | null; prob: number | null; probs: Record<string, number> | null } {
  let choice: string | null = null;
  let prob = opts.prob ?? null;
  let probs = opts.probs ?? null;
  if (typeof prediction === "string") {
    choice = prediction;
  } else if (typeof prediction === "object" && prediction !== null) {
    const obj = prediction as { choice?: unknown; prob?: unknown; probs?: unknown };
    if (typeof obj.choice === "string") choice = obj.choice;
    if (prob === null && typeof obj.prob === "number") prob = obj.prob;
    if (probs === null && typeof obj.probs === "object" && obj.probs !== null) {
      probs = obj.probs as Record<string, number>;
    }
  }
  // A probability without a selected option is meaningless for this track.
  return { choice, prob: choice === null ? null : prob, probs };
}

/** Score one decision item. `gold` is the gold label string; the comparison
 *  is `===` exactly as the v0 bench did. */
export function scoreDecision(
  prediction: unknown,
  gold: unknown,
  opts: DecisionScorerOptions = {},
): DecisionScored {
  const { choice, prob, probs } = normalizeDecisionChoice(prediction, opts);
  const correct = choice !== null && choice === gold;

  let brier: number | null = null;
  if (probs !== null) {
    // Option order: the item's option list when present (keys outside it are
    // ignored), else the sorted prob keys for determinism.
    const keys =
      opts.options && opts.options.length > 0 ? [...opts.options] : Object.keys(probs).sort();
    let sum = 0;
    let usable = keys.length > 0;
    for (const key of keys) {
      const p = probs[key] ?? 0;
      if (!Number.isFinite(p)) {
        usable = false;
        break;
      }
      sum += (p - (gold === key ? 1 : 0)) ** 2;
    }
    if (usable) brier = sum;
  }

  return { correct, score: correct ? 1 : 0, prob, brier };
}

// ---- calibration (decision secondary metric) ------------------------------

export type CalibrationSample = {
  /** confidence = probability of the selected answer, in [0, 1] */
  prob: number;
  correct: boolean;
};

export type ReliabilityBin = {
  lower: number; // inclusive
  upper: number; // exclusive, except the last bin which includes 1.0
  n: number;
  confidence: number | null; // mean prob, null when the bin is empty
  accuracy: number | null; // mean correct, null when the bin is empty
};

/** Equal-width reliability diagram: sample i lands in bin
 *  min(B−1, floor(prob_i · B)) (values outside [0, 1] clamp into range). */
export function reliabilityBins(
  samples: readonly CalibrationSample[],
  bins = 10,
): ReliabilityBin[] {
  const acc = new Array<number>(bins).fill(0);
  const conf = new Array<number>(bins).fill(0);
  const n = new Array<number>(bins).fill(0);
  for (const s of samples) {
    if (!Number.isFinite(s.prob)) continue;
    const idx = Math.min(bins - 1, Math.max(0, Math.floor(s.prob * bins)));
    n[idx] = (n[idx] ?? 0) + 1;
    acc[idx] = (acc[idx] ?? 0) + (s.correct ? 1 : 0);
    conf[idx] = (conf[idx] ?? 0) + Math.min(1, Math.max(0, s.prob));
  }
  const out: ReliabilityBin[] = [];
  for (let b = 0; b < bins; b++) {
    const count = n[b] ?? 0;
    out.push({
      lower: b / bins,
      upper: (b + 1) / bins,
      n: count,
      confidence: count > 0 ? (conf[b] ?? 0) / count : null,
      accuracy: count > 0 ? (acc[b] ?? 0) / count : null,
    });
  }
  return out;
}

/** ECE over 10 equal-width confidence bins (Guo et al. 2017, eq. 3).
 *  Returns null when no sample carries a usable probability. */
export function expectedCalibrationError(
  samples: readonly CalibrationSample[],
  bins = 10,
): number | null {
  const usable = samples.filter((s) => Number.isFinite(s.prob));
  if (usable.length === 0) return null;
  let ece = 0;
  for (const bin of reliabilityBins(usable, bins)) {
    if (bin.n > 0 && bin.accuracy !== null && bin.confidence !== null) {
      ece += (bin.n / usable.length) * Math.abs(bin.accuracy - bin.confidence);
    }
  }
  return ece;
}
