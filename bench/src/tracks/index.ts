// Track registry — one entry per track id, pairing the track's scorer with
// its suites namespace. The run loop (bench CLI) is generic over this shape:
// for each suite item it calls track.scorer({prediction, prob, probs}, item)
// and writes the ScoredItem fields onto the run item (docs/plan/04).
import type { SuiteItemOut } from "@zsib/schema";
import type { CalibrationSample, DecisionScored, ReliabilityBin } from "./decision/scorer.ts";
import { expectedCalibrationError, reliabilityBins, scoreDecision } from "./decision/scorer.ts";
import type { NerAggregate, NerItemScore, Prf } from "./ner/scorer.ts";
import { aggregateNer, prf, scoreNer } from "./ner/scorer.ts";
import type { LoadedSuite, SuiteFile, SuitesNamespace } from "./suite.ts";
import { computeSuiteHash, listSuites, loadSuite, suitesFor } from "./suite.ts";

export type {
  CalibrationSample,
  DecisionScored,
  LoadedSuite,
  NerAggregate,
  NerItemScore,
  Prf,
  ReliabilityBin,
  SuiteFile,
  SuitesNamespace,
};
// value re-exports (one import for consumers)
export {
  aggregateNer,
  computeSuiteHash,
  expectedCalibrationError,
  listSuites,
  loadSuite,
  prf,
  reliabilityBins,
  scoreDecision,
  scoreNer,
  suitesFor,
};

/** What the adapter/runner reported for one item. */
export type ScoreInput = {
  /** track-typed prediction (decision: the chosen label string or
   *  {choice, prob?, probs?}; ner: [[start,end,label], …]) */
  prediction: unknown;
  /** probability of the selected answer, when the adapter reports one */
  prob?: number | null;
  /** full option → probability vector, when the adapter reports one */
  probs?: Record<string, number> | null;
};

/** Per-item scorer output — maps onto the RunItem fields (docs/plan/04). */
export type ScoredItem = {
  correct: boolean; // RunItem.correct (strict/exact v0 rule)
  score: number; // RunItem.score (decision 0|1; ner 0|1 exact match)
  prob: number | null; // RunItem.prob
  /** named secondary metrics (brier, span f1 parts, …) for the run item */
  metrics: Record<string, number>;
  /** ner only: per-label strict counts for corpus aggregation */
  per_label?: Record<string, Prf>;
};

export type ScorerFn = (input: ScoreInput, item: SuiteItemOut) => ScoredItem;

export type TrackDef = {
  id: string;
  scorer: ScorerFn;
  suites: SuitesNamespace;
};

const decisionScorer: ScorerFn = (input, item) => {
  const r: DecisionScored = scoreDecision(input.prediction, item.gold, {
    options: item.question.options,
    prob: input.prob ?? null,
    probs: input.probs ?? null,
  });
  const metrics: Record<string, number> = {};
  if (r.brier !== null) metrics.brier = r.brier;
  return {
    correct: r.correct,
    score: r.score,
    prob: r.prob,
    metrics,
  };
};

const nerScorer: ScorerFn = (input, item) => {
  const r: NerItemScore = scoreNer(input.prediction, item.gold);
  return {
    correct: r.correct,
    score: r.score,
    prob: null,
    metrics: {
      strict_tp: r.spans.strict.tp,
      strict_fp: r.spans.strict.fp,
      strict_fn: r.spans.strict.fn,
      strict_f1: r.spans.strict.f1,
      partial_tp: r.spans.partial.tp,
      partial_fp: r.spans.partial.fp,
      partial_fn: r.spans.partial.fn,
      partial_f1: r.spans.partial.f1,
    },
    per_label: r.per_label,
  };
};

export const tracks: Record<string, TrackDef> = {
  decision: { id: "decision", scorer: decisionScorer, suites: suitesFor("decision") },
  ner: { id: "ner", scorer: nerScorer, suites: suitesFor("ner") },
};

export function getTrack(id: string): TrackDef {
  const track = tracks[id];
  if (!track) {
    throw new Error(`unknown track "${id}" (known: ${Object.keys(tracks).sort().join(", ")})`);
  }
  return track;
}

/** Score one item through the registry. */
export function runScorer(trackId: string, input: ScoreInput, item: SuiteItemOut): ScoredItem {
  return getTrack(trackId).scorer(input, item);
}

/** The run loop's handle (bench/src/cli.ts): resolve a track's scorer and
 *  score one item with just the adapter prediction — `correct` is the track's
 *  strict/exact rule, `score` its scalar. */
export function getScorer(trackId: string): {
  track: TrackDef;
  scoreItem: (item: SuiteItemOut, prediction: unknown) => { correct: boolean; score: number };
} {
  const track = getTrack(trackId);
  return {
    track,
    scoreItem: (item, prediction) => {
      const scored = track.scorer({ prediction }, item);
      return { correct: scored.correct, score: scored.score };
    },
  };
}
