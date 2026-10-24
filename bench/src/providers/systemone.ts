// The TypeSafe "System One" wire shared by the hosted decision adapters
// (typesafe.systemone, fastino.systemone, workers-ai.clef) —
// engines/jev_client.py is the contract: POST {model, state, questions}
// with Bearer auth; answer rows come back keyed by question name as
//   {choice: {pick, probabilities}} | {noul: number} | {score: number}
// Cloudflare's /ai/run wraps the payload in the REST v4 envelope
// ({success, result: {model, answers, usage}}), so the unwrap tolerates
// both flat and wrapped responses (jev_client.ask unwraps `result` when
// `answers` is not top-level).
import type { TrackRequestOut, TrackResponseOut } from "@zsib/schema";
import { type AdapterContext, AdapterError } from "./types.ts";
import { bodyModel, inputText, mergeUsage, postJson, safeJson } from "./wire.ts";

type TrackItem = TrackRequestOut["items"][number];

type ResultRow = TrackResponseOut["results"][number];

type MappedAnswer = Pick<ResultRow, "prediction" | "prob" | "error">;

/** The bench's track-typed question → the System One question row.
 *  jev_client.choice() is the contract: {type: "choice", instructions,
 *  criteria: {option: description|null}} — bare suite options carry null
 *  descriptions. Anything else forwards untouched so noul/score questions
 *  keep their native shape. */
export function systemoneQuestion(item: TrackItem): unknown {
  const q = item.question as
    | { type?: string; options?: string[]; instructions?: string }
    | undefined;
  if (q?.type === "choice" && q.options) {
    const criteria: Record<string, string | null> = {};
    for (const option of q.options) criteria[option] = null;
    return {
      type: "choice",
      instructions: q.instructions ?? "Choose the best option.",
      criteria,
    };
  }
  return item.question;
}

type AnswerRow = {
  choice?: string | { pick?: string; probabilities?: number[] };
  probabilities?: number[];
  noul?: unknown;
  score?: unknown;
};

function mapAnswerRow(answer: unknown): MappedAnswer {
  const row = answer as AnswerRow | undefined;
  if (typeof row?.choice === "string") {
    // the v0 readers (jev_client/bench) read answers[name].choice directly
    // as the picked option string
    return {
      prediction: row.choice,
      prob: pickProb(row.probabilities),
      error: null,
    };
  }
  if (row && typeof row.choice === "object" && row.choice !== null) {
    const pick = typeof row.choice.pick === "string" ? row.choice.pick : null;
    const probs = row.choice.probabilities ?? row.probabilities;
    return {
      prediction: pick,
      prob: pickProb(probs),
      error: pick === null ? "choice answer without pick" : null,
    };
  }
  if (typeof row?.noul === "number") {
    // noul answers ARE a probability (openrouter_client.noul_classify
    // reads answers[name].noul as the yes-probability)
    return { prediction: row.noul, prob: row.noul, error: null };
  }
  if (typeof row?.score === "number") {
    return { prediction: row.score, prob: null, error: null };
  }
  return {
    prediction: null,
    prob: null,
    error: answer === undefined ? "no answer for item" : "unrecognized answer row",
  };
}

function pickProb(probs: number[] | undefined): number | null {
  return Array.isArray(probs) && probs.length > 0 ? Math.max(...probs) : null;
}

/** Cloudflare's REST v4 failure envelope can ride HTTP 200
 *  ({"success": false, "errors": [...]}); a flat payload has no `success`
 *  field and passes through. */
function unwrapSystemone(payload: unknown, provider: string): Record<string, unknown> {
  if (typeof payload !== "object" || payload === null) {
    throw new AdapterError(
      `${provider} response is not an object: ${safeJson(payload)}`,
      null,
      null,
    );
  }
  const p = payload as Record<string, unknown>;
  if (p.success === false) {
    throw new AdapterError(
      `${provider} envelope failure: ${safeJson(p.errors ?? p.messages ?? p)}`,
      null,
      null,
    );
  }
  if (!("answers" in p) && p.result && typeof p.result === "object") {
    return p.result as Record<string, unknown>;
  }
  return p;
}

/** Send one track request over the System One wire. The request carries a
 *  single `state`, so items are grouped by input text — the bench runs one
 *  item per request through these adapters, and shared-text batches stay a
 *  single call (jev_client.classify's batching embeds the text in the
 *  question instructions; the bench keeps the text in the state). */
export async function sendSystemone(
  req: TrackRequestOut,
  ctx: AdapterContext,
  opts: { url: string; bearer: string | undefined; provider: string },
): Promise<TrackResponseOut> {
  if (req.items.length === 0) throw new AdapterError("empty request", 400, null);
  const groups = new Map<string, TrackRequestOut["items"]>();
  for (const item of req.items) {
    const state = inputText(item);
    const group = groups.get(state);
    if (group) group.push(item);
    else groups.set(state, [item]);
  }
  const answers = new Map<string, unknown>();
  const usageRows: unknown[] = [];
  for (const [state, items] of groups) {
    const questions: Record<string, unknown> = {};
    for (const item of items) questions[item.item_id] = systemoneQuestion(item);
    const payload = await postJson(
      opts.url,
      { model: ctx.model ?? bodyModel(req, ctx), state, questions },
      ctx,
      opts.provider,
      opts.bearer,
    );
    const body = unwrapSystemone(payload, opts.provider);
    const rows = body.answers;
    if (typeof rows !== "object" || rows === null) {
      throw new AdapterError(
        `${opts.provider} response missing answers: ${safeJson(body)}`,
        null,
        null,
      );
    }
    for (const [name, row] of Object.entries(rows)) answers.set(name, row);
    usageRows.push(body.usage);
  }
  return {
    model_id: req.model_id,
    compute_ms: 0, // hosted: measured client-side by the bench, not here
    usage: usageRows.length === 1 ? (usageRows[0] ?? null) : mergeUsage(usageRows),
    results: req.items.map((item) => {
      const mapped = mapAnswerRow(answers.get(item.item_id));
      return { item_id: item.item_id, ...mapped };
    }),
  };
}
