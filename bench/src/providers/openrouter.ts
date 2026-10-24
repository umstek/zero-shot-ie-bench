// openrouter adapters. The systemone contract rides OpenRouter's
// /api/v1/systemone router (same {model, state, questions} wire as TypeSafe,
// per legacy/engines/jev_client.py + openrouter_client.py).
import type { TrackRequestOut, TrackResponseOut } from "@zsib/schema";
import { type Adapter, type AdapterContext, AdapterError } from "./types.ts";
import { mergeUsage } from "./wire.ts";

async function postJson(
  url: string,
  body: unknown,
  ctx: AdapterContext,
): Promise<{ status: number; json: unknown; retryAfterMs: number | null }> {
  const doFetch = ctx.fetchImpl ?? fetch;
  const headers: Record<string, string> = { "content-type": "application/json" };
  const bearer = ctx.auth.OPENROUTER_API_KEY;
  if (bearer) headers.authorization = `Bearer ${bearer}`;
  const res = await doFetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  const retryAfterMs = res.headers.get("retry-after")
    ? Number(res.headers.get("retry-after")) * 1000
    : null;
  const json = await res.json().catch(() => null);
  if (!res.ok) {
    throw new AdapterError(
      `openrouter HTTP ${res.status}: ${JSON.stringify(json).slice(0, 300)}`,
      res.status,
      retryAfterMs,
    );
  }
  return { status: res.status, json, retryAfterMs };
}

/** {model, state, questions} → {model, answers, usage}; answers keyed by
 *  question id, matching the v0 JevClient wire (engines/jev_client.py). */
export const systemone: Adapter = {
  id: "openrouter.systemone",
  async send(req: TrackRequestOut, ctx: AdapterContext): Promise<TrackResponseOut> {
    const questions: Record<string, unknown> = {};
    const states: Record<string, string> = {};
    for (const item of req.items) {
      states[item.item_id] = (item.input as { text?: string } | undefined)?.text ?? "";
      const q = item.question as { type?: string; options?: string[] } | undefined;
      if (q?.type === "choice" && q.options) {
        questions[item.item_id] = { choice: { options: q.options } };
      } else {
        questions[item.item_id] = item.question;
      }
    }
    // the systemone router takes ONE state per request; bench runs one item
    // per request through this adapter unless items share a state
    const first = req.items[0];
    if (!first) throw new AdapterError("empty request", 400, null);
    const { json } = await postJson(
      `${ctx.baseUrl}/systemone`,
      { model: req.model_id, state: states[first.item_id], questions },
      ctx,
    );
    const payload = json as {
      answers?: Record<string, { choice?: { pick?: string; probabilities?: number[] } }>;
      usage?: unknown;
    };
    return {
      model_id: req.model_id,
      compute_ms: 0, // hosted: measured client-side by the bench, not here
      usage: payload.usage ?? null,
      results: req.items.map((item) => {
        const answer = payload.answers?.[item.item_id];
        const pick = answer?.choice?.pick ?? null;
        const probs = answer?.choice?.probabilities ?? null;
        return {
          item_id: item.item_id,
          prediction: pick,
          prob: probs ? Math.max(...probs) : null,
          error: pick === null && !answer ? "no answer for item" : null,
        };
      }),
    };
  },
};

/** Softmax over the returned relevance scores, read at `at` — the rerank
 *  endpoint reports bare relevance scores, not calibrated probabilities,
 *  so the confidence is this argmax's share of the exp mass (rows the
 *  endpoint omitted are left out of the denominator). */
function softmaxAt(scores: Map<number, number>, at: number): number | null {
  const target = scores.get(at);
  if (target === undefined) return null;
  let max = Number.NEGATIVE_INFINITY;
  for (const score of scores.values()) max = Math.max(max, score);
  let sum = 0;
  for (const score of scores.values()) sum += Math.exp(score - max);
  return Math.exp(target - max) / sum;
}

/** openrouter.rerank — the Cohere-style {model, query, documents} contract
 *  used as a decision engine (engines/openrouter_client.py
 *  rerank_classify): a choice item scores one (query, label) pair per
 *  option — query is the input text, documents the options — and the
 *  highest relevance_score wins. One request per item (each has its own
 *  query); top_n stays unset, like rerank_classify. */
export const rerank: Adapter = {
  id: "openrouter.rerank",
  async send(req: TrackRequestOut, ctx: AdapterContext): Promise<TrackResponseOut> {
    if (req.items.length === 0) throw new AdapterError("empty request", 400, null);
    const results: TrackResponseOut["results"] = [];
    const usageRows: unknown[] = [];
    for (const item of req.items) {
      const q = item.question as { type?: string; options?: string[] } | undefined;
      const options = q?.type === "choice" ? q.options : undefined;
      if (!options) {
        results.push({
          item_id: item.item_id,
          prediction: null,
          prob: null,
          error: "openrouter.rerank needs a choice question",
        });
        continue;
      }
      const text = (item.input as { text?: string } | undefined)?.text ?? "";
      const { json } = await postJson(
        `${ctx.baseUrl}/rerank`,
        { model: ctx.endpoint || req.model_id, query: text, documents: options },
        ctx,
      );
      const payload = json as {
        results?: Array<{ index?: unknown; relevance_score?: unknown }>;
        usage?: unknown;
      };
      usageRows.push(payload.usage);
      const scores = new Map<number, number>();
      for (const row of Array.isArray(payload.results) ? payload.results : []) {
        if (typeof row?.index === "number" && typeof row.relevance_score === "number") {
          scores.set(row.index, row.relevance_score);
        }
      }
      if (scores.size === 0) {
        results.push({
          item_id: item.item_id,
          prediction: null,
          prob: null,
          error: "rerank response missing results",
        });
        continue;
      }
      let best = -1;
      let bestScore = Number.NEGATIVE_INFINITY;
      for (const [index, score] of scores) {
        if (score > bestScore) {
          best = index;
          bestScore = score;
        }
      }
      results.push({
        item_id: item.item_id,
        prediction: options[best] ?? null,
        prob: softmaxAt(scores, best),
        error: null,
      });
    }
    return {
      model_id: req.model_id,
      compute_ms: 0, // hosted: measured client-side by the bench, not here
      usage: mergeUsage(usageRows),
      results,
    };
  },
};
