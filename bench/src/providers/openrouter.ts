// openrouter adapters. The systemone contract rides OpenRouter's
// /api/v1/systemone router (same {model, state, questions} wire as TypeSafe,
// per legacy/engines/jev_client.py + openrouter_client.py).
import type { TrackRequestOut, TrackResponseOut } from "@zsib/schema";
import { type Adapter, type AdapterContext, AdapterError } from "./types.ts";

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
