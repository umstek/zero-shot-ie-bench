// typellm.generate — TypeLLM's /v1/generate: JSON-Schema-constrained
// decoding over SGLang (engines/typellm_client.py). A choice question
// becomes {type: "object", properties: {answer: {type: "string", enum:
// options}}} — one constrained pick per question; the response is flat
// {result: {<name>: {answer}}, usage: {input_tokens, thinking_tokens}}.
// The request carries ONE context, so items are grouped by input text
// (the bench runs one item per request). return_probabilities stays off —
// the benches take the bare pick — so prob is null.
//
// Retry gap (flagged): typellm_client.generate retries a 429, a 5xx other
// than 504 and connection errors up to twice; here the adapter reports
// them as AdapterError (status + Retry-After) and the bench owns the
// retry policy — a 504 must NOT be retried, the others should.
import type { TrackRequestOut, TrackResponseOut } from "@zsib/schema";
import { type Adapter, type AdapterContext, AdapterError } from "./types.ts";
import { bodyModel, inputText, mergeUsage, postJson, safeJson } from "./wire.ts";

// typellm_client.MAX_ENUM_VALUES (v0.6.9)
const MAX_ENUM_VALUES = 512;

type ResultRow = TrackResponseOut["results"][number];

export const generate: Adapter = {
  id: "typellm.generate",
  async send(req: TrackRequestOut, ctx: AdapterContext): Promise<TrackResponseOut> {
    if (req.items.length === 0) throw new AdapterError("empty request", 400, null);
    const results = new Map<string, ResultRow>();
    const usageRows: unknown[] = [];
    // enum ceilings: 2-512 values per constrained pick (typellm_client
    // raises before sending; here the item records the error instead)
    const groups = new Map<string, TrackRequestOut["items"]>();
    for (const item of req.items) {
      const q = item.question as { type?: string; options?: string[] } | undefined;
      const options = q?.type === "choice" ? q.options : undefined;
      if (!options || options.length < 2 || options.length > MAX_ENUM_VALUES) {
        results.set(item.item_id, {
          item_id: item.item_id,
          prediction: null,
          prob: null,
          error: options
            ? `typellm.generate needs 2-${MAX_ENUM_VALUES} choice options, got ${options.length}`
            : `typellm.generate supports choice questions only, got ${safeJson(q?.type ?? null)}`,
        });
        continue;
      }
      const context = inputText(item);
      const group = groups.get(context);
      if (group) group.push(item);
      else groups.set(context, [item]);
    }
    for (const [context, items] of groups) {
      const questions: Record<string, unknown> = {};
      const images: unknown[] = [];
      for (const item of items) {
        questions[item.item_id] = {
          type: "object",
          properties: {
            answer: {
              type: "string",
              enum: (item.question as { options: string[] }).options,
            },
          },
        };
        const rows = (item.input as { images?: unknown[] } | undefined)?.images;
        if (Array.isArray(rows)) images.push(...rows);
      }
      const body: Record<string, unknown> = {
        context,
        questions,
        model: bodyModel(req, ctx),
        timeout: 60, // typellm_client's default; the request field maxes at 90
      };
      if (images.length > 0) body.images = images; // base64 data URIs, max 8/call
      const payload = (await postJson(
        `${ctx.baseUrl}/v1/generate`,
        body,
        ctx,
        "typellm",
        ctx.auth.TYPELLM_API_KEY,
      )) as Record<string, unknown>;
      usageRows.push(payload.usage);
      const result = payload.result;
      if (typeof result !== "object" || result === null) {
        throw new AdapterError(
          `typellm response result is not an object: ${safeJson(payload)}`,
          null,
          null,
        );
      }
      const rows = result as Record<string, unknown>;
      for (const item of items) {
        // the object schema pins the pick under `answer`; tolerate the
        // bare-string value the native enum question shape returns
        const node = rows[item.item_id];
        const pick =
          typeof node === "string" ? node : (node as { answer?: unknown } | undefined)?.answer;
        results.set(item.item_id, {
          item_id: item.item_id,
          prediction: typeof pick === "string" ? pick : null,
          prob: null,
          error: typeof pick === "string" ? null : "no answer for item",
        });
      }
    }
    return {
      model_id: req.model_id,
      compute_ms: 0, // hosted: measured client-side by the bench, not here
      usage: mergeUsage(usageRows),
      results: req.items.map(
        (item) =>
          results.get(item.item_id) ?? {
            item_id: item.item_id,
            prediction: null,
            prob: null,
            error: "unanswered item",
          },
      ),
    };
  },
};
