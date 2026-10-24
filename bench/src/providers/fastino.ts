// fastino adapters (engines/fastino_client.py). Two contracts behind one
// Bearer key:
// - fastino.systemone — GLiDE rides TypeSafe's System One wire (flat
//   payload, no envelope to unwrap) at {base}/systemone.
// - fastino.chat — the hosted GLiNER twins (extraction checkpoints and
//   GLiNER-2.5-Decide) ride the OpenAI-compatible chat-completions
//   endpoint, one conversation per request; the JSON string in
//   choices[0].message.content carries NER entities or a classification.
// Cold starts answer 425/429/503 with Retry-After; the adapter reports
// them as AdapterError and the bench retries (fastino_client's wait
// ladder is the bench's policy now).
import type { TrackRequestOut, TrackResponseOut } from "@zsib/schema";
import { sendSystemone } from "./systemone.ts";
import { type Adapter, type AdapterContext, AdapterError } from "./types.ts";
import { bodyModel, inputText, mergeUsage, postJson, safeJson } from "./wire.ts";

export const systemone: Adapter = {
  id: "fastino.systemone",
  send: (req: TrackRequestOut, ctx: AdapterContext): Promise<TrackResponseOut> =>
    sendSystemone(req, ctx, {
      url: `${ctx.baseUrl}/systemone`,
      bearer: ctx.auth.FASTINO_API_KEY,
      provider: "fastino",
    }),
};

// classify_text asks one task per request; the bench choice has no task
// name, so the schema/answer both key on this fixed one
const CHAT_TASK = "classification";

type ContentParse = { prediction: unknown; prob: number | null; error: string | null };

function contentNode(content: unknown): Record<string, unknown> {
  return typeof content === "object" && content !== null
    ? (content as Record<string, unknown>)
    : {};
}

/** {"<task>": {"label", "confidence"}} — tolerate the docs' wrapped
 *  {"classifications": {...}} form and the bare-label value
 *  (fastino_client.classify_text). */
function choiceFromContent(content: unknown, task: string): ContentParse {
  const node = contentNode(content);
  const wrapped =
    typeof node.classifications === "object" && node.classifications !== null
      ? (node.classifications as Record<string, unknown>)
      : null;
  const answer = wrapped ? wrapped[task] : node[task];
  if (typeof answer === "string") return { prediction: answer, prob: null, error: null };
  if (typeof answer === "object" && answer !== null) {
    const row = answer as { label?: unknown; confidence?: unknown };
    const label = typeof row.label === "string" ? row.label : null;
    return {
      prediction: label,
      prob: typeof row.confidence === "number" ? row.confidence : null,
      error: label === null ? "classification content without label" : null,
    };
  }
  return { prediction: null, prob: null, error: "classification content missing" };
}

/** {"entities": {label: [{start, end, text}]}} → sorted, deduped
 *  [start, end, label] char-span triples — the local gliner runner's NER
 *  prediction shape (gliner_relex.run_ner). Tolerates the unwrapped
 *  {label: [...]} form seen on single-label schemas; confidence rows are
 *  parsed but dropped. */
function spansFromContent(content: unknown): Array<[number, number, string]> {
  const node = contentNode(content);
  const ents =
    typeof node.entities === "object" && node.entities !== null
      ? (node.entities as Record<string, unknown>)
      : node;
  const seen = new Set<string>();
  const spans: Array<[number, number, string]> = [];
  for (const [label, items] of Object.entries(ents)) {
    if (!Array.isArray(items)) continue;
    for (const item of items) {
      if (typeof item !== "object" || item === null) continue;
      const row = item as { start?: unknown; end?: unknown };
      if (typeof row.start !== "number" || typeof row.end !== "number") continue;
      const key = `${row.start}\u0000${row.end}\u0000${label}`;
      if (seen.has(key)) continue;
      seen.add(key);
      spans.push([row.start, row.end, label]);
    }
  }
  spans.sort((a, b) => a[0] - b[0] || a[1] - b[1] || (a[2] < b[2] ? -1 : a[2] > b[2] ? 1 : 0));
  return spans;
}

function chatContent(payload: unknown): unknown {
  const p = payload as { choices?: Array<{ message?: { content?: unknown } }> } | null;
  const content = p?.choices?.[0]?.message?.content;
  if (typeof content !== "string") {
    throw new AdapterError(
      `fastino chat-completions response without a JSON content string: ${safeJson(payload)}`,
      null,
      null,
    );
  }
  try {
    return JSON.parse(content) as unknown;
  } catch {
    throw new AdapterError(
      `fastino chat-completions content is not JSON: ${content.slice(0, 300)}`,
      null,
      null,
    );
  }
}

/** AutoExtractor's default threshold (fastino_client._ask sends 0.5);
 *  an item's params.threshold overrides it, matching the local runner. */
function thresholdOf(item: TrackRequestOut["items"][number]): number {
  const t = (item.params as { threshold?: unknown } | undefined)?.threshold;
  return typeof t === "number" && t >= 0 && t <= 1 ? t : 0.5;
}

export const chat: Adapter = {
  id: "fastino.chat",
  async send(req: TrackRequestOut, ctx: AdapterContext): Promise<TrackResponseOut> {
    if (req.items.length === 0) throw new AdapterError("empty request", 400, null);
    const results: TrackResponseOut["results"] = [];
    const usageRows: unknown[] = [];
    for (const item of req.items) {
      const q = item.question as { type?: string; options?: string[] } | undefined;
      let schema: Record<string, unknown>;
      let parse: (content: unknown) => ContentParse;
      if (q?.type === "choice" && Array.isArray(q.options)) {
        schema = {
          classifications: [{ task: CHAT_TASK, labels: q.options, multi_label: false, top_k: 1 }],
        };
        parse = (content) => choiceFromContent(content, CHAT_TASK);
      } else if (q?.type === "ner-labels" && Array.isArray(q.options)) {
        schema = { entities: q.options.map((label) => ({ name: label, description: label })) };
        parse = (content) => ({ prediction: spansFromContent(content), prob: null, error: null });
      } else {
        results.push({
          item_id: item.item_id,
          prediction: null,
          prob: null,
          error: `fastino.chat does not support question type ${safeJson(q?.type ?? null)}`,
        });
        continue;
      }
      const payload = await postJson(
        `${ctx.baseUrl}/chat/completions`,
        {
          model: bodyModel(req, ctx),
          messages: [{ role: "user", content: inputText(item) }],
          schema,
          threshold: thresholdOf(item),
        },
        ctx,
        "fastino",
        ctx.auth.FASTINO_API_KEY,
      );
      usageRows.push((payload as { usage?: unknown }).usage);
      results.push({ item_id: item.item_id, ...parse(chatContent(payload)) });
    }
    return {
      model_id: req.model_id,
      compute_ms: 0, // hosted: measured client-side by the bench, not here
      usage: mergeUsage(usageRows),
      results,
    };
  },
};
