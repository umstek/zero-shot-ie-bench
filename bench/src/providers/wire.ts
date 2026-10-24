// Shared wire plumbing for the hosted adapters: one JSON POST helper that
// turns transport/HTTP failures into AdapterError (status + Retry-After in
// both RFC 9110 forms), plus small mappings every adapter needs. The
// legacy engines each grew their own retry ladders (fastino's
// 425/429/503 wait table, typellm's 429/5xx backoff, openrouter's 429
// sleeps); here the adapter only reports the failure with its Retry-After
// and the bench layer owns the retry policy.
import type { TrackRequestOut } from "@zsib/schema";
import { type AdapterContext, AdapterError } from "./types.ts";

type TrackItem = TrackRequestOut["items"][number];

/** Retry-After header as milliseconds — RFC 9110 delay-seconds or
 *  HTTP-date (typellm_client._retry_after_seconds honors both) — or null
 *  when absent/unparseable. No clamp: how long to actually wait is the
 *  bench's policy, not the wire's. */
export function retryAfterMs(res: Response): number | null {
  const raw = res.headers.get("retry-after");
  if (!raw) return null;
  const seconds = Number(raw);
  if (Number.isFinite(seconds)) return Math.max(seconds, 0) * 1000;
  const when = Date.parse(raw);
  return Number.isNaN(when) ? null : Math.max(when - Date.now(), 0);
}

/** JSON.stringify clipped for error messages (payloads can be huge). */
export function safeJson(value: unknown): string {
  return JSON.stringify(value)?.slice(0, 300) ?? "unserializable";
}

export async function postJson(
  url: string,
  body: unknown,
  ctx: AdapterContext,
  provider: string,
  bearer?: string,
): Promise<unknown> {
  const doFetch = ctx.fetchImpl ?? fetch;
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (bearer) headers.authorization = `Bearer ${bearer}`;
  let res: Response;
  try {
    res = await doFetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });
  } catch (err) {
    throw new AdapterError(
      `${provider} transport error: ${err instanceof Error ? err.message : String(err)}`,
      null,
      null,
    );
  }
  const json: unknown = await res.json().catch(() => null);
  if (!res.ok) {
    throw new AdapterError(
      `${provider} HTTP ${res.status}: ${safeJson(json)}`,
      res.status,
      retryAfterMs(res),
    );
  }
  return json;
}

/** The item's text input (suite inputs are {text, images}). */
export function inputText(item: TrackItem): string {
  return (item.input as { text?: string } | undefined)?.text ?? "";
}

/** Deployment endpoints carry the api_slug (fastino/GLiDE,
 *  @cf/cloudflare/clef-flash, typellm-latest — exactly what the legacy
 *  clients sent as the body `model`); when the endpoint is an absolute
 *  URL instead (typesafe's /v1/systemone), the catalog model id is the
 *  only selector left. */
export function endpointIsUrl(endpoint: string): boolean {
  return /^https?:\/\//i.test(endpoint);
}

export function bodyModel(req: TrackRequestOut, ctx: AdapterContext): string {
  return endpointIsUrl(ctx.endpoint) ? req.model_id : ctx.endpoint;
}

/** Sum the numeric fields of per-request usage rows (request-per-item
 *  adapters accumulate; the bench normalizes the row per run item).
 *  Null when nothing numeric was reported. */
export function mergeUsage(rows: unknown[]): Record<string, number> | null {
  const out: Record<string, number> = {};
  for (const row of rows) {
    if (typeof row !== "object" || row === null) continue;
    for (const [key, value] of Object.entries(row)) {
      if (typeof value === "number" && Number.isFinite(value)) {
        out[key] = (out[key] ?? 0) + value;
      }
    }
  }
  return Object.keys(out).length > 0 ? out : null;
}
