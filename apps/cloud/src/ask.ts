// Client-side hosted ask: the same adapter call the old playground's server
// made (and the bench CLI makes), but in the browser with the USER'S OWN key
// instead of server env vars. Keys live in React state only — they ride the
// adapter call straight to the provider and are never stored, logged, or put
// in a URL. The page is static, so there is no server to receive anything.
//
// bench has no package "exports" field yet, so the adapter imports ride
// relative paths into bench/src (same trick the old playground used); switch
// to "@zsib/bench/..." once bench declares subpath exports. Both modules are
// browser-safe: plain fetch, no node:/fs/process/Bun imports, and their
// @zsib/schema imports are type-only (erased at build).
import type { TrackRequestOut, TrackResponseOut } from "@zsib/schema";
import {
  type AdapterContext,
  AdapterError,
  getAdapter,
} from "../../../bench/src/providers/index.ts";
import { endpointIsUrl } from "../../../bench/src/providers/wire.ts";
import type { HostedDeployment, ProviderRow } from "./types";

export type AskOutcome = {
  deployment: string;
  adapter: string;
  model: string;
  prediction: unknown;
  prob: number | null;
  error: string | null;
  latencyMs: number;
  computeMs: number | null;
};

export class AskError extends Error {
  /** fetch TypeError — most likely the provider sent no CORS headers. */
  corsSuspected: boolean;
  constructor(message: string, opts: { corsSuspected?: boolean } = {}) {
    super(message);
    this.name = "AskError";
    this.corsSuspected = opts.corsSuspected ?? false;
  }
}

const CORS_HINT =
  "this provider may not allow direct browser calls (CORS) — it may work through the local app or CLI";

export async function askHosted(args: {
  deployment: HostedDeployment;
  providers: Array<ProviderRow>;
  /** user keys keyed by the provider's env var NAMES (from provider.auth) */
  keys: Record<string, string>;
  state: string;
  question: { type: "choice"; options: Array<string> };
}): Promise<AskOutcome> {
  const { deployment, providers, keys, state, question } = args;
  const provider = providers.find((p) => p.id === deployment.provider);
  if (!provider) throw new AskError(`unknown provider "${deployment.provider}"`);

  const missing = (provider.auth ?? []).filter((name) => !(keys[name] ?? "").trim());
  if (missing.length > 0) throw new AskError(`missing key(s): ${missing.join(", ")}`);

  // same AdapterContext the bench CLI builds server-side — except auth comes
  // from the form, keyed by the catalog provider's env var names
  const ctx: AdapterContext = {
    baseUrl: provider.base_url ?? "",
    auth: Object.fromEntries(
      (provider.auth ?? []).map((name) => [name, (keys[name] ?? "").trim()]),
    ),
    endpoint: deployment.endpoint,
    // same rule as bench/src/cli.ts: a URL endpoint selects by the pinned
    // api_slug, a slug endpoint IS the wire model
    model: endpointIsUrl(deployment.endpoint) ? (deployment.api_slug ?? undefined) : undefined,
  };
  const adapter = getAdapter(deployment.adapter);
  const req: TrackRequestOut = {
    model_id: deployment.endpoint,
    items: [{ item_id: "cloud", input: { text: state }, question }],
  };

  let response: TrackResponseOut;
  let latencyMs: number;
  try {
    const t0 = performance.now();
    response = await adapter.send(req, ctx);
    latencyMs = Math.round(performance.now() - t0);
  } catch (err) {
    if (err instanceof AdapterError) throw new AskError(err.message);
    // fetch() network failure surfaces as TypeError — in a browser that is
    // almost always a CORS block (or the provider being unreachable)
    if (err instanceof TypeError) {
      throw new AskError(`${err.message} — ${CORS_HINT}`, { corsSuspected: true });
    }
    throw new AskError(err instanceof Error ? err.message : String(err));
  }

  const result = response.results.find((r) => r.item_id === "cloud");
  return {
    deployment: deployment.deployment_id,
    adapter: deployment.adapter,
    model: ctx.model ?? ctx.endpoint,
    prediction: result?.prediction ?? null,
    prob: result?.prob ?? null,
    error: result?.error ?? null,
    latencyMs,
    computeMs: response.compute_ms || null,
  };
}
