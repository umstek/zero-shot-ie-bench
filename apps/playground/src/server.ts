// apps/playground — the hosted side of the bench as a try-it UI
// (docs/plan/02 "Hosted side"): Bun.serve + React via HTML imports, importing
// the same provider adapters the bench CLI drives, so a provider is wired
// once. Keys stay server-side (src/keys.ts, process.env only); the browser
// never sees one. Live local runners are listed and proxied here too, so one
// UI tries both hosted deployments and a runner island.
//
//   GET  /                  the React page (Bun HTML import — no bundler)
//   GET  /api/deployments   hosted deployments from the catalog + key NAMES
//   POST /api/ask           {deploymentId?, model?, state, question} → adapter
//   GET  /api/runners       runner islands from the catalog + ?baseUrl/?runner probe
//   POST /api/runner-ask    {baseUrl, model, state, question} → runner /v1/decision
//
// Run: `bun run dev` (from apps/playground) after `bun install` at the root.
import { loadCatalog } from "@zsib/lake";
import type { TrackRequestOut } from "@zsib/schema";
// bench has no package "exports" field yet, so the same-file imports the
// architecture doc asks for ("imports the same adapters from bench") ride
// relative paths into bench/src; switch to "@zsib/bench/..." once bench
// declares subpath exports.
import {
  fetchInfo,
  RunnerUnavailableError,
  sendTrack,
  type TrackResponseOut,
} from "../../../bench/src/local/runner-client.ts";
import {
  type AdapterContext,
  AdapterError,
  getAdapter,
} from "../../../bench/src/providers/index.ts";
import { endpointIsUrl } from "../../../bench/src/providers/wire.ts";
import { authFor, expectedKeyNames, missingKeyNames } from "./keys.ts";
import ui from "./ui/index.html";

const PORT = Number(process.env.PORT ?? 4010);
const PROBE_TIMEOUT_MS = 2500;
const MAX_STATE_CHARS = 20_000;
const MAX_OPTIONS = 64;

type Question = { type: "choice"; options: Array<string> };

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

function errorStatus(err: unknown): number {
  if (err instanceof AdapterError) return err.status ?? 502;
  if (err instanceof RunnerUnavailableError) return 502;
  return 500;
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

// ---- request parsing -------------------------------------------------------

class BadRequestError extends Error {}

function parseAskBody(body: unknown): {
  deploymentId: string | null;
  model: string | null;
  state: string;
  question: Question;
} {
  if (typeof body !== "object" || body === null)
    throw new BadRequestError("body must be a JSON object");
  const b = body as Record<string, unknown>;
  const state = typeof b.state === "string" ? b.state.trim() : "";
  if (state.length === 0) throw new BadRequestError("state is required (free text)");
  if (state.length > MAX_STATE_CHARS) {
    throw new BadRequestError(`state longer than ${MAX_STATE_CHARS} chars`);
  }
  const q = b.question as Record<string, unknown> | undefined;
  const options = Array.isArray(q?.options) ? q.options : [];
  const cleaned = options.filter((o): o is string => typeof o === "string" && o.trim().length > 0);
  if (q?.type !== "choice" || cleaned.length < 2) {
    throw new BadRequestError('question must be {type: "choice", options: [at least two strings]}');
  }
  if (cleaned.length > MAX_OPTIONS) throw new BadRequestError(`at most ${MAX_OPTIONS} options`);
  const model = typeof b.model === "string" && b.model.trim().length > 0 ? b.model.trim() : null;
  const deploymentId =
    typeof b.deploymentId === "string" && b.deploymentId.trim().length > 0
      ? b.deploymentId.trim()
      : null;
  return { deploymentId, model, state, question: { type: "choice", options: cleaned } };
}

function parseRunnerBody(body: unknown): {
  baseUrl: string;
  model: string;
  state: string;
  question: Question;
} {
  const parsed = parseAskBody(body);
  if (typeof (body as Record<string, unknown>).baseUrl !== "string") {
    throw new BadRequestError("baseUrl is required (e.g. http://127.0.0.1:8030)");
  }
  const url = new URL((body as Record<string, unknown>).baseUrl as string);
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new BadRequestError("baseUrl must be http(s)");
  }
  const model = parsed.model;
  if (!model) throw new BadRequestError("model is required (the runner's model_id, see /info)");
  return { baseUrl: url.origin, model, state: parsed.state, question: parsed.question };
}

// ---- catalog resolution ----------------------------------------------------

function findDeployment(catalog: ReturnType<typeof loadCatalog>, deploymentId: string) {
  for (const model of catalog.models) {
    const deployment = model.deployments.find((d) => d.id === deploymentId);
    if (deployment) return { model, deployment };
  }
  return null;
}

function apiSlugOf(
  model: ReturnType<typeof loadCatalog>["models"][number],
  versionId: string,
): string | undefined {
  return model.versions.find((v) => v.id === versionId)?.api_slug ?? undefined;
}

// ---- GET /api/deployments --------------------------------------------------

async function listDeployments(): Promise<Response> {
  const catalog = loadCatalog();
  const deployments = catalog.models
    .flatMap((model) =>
      model.deployments.flatMap((d) => {
        if (d.kind !== "hosted") return [];
        const provider = catalog.providers.find((p) => p.id === d.provider);
        const missing = provider ? missingKeyNames(provider) : [];
        return [
          {
            id: d.id,
            modelId: model.id,
            family: model.family,
            display: d.display ?? d.id,
            provider: d.provider,
            adapter: d.adapter,
            endpoint: d.endpoint,
            version: d.version,
            zdrSafe: d.zdr_safe ?? null,
            // env var NAMES only — the values never leave the server
            keysRequired: provider ? expectedKeyNames(provider) : [],
            keysMissing: missing,
            keysConfigured: missing.length === 0,
          },
        ];
      }),
    )
    .sort((a, b) => a.display.localeCompare(b.display));
  return json({ deployments });
}

// ---- POST /api/ask ---------------------------------------------------------

async function askHosted(body: unknown): Promise<Response> {
  const { deploymentId, model: modelOverride, state, question } = parseAskBody(body);
  const catalog = loadCatalog();

  let found: ReturnType<typeof findDeployment> = null;
  if (deploymentId) {
    found = findDeployment(catalog, deploymentId);
    if (!found) throw new BadRequestError(`unknown deployment "${deploymentId}"`);
  } else {
    // no deployment picked: default to the first hosted one whose keys are all
    // configured, so the page works before the picker is touched
    for (const model of catalog.models) {
      for (const d of model.deployments) {
        if (d.kind !== "hosted") continue;
        const provider = catalog.providers.find((p) => p.id === d.provider);
        if (provider && missingKeyNames(provider).length === 0) {
          found = { model, deployment: d };
          break;
        }
      }
      if (found) break;
    }
    if (!found) {
      throw new BadRequestError(
        "no hosted deployment has its provider keys in the server environment — export them (see README) or pass a deploymentId",
      );
    }
  }

  const { model, deployment } = found;
  if (deployment.kind !== "hosted")
    throw new BadRequestError(`"${deployment.id}" is not a hosted deployment`);
  const provider = catalog.providers.find((p) => p.id === deployment.provider);
  if (!provider) throw new BadRequestError(`unknown provider "${deployment.provider}"`);

  // secrets come from process.env here on the server; the client only ever
  // sees the response below
  const ctx: AdapterContext = {
    baseUrl: provider.base_url ?? "",
    auth: authFor(provider),
    endpoint: deployment.endpoint,
    // same rule as bench/src/cli.ts: a URL endpoint selects by the pinned
    // api_slug, a slug endpoint IS the wire model — the request `model`
    // overrides both for playground experiments
    model:
      modelOverride ??
      (endpointIsUrl(deployment.endpoint)
        ? (apiSlugOf(model, deployment.version) ?? undefined)
        : undefined),
  };
  const adapter = getAdapter(deployment.adapter);
  const req: TrackRequestOut = {
    model_id: deployment.endpoint,
    items: [{ item_id: "playground", input: { text: state }, question }],
  };

  const t0 = performance.now();
  const response = await adapter.send(req, ctx);
  const latencyMs = Math.round(performance.now() - t0);
  const result = response.results.find(
    (r: TrackResponseOut["results"][number]) => r.item_id === "playground",
  );
  return json({
    deployment: deployment.id,
    adapter: deployment.adapter,
    model: ctx.model ?? ctx.endpoint,
    prediction: result?.prediction ?? null,
    prob: result?.prob ?? null,
    error: result?.error ?? null,
    latencyMs,
    computeMs: response.compute_ms || null,
    usage: response.usage ?? null,
  });
}

// ---- GET /api/runners ------------------------------------------------------

async function listRunners(req: Request): Promise<Response> {
  const catalog = loadCatalog();
  const url = new URL(req.url);

  type IslandModel = { id: string; display: string; versions: Array<string> };
  const islands = new Map<
    string,
    { runner: string; envVar: string; baseUrl: string | null; models: Array<IslandModel> }
  >();
  for (const model of catalog.models) {
    for (const d of model.deployments) {
      if (d.kind !== "local") continue;
      // same naming the bench CLI uses: ZSIB_RUNNER_<ISLAND> (dashes → underscores)
      const envVar = `ZSIB_RUNNER_${d.runner.replaceAll("-", "_").toUpperCase()}`;
      let island = islands.get(d.runner);
      if (!island) {
        island = {
          runner: d.runner,
          envVar,
          baseUrl: process.env[envVar] ?? null,
          models: [],
        };
        islands.set(d.runner, island);
      }
      let entry = island.models.find((m) => m.id === model.id);
      if (!entry) {
        entry = { id: model.id, display: d.display ?? model.id, versions: [] };
        island.models.push(entry);
      }
      if (!entry.versions.includes(d.version)) entry.versions.push(d.version);
    }
  }
  const islandList = [...islands.values()].sort((a, b) => a.runner.localeCompare(b.runner));

  // probe option: ?baseUrl=<url> probes that runner; ?runner=<island> probes
  // the island's env-configured URL. /info with a short timeout; reachable
  // models reported back so the UI can prefill the form.
  const probeTarget =
    url.searchParams.get("baseUrl") ??
    (() => {
      const runner = url.searchParams.get("runner");
      if (!runner) return null;
      return islands.get(runner)?.baseUrl ?? null;
    })();

  let probe: unknown = null;
  if (probeTarget) {
    try {
      const parsed = new URL(probeTarget);
      if (parsed.protocol !== "http:" && parsed.protocol !== "https:")
        throw new Error("not http(s)");
      const info = await fetchInfo({ baseUrl: parsed.origin, timeoutMs: PROBE_TIMEOUT_MS });
      probe = {
        baseUrl: parsed.origin,
        reachable: true,
        runner: info.runner,
        protocol: info.protocol,
        models: info.models.map((m) => ({ modelId: m.model_id, version: m.version })),
        host: {
          hostname: info.host.hostname,
          cpu: info.host.cpu,
          ramGb: info.host.ram_gb,
          gpus: info.host.gpus,
          os: info.host.os,
        },
      };
    } catch (err) {
      probe = { baseUrl: probeTarget, reachable: false, error: errorMessage(err) };
    }
  }
  return json({ islands: islandList, probe });
}

// ---- POST /api/runner-ask --------------------------------------------------

async function askRunner(body: unknown): Promise<Response> {
  const { baseUrl, model, state, question } = parseRunnerBody(body);
  // the server does the call — the browser never talks to the runner directly
  // (no CORS involved), and latency is measured around the client call like
  // the bench does (the runner's own compute_ms rides the response)
  const { response, latencyMs } = await sendTrack({ baseUrl }, "decision", {
    model_id: model,
    items: [{ item_id: "playground", input: { text: state }, question }],
  });
  const result = response.results.find((r) => r.item_id === "playground");
  return json({
    baseUrl,
    model,
    prediction: result?.prediction ?? null,
    prob: result?.prob ?? null,
    error: result?.error ?? null,
    latencyMs: Math.round(latencyMs),
    computeMs: response.compute_ms || null,
    usage: response.usage ?? null,
  });
}

// ---- server ----------------------------------------------------------------

Bun.serve({
  port: PORT,
  routes: {
    "/": ui,
  },
  async fetch(req) {
    const { pathname } = new URL(req.url);
    try {
      if (req.method === "GET" && pathname === "/api/deployments") return await listDeployments();
      if (req.method === "GET" && pathname === "/api/runners") return await listRunners(req);
      if (req.method === "POST" && pathname === "/api/ask")
        return await askHosted(await req.json());
      if (req.method === "POST" && pathname === "/api/runner-ask") {
        return await askRunner(await req.json());
      }
      return json({ error: `no route: ${req.method} ${pathname}` }, 404);
    } catch (err) {
      if (err instanceof BadRequestError) return json({ error: err.message }, 400);
      if (err instanceof SyntaxError)
        return json({ error: `invalid JSON body: ${err.message}` }, 400);
      return json({ error: errorMessage(err) }, errorStatus(err));
    }
  },
});

console.log(`zsib playground → http://localhost:${PORT}`);
console.log("keys are read from process.env on this server only; the browser never sees them");
