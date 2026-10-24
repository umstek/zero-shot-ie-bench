// bench CLI — run | canary | doctor (docs/plan/05 phase 4).
// `bun src/cli.ts <command>` (or `bun run dev`).
import { loadCatalog } from "@zsib/lake";
import {
  fetchInfo,
  matchMachine,
  RunnerUnavailableError,
  sendTrack,
} from "./local/runner-client.ts";
import type { AdapterContext } from "./providers/index.ts";
import { AdapterError, getAdapter } from "./providers/index.ts";
import { type RunIdentity, RunWriter } from "./run/writer.ts";
import { getScorer } from "./tracks/index.ts";
import { loadSuite } from "./tracks/suite.ts";

const ENV_URL_PREFIX = "ZSIB_RUNNER_"; // ZSIB_RUNNER_GLINER=http://127.0.0.1:8000

function authFor(
  provider: string,
  env: Record<string, string | undefined>,
): Record<string, string> {
  const names: Record<string, string[]> = {
    openrouter: ["OPENROUTER_API_KEY"],
    typesafe: ["TYPESAFE_API_KEY"],
    fastino: ["FASTINO_API_KEY"],
    typellm: ["TYPELLM_API_KEY"],
    "cloudflare-workers-ai": ["CLOUDFLARE_AUTH_TOKEN", "CLOUDFLARE_ACCOUNT_ID"],
  };
  const out: Record<string, string> = {};
  for (const name of names[provider] ?? []) {
    const value = env[name];
    if (value) out[name] = value;
  }
  return out;
}

type Totals = {
  items: number;
  correct: number | null;
  cost_usd: number;
  p50_ms: number | null;
  p95_ms: number | null;
  errors: number;
};

function totalsOf(
  items: ReadonlyArray<{
    correct: boolean | null;
    latency_ms: number | null;
    cost_usd: number;
    error: string | null;
  }>,
): Totals {
  const scored = items.filter((i) => typeof i.correct === "boolean");
  const latencies = items
    .map((i) => i.latency_ms)
    .filter((v): v is number => typeof v === "number")
    .sort((a, b) => a - b);
  const pct = (p: number) =>
    latencies.length === 0
      ? null
      : (latencies[Math.min(Math.ceil((p / 100) * latencies.length), latencies.length) - 1] ??
        null);
  return {
    items: items.length,
    correct: scored.length > 0 ? scored.filter((i) => i.correct === true).length : null,
    cost_usd: items.reduce((s, i) => s + i.cost_usd, 0),
    p50_ms: pct(50),
    p95_ms: pct(95),
    errors: items.filter((i) => typeof i.error === "string").length,
  };
}

async function runDeployment(
  deploymentId: string,
  suiteId: string,
  opts: { root?: string },
): Promise<void> {
  const catalog = loadCatalog();
  const model = catalog.models.find((m) => m.deployments.some((d) => d.id === deploymentId));
  const deployment = model?.deployments.find((d) => d.id === deploymentId);
  if (!model || !deployment) throw new Error(`unknown deployment "${deploymentId}"`);
  const suite = await loadSuite(suiteId);
  const scorer = getScorer(suite.meta.track);
  const identity: RunIdentity = {
    deployment: deployment.id,
    model_version: deployment.version,
    suite: suite.meta.id,
    suite_version: suite.meta.version,
    suite_hash: suite.hash,
    params: {},
  };

  if (deployment.kind === "hosted") {
    const provider = catalog.providers.find((p) => p.id === deployment.provider);
    if (!provider) throw new Error(`unknown provider ${deployment.provider}`);
    const ctx: AdapterContext = {
      baseUrl: provider.base_url ?? "",
      auth: authFor(provider.id, process.env as Record<string, string | undefined>),
      endpoint: deployment.endpoint,
      model: model.versions.find((v) => v.id === deployment.version)?.api_slug ?? undefined,
    };
    const adapter = getAdapter(deployment.adapter);
    const writer = new RunWriter(identity, { kind: "hosted", provider: provider.id });
    for (const item of suite.items) {
      try {
        const t0 = performance.now();
        const response = await adapter.send(
          {
            model_id: deployment.endpoint,
            items: [{ item_id: item.item_id, input: item.input, question: item.question }],
          },
          ctx,
        );
        const latencyMs = performance.now() - t0;
        const result = response.results.find((r) => r.item_id === item.item_id);
        const prediction = result?.prediction ?? null;
        const scored = scorer.scoreItem(item, prediction);
        writer.push({
          item_id: item.item_id,
          lang: item.lang,
          prediction,
          gold: item.gold,
          correct: scored.correct,
          score: scored.score,
          prob: result?.prob ?? null,
          latency_ms: Math.round(latencyMs),
          compute_ms: response.compute_ms || null,
          cost_usd: 0,
          usage: null,
          error: result?.error ?? null,
        });
      } catch (err) {
        if (err instanceof AdapterError) {
          writer.push({
            item_id: item.item_id,
            lang: item.lang,
            prediction: null,
            gold: item.gold,
            correct: false,
            score: null,
            prob: null,
            latency_ms: null,
            compute_ms: null,
            cost_usd: 0,
            usage: null,
            error: `adapter: ${err.message}`,
          });
          continue;
        }
        throw err;
      }
    }
    const written = await writer.write({ status: "ok", totals: totalsOf(writer.snapshot()) });
    console.log(`run ${writer.runId} written (${written.metaPath})`);
    return;
  }

  // local: talk to the island over HTTP; feasibility first (R2)
  const island = deployment.runner.replace(/-/g, "_").toUpperCase();
  const baseUrl = process.env[`${ENV_URL_PREFIX}${island}`];
  if (!baseUrl) {
    const written = await RunWriter.writeDeferred(
      identity,
      { kind: "local", machine: "seoul", runtime: deployment.runtime },
      `no ${ENV_URL_PREFIX}${island} URL configured (start the runner: runners/${deployment.runner})`,
      opts.root,
    );
    console.log(`deferred run written (${written.metaPath})`);
    return;
  }
  const end = { baseUrl };
  let info: RunnerInfoOut | null = null;
  try {
    info = await fetchInfo(end);
  } catch (err) {
    if (err instanceof RunnerUnavailableError) {
      const written = await RunWriter.writeDeferred(
        identity,
        { kind: "local", machine: "seoul", runtime: deployment.runtime },
        `runner unreachable: ${err.message}`,
        opts.root,
      );
      console.log(`deferred run written (${written.metaPath})`);
      return;
    }
    throw err;
  }
  const machine = matchMachine(info, catalog.machines);
  if (!machine) {
    throw new Error(`runner host fingerprint ${info.host.hostname} matches no catalog machine`);
  }
  // resource gate: requirements vs machine (deferred with reason when short)
  const machineSpec = catalog.machines.find((m) => m.id === machine);
  const requires = deployment.requires;
  const vramTotal = machineSpec?.gpus.reduce((s, g) => s + g.vram_gb, 0) ?? 0;
  if (requires?.vram_gb != null && requires.vram_gb > vramTotal) {
    const written = await RunWriter.writeDeferred(
      identity,
      { kind: "local", machine, runtime: deployment.runtime },
      `requires vram_gb ${requires.vram_gb} > ${vramTotal} on ${machine}`,
      opts.root,
    );
    console.log(`deferred run written (${written.metaPath})`);
    return;
  }
  const served = info.models.find((m) => m.model_id === model.id);
  const writer = new RunWriter(identity, {
    kind: "local",
    machine,
    runtime: deployment.runtime,
  });
  for (const item of suite.items) {
    try {
      const { response, latencyMs } = await sendTrack(end, suite.meta.track, {
        model_id: model.id,
        items: [{ item_id: item.item_id, input: item.input, question: item.question }],
      });
      const result = response.results.find((r) => r.item_id === item.item_id);
      const prediction = result?.prediction ?? null;
      const scored = scorer.scoreItem(item, prediction);
      writer.push({
        item_id: item.item_id,
        lang: item.lang,
        prediction,
        gold: item.gold,
        correct: scored.correct,
        score: scored.score,
        prob: result?.prob ?? null,
        latency_ms: Math.round(latencyMs),
        compute_ms: response.compute_ms || null,
        cost_usd: 0,
        usage: null,
        error: result?.error ?? null,
      });
    } catch (err) {
      if (err instanceof RunnerUnavailableError) {
        writer.push({
          item_id: item.item_id,
          lang: item.lang,
          prediction: null,
          gold: item.gold,
          correct: false,
          score: null,
          prob: null,
          latency_ms: null,
          compute_ms: null,
          cost_usd: 0,
          usage: null,
          error: `runner: ${err.message}`,
        });
        continue;
      }
      throw err;
    }
  }
  const written = await writer.write({
    status: "ok",
    served_version: served?.version ?? null,
    totals: totalsOf(writer.snapshot()),
  });
  console.log(`run ${writer.runId} written on ${machine} (${written.metaPath})`);
}

async function doctor(): Promise<void> {
  const catalog = loadCatalog();
  console.log(
    `catalog: ${catalog.models.length} models, ${catalog.providers.length} providers, ${catalog.machines.length} machines`,
  );
  const adapters = new Set(
    catalog.models
      .flatMap((m) => m.deployments.map((d) => (d.kind === "hosted" ? d.adapter : null)))
      .filter(Boolean),
  );
  for (const id of adapters) {
    try {
      getAdapter(id as string);
      console.log(`adapter ok: ${id}`);
    } catch {
      console.log(`adapter MISSING: ${id}`);
    }
  }
  for (const machine of catalog.machines) {
    console.log(
      `machine ${machine.id}: ${machine.cpu}, cost/h ${machine.cost_per_hour_usd ?? "unset"}`,
    );
  }
}

const [command, ...args] = process.argv.slice(2);
const flags = new Map<string, string>();
for (let i = 0; i < args.length; i++) {
  if (args[i]?.startsWith("--")) flags.set(args[i]!.slice(2), args[i + 1] ?? "");
}

switch (command) {
  case "run": {
    const deployment = flags.get("deployment");
    const suite = flags.get("suite") ?? "decision/mixed-pool";
    if (!deployment) throw new Error("usage: bench run --deployment <id> [--suite <track/id>]");
    await runDeployment(deployment, suite, { root: flags.get("root") });
    break;
  }
  case "canary": {
    const deployment = flags.get("deployment");
    const track = flags.get("track") ?? "decision";
    if (!deployment) throw new Error("usage: bench canary --deployment <id> [--track decision]");
    await runDeployment(deployment, `${track}/canary`, { root: flags.get("root") });
    break;
  }
  case "doctor":
    await doctor();
    break;
  default:
    console.log(
      "zsib-bench — commands: run --deployment <id> --suite <id> | canary --deployment <id> | doctor",
    );
}
