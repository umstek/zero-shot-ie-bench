// Lake build: catalog + runs → build/*.json per track (docs/plan/02 "Lake").
// Plain TypeScript aggregation; build/ is gitignored board/render input.
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { aggregateRun } from "./aggregate.ts";
import { loadCatalog, loadRuns } from "./load.ts";

const repoRoot = join(import.meta.dir, "..", "..", "..");
const buildDir = join(repoRoot, "build");
mkdirSync(buildDir, { recursive: true });

const catalog = loadCatalog();
const runs = loadRuns();
console.log(
  `lake: ${catalog.models.length} models, ${runs.length} runs (${catalog.tracks.length} tracks)`,
);

// group runs by track (runs/<track>/…)
const byTrack = new Map<string, typeof runs>();
for (const run of runs) {
  const track = run.meta.suite.split("/")[0] ?? "unknown";
  const group = byTrack.get(track) ?? [];
  group.push(run);
  byTrack.set(track, group);
}

for (const [track, trackRuns] of byTrack) {
  const aggregates = trackRuns.map(({ meta, items }) => aggregateRun(meta, items));
  aggregates.sort((a, b) => a.started_at.localeCompare(b.started_at));
  writeFileSync(join(buildDir, `${track}.runs.json`), `${JSON.stringify(aggregates, null, 2)}\n`);

  const itemsDir = join(buildDir, `${track}.items`);
  mkdirSync(itemsDir, { recursive: true });
  for (const { meta, items } of trackRuns) {
    // item rows joined with the run's identity for drill-down views
    const rows = items.map((i) => ({
      run_id: meta.run_id,
      deployment: meta.deployment,
      model_version: meta.model_version,
      suite: meta.suite,
      ...i,
    }));
    writeFileSync(join(itemsDir, `${meta.run_id}.json`), `${JSON.stringify(rows, null, 2)}\n`);
  }
  console.log(`lake: build/${track}.runs.json (${aggregates.length} runs)`);
}

// difficulty spectrum: per-item correctness across the canonical
// decision/mixed-pool runs — the same latest-scored-run-per-deployment
// selection the board's selectors use — answering which questions are hard
// for everyone. An item only counts a system that actually answered it:
// adapter failures (401s, timeouts) say nothing about an item's difficulty.
{
  const canonical = new Map<
    string,
    { started_at: string; run_id: string; items: (typeof runs)[number]["items"] }
  >();
  for (const { meta, items } of runs) {
    if (meta.suite !== "decision/mixed-pool") continue;
    if (meta.status === "failed" || meta.status === "deferred") continue;
    if (aggregateRun(meta, items).accuracy == null) continue;
    const prev = canonical.get(meta.deployment);
    const newer =
      !prev ||
      meta.started_at > prev.started_at ||
      (meta.started_at === prev.started_at && meta.run_id > prev.run_id);
    if (newer)
      canonical.set(meta.deployment, { started_at: meta.started_at, run_id: meta.run_id, items });
  }

  const perItem = new Map<string, { lang: string; attempts: number; correct: number }>();
  for (const { items } of canonical.values()) {
    for (const item of items) {
      if (typeof item.correct !== "boolean" || item.error != null) continue;
      const agg = perItem.get(item.item_id) ?? {
        lang: item.lang ?? "unknown",
        attempts: 0,
        correct: 0,
      };
      agg.attempts += 1;
      if (item.correct) agg.correct += 1;
      perItem.set(item.item_id, agg);
    }
  }

  const spectrum = {
    suite: "decision/mixed-pool",
    systems: canonical.size,
    items: [...perItem.entries()]
      .map(([item_id, agg]) => ({
        item_id,
        lang: agg.lang,
        attempts: agg.attempts,
        correct: agg.correct,
        accuracy: agg.attempts > 0 ? agg.correct / agg.attempts : null,
      }))
      // hardest first: low accuracy at the top of the board chart
      .sort((a, b) => (a.accuracy ?? 1) - (b.accuracy ?? 1) || a.item_id.localeCompare(b.item_id)),
  };
  writeFileSync(join(buildDir, "spectrum.json"), `${JSON.stringify(spectrum, null, 2)}\n`);
  console.log(
    `lake: build/spectrum.json (${spectrum.items.length} items across ${spectrum.systems} runs)`,
  );
}

// deployments: catalog flattened + latest run per deployment + machine feasibility
const latestByDeployment = new Map<
  string,
  (typeof runs)[number]["meta"] & { accuracy: number | null }
>();
for (const { meta, items } of runs) {
  const prev = latestByDeployment.get(meta.deployment);
  if (!prev || prev.started_at < meta.started_at) {
    latestByDeployment.set(meta.deployment, {
      ...meta,
      accuracy: aggregateRun(meta, items).accuracy,
    });
  }
}

const deployments = catalog.models.flatMap((model) =>
  model.deployments.map((deployment) => {
    const latest = latestByDeployment.get(deployment.id);
    const machineFeasibility =
      deployment.kind === "local" && deployment.requires
        ? Object.fromEntries(
            catalog.machines.map((machine) => {
              const vramOk =
                deployment.requires?.vram_gb == null ||
                (machine.gpus.reduce((sum, g) => sum + g.vram_gb, 0) ?? 0) >=
                  deployment.requires.vram_gb;
              const ramOk =
                deployment.requires?.ram_gb == null || machine.ram_gb >= deployment.requires.ram_gb;
              return [machine.id, vramOk && ramOk];
            }),
          )
        : undefined;
    return {
      deployment_id: deployment.id,
      model_id: model.id,
      display: deployment.display,
      zdr_safe: deployment.zdr_safe,
      family: model.family,
      org: model.org,
      license: model.license,
      open_weights: model.open_weights,
      status: model.status,
      claims: model.claims,
      kind: deployment.kind,
      version: deployment.version,
      provider: deployment.kind === "hosted" ? deployment.provider : undefined,
      // hosted BYOK tryer (apps/cloud) drives the adapter in the browser and
      // needs the same fields the bench CLI resolves server-side
      adapter: deployment.kind === "hosted" ? deployment.adapter : undefined,
      endpoint: deployment.kind === "hosted" ? deployment.endpoint : undefined,
      api_slug:
        deployment.kind === "hosted"
          ? (model.versions.find((v) => v.id === deployment.version)?.api_slug ?? undefined)
          : undefined,
      runner: deployment.kind === "local" ? deployment.runner : undefined,
      requires: deployment.kind === "local" ? deployment.requires : undefined,
      feasible_on: machineFeasibility,
      latest_run: latest
        ? {
            run_id: latest.run_id,
            suite: latest.suite,
            started_at: latest.started_at,
            accuracy: latest.accuracy,
            cost_usd: latest.totals.cost_usd,
          }
        : undefined,
    };
  }),
);

writeFileSync(join(buildDir, "deployments.json"), `${JSON.stringify(deployments, null, 2)}\n`);
console.log(`lake: build/deployments.json (${deployments.length} deployments)`);

// Static, dated browser-CORS verdicts for the BYOK tryer (apps/cloud): its
// adapters run client-side, so a provider is only usable there if its API
// answers the CORS preflight with Access-Control-Allow-Origin. Measured by
// hand with a credential-free OPTIONS (no key sent, none printed):
//   curl -s -i -X OPTIONS "<endpoint>" \
//     -H "Origin: https://example.pages.dev" \
//     -H "Access-Control-Request-Method: POST" \
//     -H "Access-Control-Request-Headers: authorization, content-type"
// The endpoints are the real POST paths the bench adapters call
// (bench/src/providers/*), not the provider root. Remeasure when a provider
// changes its CORS posture and bump the dates.
export type BrowserCors = {
  /** whether the provider lets a browser call it (preflight answered) */
  ok: boolean;
  /** dated preflight measurement detail */
  note: string;
};

const BROWSER_CORS: Record<string, BrowserCors> = {
  openrouter: {
    ok: true,
    note:
      "preflight OPTIONS measured 2026-11-11 (POST /api/v1/systemone and /api/v1/rerank): " +
      "HTTP 204, Access-Control-Allow-Origin: * — works in browser",
  },
  typellm: {
    ok: false,
    note:
      "preflight OPTIONS measured 2026-11-11 (POST /v1/generate): HTTP 401 with no CORS " +
      "headers — the browser blocks the call",
  },
  fastino: {
    ok: false,
    note:
      "preflight OPTIONS measured 2026-11-11 (POST /v1/systemone and /v1/chat/completions): " +
      "HTTP 400 with allow-methods/allow-headers but no Access-Control-Allow-Origin " +
      "(origin-gated) — the preflight fails",
  },
  typesafe: {
    ok: false,
    note:
      "preflight OPTIONS measured 2026-11-11 (POST /v1/systemone): HTTP 400 with allow-" +
      "methods/allow-headers but no Access-Control-Allow-Origin (origin-gated) — the " +
      "preflight fails",
  },
  "cloudflare-workers-ai": {
    ok: false,
    note:
      "preflight OPTIONS measured 2026-11-11 (POST /accounts/<account>/ai/run/<model>): " +
      "HTTP 405 Method Not Allowed with no CORS headers — the browser blocks the call",
  },
};

// providers: what the hosted BYOK tryer (apps/cloud) needs to drive an
// adapter from the browser — base_url plus the env var NAMES each adapter
// reads. Names only, never values: the user's own keys are typed into the
// browser and stay there.
const providers = catalog.providers.map((provider) => ({
  id: provider.id,
  name: provider.name,
  base_url: provider.base_url,
  auth: provider.auth,
  // measured browser usability (see BROWSER_CORS); dropped from the JSON
  // until a provider has been measured
  browser: BROWSER_CORS[provider.id],
}));
writeFileSync(join(buildDir, "providers.json"), `${JSON.stringify(providers, null, 2)}\n`);
console.log(`lake: build/providers.json (${providers.length} providers)`);
