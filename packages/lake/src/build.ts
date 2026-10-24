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
      family: model.family,
      org: model.org,
      license: model.license,
      open_weights: model.open_weights,
      status: model.status,
      claims: model.claims,
      kind: deployment.kind,
      version: deployment.version,
      provider: deployment.kind === "hosted" ? deployment.provider : undefined,
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
