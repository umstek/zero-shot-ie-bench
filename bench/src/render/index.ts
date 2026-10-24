// bench render — regenerates the numeric README tables from the lake
// (build/*.runs.json + build/deployments.json) between <!-- gen:… --> markers.
// Hand-written prose tables (system table, feature comparison, family maps)
// stay untouched. Phase-2 gate: scripts/acceptance-phase2.ts diffs these
// cell-for-cell against the v0-final README.
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { RunAggregate } from "@zsib/lake";
import { parse as parseYaml } from "yaml";
import type { DeploymentInfo } from "./formatters.ts";
import { costTable, mixedPoolTable, multilingualTable } from "./formatters.ts";

const repoRoot = join(import.meta.dir, "..", "..", "..");

const readJson = (p: string) => JSON.parse(readFileSync(join(repoRoot, p), "utf8"));
const loadRuns = (track: string): RunAggregate[] => {
  try {
    return readJson(`build/${track}.runs.json`);
  } catch {
    return []; // no runs for this track yet
  }
};

const deployments: DeploymentInfo[] = readJson("build/deployments.json");
const decisionRuns = loadRuns("decision");
const nerRuns = loadRuns("ner");

const languages = parseYaml(
  readFileSync(join(repoRoot, "catalog/languages.yaml"), "utf8"),
) as Array<{ code: string; tier: string | null }>;
const tierOf = Object.fromEntries(languages.map((l) => [l.code, l.tier ?? "popular"]));

const providerFiles = readdirSync(join(repoRoot, "catalog/providers")).filter((f) =>
  f.endsWith(".yaml"),
);
const zdrSafeOf: Record<string, boolean | null> = {};
for (const f of providerFiles) {
  const provider = parseYaml(readFileSync(join(repoRoot, "catalog/providers", f), "utf8")) as {
    id: string;
    zdr_safe: boolean | null;
  };
  zdrSafeOf[provider.id] = provider.zdr_safe;
}
for (const dep of deployments) {
  // v0 † marker is per deployment (endpoint), with the provider flag as a
  // fallback for anything not flagged explicitly
  dep.dagger =
    dep.zdr_safe === false ||
    (dep.zdr_safe == null && dep.provider ? zdrSafeOf[dep.provider] === false : false);
}

const readme = readFileSync(join(repoRoot, "README.md"), "utf8");

function replaceSection(readmeText: string, name: string, body: string): string {
  const open = `<!-- gen:${name} -->`;
  const close = `<!-- /gen:${name} -->`;
  const start = readmeText.indexOf(open);
  const end = readmeText.indexOf(close);
  if (start === -1 || end === -1) {
    throw new Error(`README markers for "${name}" not found`);
  }
  return `${readmeText.slice(0, start)}${open}\n${body}\n${close}${readmeText.slice(end + close.length)}`;
}

let out = readme;
out = replaceSection(out, "mixed-pool", mixedPoolTable(decisionRuns, nerRuns, deployments));
out = replaceSection(out, "multilingual", multilingualTable(decisionRuns, deployments, tierOf));
out = replaceSection(out, "cost", costTable(decisionRuns, deployments));
writeFileSync(join(repoRoot, "README.md"), out);
console.log("bench render: mixed-pool, multilingual, cost sections updated");
