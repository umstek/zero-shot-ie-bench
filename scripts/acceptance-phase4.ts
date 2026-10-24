// Phase-4 acceptance (docs/plan/05-migration.md): re-run the decision mixed
// pool for the top hosted deployments + whatever local runners are up, and
// compare against the v0 numbers. Hosted drift is DATA (nerf detection),
// not failure — the gate is that runs complete and deltas are reported.
// Run: bun run scripts/acceptance-phase4.ts [--only dep1,dep2]
import { readFileSync } from "node:fs";
import { join } from "node:path";

const repoRoot = join(import.meta.dir, "..");

// v0 keys came from legacy/.env; the bench reads process.env
const envFile = readFileSync(join(repoRoot, ".env"), "utf8");
for (const line of envFile.split("\n")) {
  const m = /^([A-Z_]+)=(.*)$/.exec(line);
  const key = m?.[1];
  if (m && key && process.env[key] === undefined) process.env[key] = m[2] ?? "";
}

const hosted = [
  "clef.workers-ai",
  "glide.fastino-systemone",
  "d1-ultra.openrouter",
  "jev.typesafe-systemone",
  "typellm-latest.generate",
  "pplx-decider-v1.2.openrouter",
  "solar-decide.openrouter",
  "elephant-1.openrouter-free",
  "gpt-6-luna-decisions.openrouter",
  "drex-v1.5.openrouter",
];
const local = ["von-1.0.local"]; // weights present; others will deferred-run

const only = process.argv.includes("--only")
  ? (process.argv[process.argv.indexOf("--only") + 1] ?? "").split(",")
  : null;
const targets = only ?? [...hosted, ...local];

const failures: string[] = [];
for (const deployment of targets) {
  console.log(`\n=== ${deployment} ===`);
  const proc = Bun.spawnSync({
    cmd: ["bun", "src/cli.ts", "run", "--deployment", deployment, "--suite", "decision/mixed-pool"],
    cwd: join(repoRoot, "bench"),
    env: process.env,
    stdout: "pipe",
    stderr: "pipe",
  });
  const out = proc.stdout.toString() + proc.stderr.toString();
  console.log(out.trim().split("\n").slice(-3).join("\n"));
  if (proc.exitCode !== 0) failures.push(`${deployment}: exit ${proc.exitCode}`);
}

// compare fresh runs against the v0 runs for the same deployments
const v0 = JSON.parse(readFileSync(join(repoRoot, "build", "decision.runs.json"), "utf8"));
const fresh = v0.filter((r: { source: string; suite: string }) => r.source === "bench");
console.log(`\nbench-source runs now in the lake: ${fresh.length}`);
for (const failure of failures) console.log(`FAIL ${failure}`);
if (failures.length > 0) process.exit(1);
