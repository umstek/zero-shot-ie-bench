// bench render — regenerates the numeric README tables from the lake
// (build/*.runs.json) between <!-- gen:… --> markers. Hand-written prose
// tables (system table, feature comparison, family maps) stay untouched.
// Phase-2 acceptance: scripts/acceptance-phase2.ts diffs these cell-for-cell
// against the v0-final README.
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { renderTable } from "./markdown.ts";
import type { RunAggregate } from "@zsib/lake";

const repoRoot = join(import.meta.dir, "..", "..", "..");

function loadRuns(track: string): RunAggregate[] {
  const raw = readFileSync(join(repoRoot, "build", `${track}.runs.json`), "utf8");
  return JSON.parse(raw) as RunAggregate[];
}

function replaceSection(readme: string, name: string, body: string): string {
  const open = `<!-- gen:${name} -->`;
  const close = `<!-- /gen:${name} -->`;
  const start = readme.indexOf(open);
  const end = readme.indexOf(close);
  if (start === -1 || end === -1) {
    throw new Error(`README markers for "${name}" not found`);
  }
  return `${readme.slice(0, start)}${open}\n${body}\n${close}${readme.slice(end + close.length)}`;
}

// TODO(phase 2): formatters below are wired once legacy runs land in runs/;
// they reproduce the v0 table cell formats exactly (accuracy %, s/question,
// n/a, bold leaders, † non-ZDR markers) — scripts/acceptance-phase2.ts gates.
function mixedPoolTable(_runs: RunAggregate[]): string {
  return renderTable(["System", "Classification", "s/question", "NER exact", "s/question"], []);
}

const readme = readFileSync(join(repoRoot, "README.md"), "utf8");
let out = readme;
out = replaceSection(out, "mixed-pool", mixedPoolTable(loadRuns("decision")));
writeFileSync(join(repoRoot, "README.md"), out);
console.log("bench render: README generated sections updated");
