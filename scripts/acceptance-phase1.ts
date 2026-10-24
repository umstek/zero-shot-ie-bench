// Phase-1 acceptance (docs/plan/05-migration.md): every system in the v0-final
// README has a catalog file, and generated counts (systems / benched /
// families) match the README at v0-final. Run: `bun run scripts/acceptance-phase1.ts`
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { parse as parseYaml } from "yaml";

type ModelFile = {
  id: string;
  family: string;
  v0?: { benched?: boolean };
};

const modelsDir = join(import.meta.dir, "..", "catalog", "models");

function walkYaml(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) walkYaml(full, out);
    else if (entry.name.endsWith(".yaml")) out.push(full);
  }
  return out;
}

const files = walkYaml(modelsDir);
const models = files.map((f) => parseYaml(readFileSync(f, "utf8")) as ModelFile);

const systems = models.length;
const benched = models.filter((m) => m.v0?.benched === true).length;
const families = new Set(models.map((m) => m.family));

// the README at v0-final is the source of the expected numbers
const numberWords: Record<string, number> = {
  "eighty-one": 81,
  "seventy-eight": 78,
  "thirty-eight": 38,
};
const readme = await Bun.$`git show v0-final:README.md`.text();
const headline =
  readme.match(
    /(\w+-?\w*) zero-shot systems \((\w+-?\w*) of them benchmarked\) across\s+(\w+-?\w*) /i,
  ) ?? [];
const expected = headline
  .slice(1, 4)
  .map((w) => numberWords[w.toLowerCase()])
  .filter((n): n is number => typeof n === "number");

console.log(`catalog:      ${systems} system files, ${benched} benched, ${families.size} families`);
console.log(`v0-final README: ${expected.length === 3 ? expected.join(" / ") : "NOT PARSED"}`);
console.log(`families:     ${[...families].sort().join(", ")}`);

if (expected.length !== 3) {
  console.error("FAIL could not parse the v0-final README headline numbers");
  process.exit(1);
}
if (systems !== expected[0] || benched !== expected[1] || families.size !== expected[2]) {
  console.error("FAIL counts do not match the v0-final README");
  process.exit(1);
}
console.log("OK phase-1 acceptance counts match");
