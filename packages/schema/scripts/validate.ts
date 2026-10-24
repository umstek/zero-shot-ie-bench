// Validates every catalog / suite / run data file against catalog/schema/.
// `bun run validate` from the root. Zero-tolerance: exits 1 on the first
// pass with failures, printing each offending file and its issues.
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { parse as parseYaml } from "yaml";
import { z } from "zod";
import * as catalog from "../src/catalog.ts";
import * as run from "../src/run.ts";
import * as suite from "../src/suite.ts";

const root = join(import.meta.dir, "..", "..", "..");

// rule per data file; schema validated against the Zod type directly (the
// JSON Schema files in catalog/schema/ are generated from the same types)
type Rule = { schema: z.ZodType; parse: (raw: string) => unknown[] };
const yamlDoc: Rule["parse"] = (raw) => [parseYaml(raw)];
const jsonDoc: Rule["parse"] = (raw) => [JSON.parse(raw)];
const jsonlLines: Rule["parse"] = (raw) =>
  raw
    .split("\n")
    .filter((l) => l.trim().length > 0)
    .map((l) => JSON.parse(l));

const rules: { glob: (p: string) => Rule | null }[] = [
  {
    glob: (p) =>
      p.startsWith("catalog/models/") && p.endsWith(".yaml")
        ? { schema: catalog.Model, parse: yamlDoc }
        : null,
  },
  {
    glob: (p) =>
      p.startsWith("catalog/providers/") && p.endsWith(".yaml")
        ? { schema: catalog.Provider, parse: yamlDoc }
        : null,
  },
  {
    glob: (p) =>
      p.startsWith("catalog/machines/") && p.endsWith(".yaml")
        ? { schema: catalog.Machine, parse: yamlDoc }
        : null,
  },
  {
    glob: (p) =>
      p === "catalog/tracks.yaml" ? { schema: z.array(catalog.Track), parse: yamlDoc } : null,
  },
  {
    glob: (p) =>
      p === "catalog/languages.yaml" ? { schema: z.array(catalog.Language), parse: yamlDoc } : null,
  },
  {
    glob: (p) =>
      p.startsWith("suites/") && p.endsWith("suite.yaml")
        ? { schema: suite.SuiteMeta, parse: yamlDoc }
        : null,
  },
  {
    glob: (p) =>
      p.startsWith("suites/") && p.endsWith(".jsonl")
        ? { schema: suite.SuiteItem, parse: jsonlLines }
        : null,
  },
  {
    glob: (p) =>
      p.startsWith("runs/") && p.endsWith(".meta.json")
        ? { schema: run.RunMeta, parse: jsonDoc }
        : null,
  },
  {
    glob: (p) =>
      p.startsWith("runs/") && p.endsWith(".items.jsonl")
        ? { schema: run.RunItem, parse: jsonlLines }
        : null,
  },
  // runner protocol shapes are HTTP payloads, not repo files — validated by
  // bench/runner tests against runner-info/track-request/track-response
  { glob: () => null },
];

function walk(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) {
      if (entry === "node_modules" || entry === ".git" || entry === "build") continue;
      walk(full, out);
    } else {
      out.push(relative(root, full).replaceAll("\\", "/"));
    }
  }
  return out;
}

let checked = 0;
const failures: string[] = [];
for (const p of walk(root)) {
  const rule = rules.map((r) => r.glob(p)).find((r): r is Rule => r !== null);
  if (!rule) continue;
  checked++;
  // rule.parse returns the UNITS to validate: whole documents for .yaml/.json
  // (an array-schema file like tracks.yaml yields its whole array as one
  // unit), individual lines for .jsonl
  let docs: unknown[];
  try {
    docs = rule.parse(readFileSync(join(root, p), "utf8"));
  } catch (err) {
    failures.push(`${p}: parse error — ${err instanceof Error ? err.message : err}`);
    continue;
  }
  for (const [i, doc] of docs.entries()) {
    const result = rule.schema.safeParse(doc);
    if (!result.success) {
      const where = docs.length > 1 ? ` [entry ${i}]` : "";
      const issues = result.error.issues
        .slice(0, 5)
        .map((iss) => `${iss.path.join(".") || "(root)"}: ${iss.message}`)
        .join("; ");
      failures.push(`${p}${where}: ${issues}`);
    }
  }
}

for (const f of failures) console.error(`FAIL ${f}`);
console.log(`validated ${checked} data files, ${failures.length} failures`);
if (failures.length > 0) process.exit(1);
