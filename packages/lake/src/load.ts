// Filesystem loaders: catalog + runs → typed objects. Validates every file
// against @zsib/schema and fails loudly on drift (docs/plan/02 invariant 2).
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import {
  Language,
  type LanguageOut,
  Machine,
  type MachineOut,
  Model,
  type ModelOut,
  Provider,
  type ProviderOut,
  RunItem,
  type RunItemOut,
  RunMeta,
  type RunMetaOut,
  Track,
  type TrackOut,
} from "@zsib/schema";
import { parse as parseYaml } from "yaml";
import { z } from "zod";

const repoRoot = join(import.meta.dir, "..", "..", "..");

// walk only the data roots we read (catalog/, runs/) — never legacy/ or venvs
function walk(rootDir: string, out: string[] = []): string[] {
  const dir = join(repoRoot, rootDir);
  if (!existsSync(dir)) return out; // runs/ may not exist yet
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    const rel = relative(repoRoot, full).replaceAll("\\", "/");
    if (statSync(full).isDirectory()) {
      if (entry === "node_modules" || entry === ".git" || entry === "build") continue;
      walk(rel, out);
    } else {
      out.push(rel);
    }
  }
  return out;
}

function parse<T extends z.ZodType>(schema: T, path: string, raw: string): z.output<T> {
  const doc = path.endsWith(".json") ? JSON.parse(raw) : parseYaml(raw);
  const result = schema.safeParse(doc);
  if (!result.success) {
    const issues = result.error.issues
      .slice(0, 3)
      .map((i) => `${i.path.join(".") || "(root)"}: ${i.message}`)
      .join("; ");
    throw new Error(`${path}: ${issues}`);
  }
  return result.data;
}

export type Catalog = {
  // parsed shapes of the catalog entities
  models: ModelOut[];
  providers: ProviderOut[];
  machines: MachineOut[];
  tracks: TrackOut[];
  languages: LanguageOut[];
};

export function loadCatalog(): Catalog {
  const files = walk("catalog");
  const models = files
    .filter((p) => p.startsWith("catalog/models/") && p.endsWith(".yaml"))
    .map((p) => parse(Model, p, readFileSync(join(repoRoot, p), "utf8")));
  const providers = files
    .filter((p) => p.startsWith("catalog/providers/") && p.endsWith(".yaml"))
    .map((p) => parse(Provider, p, readFileSync(join(repoRoot, p), "utf8")));
  const machines = files
    .filter((p) => p.startsWith("catalog/machines/") && p.endsWith(".yaml"))
    .map((p) => parse(Machine, p, readFileSync(join(repoRoot, p), "utf8")));
  const tracks = parse(
    z.array(Track),
    "catalog/tracks.yaml",
    readFileSync(join(repoRoot, "catalog/tracks.yaml"), "utf8"),
  );
  const languages = parse(
    z.array(Language),
    "catalog/languages.yaml",
    readFileSync(join(repoRoot, "catalog/languages.yaml"), "utf8"),
  );
  return { models, providers, machines, tracks, languages };
}

export type LoadedRun = { meta: RunMetaOut; items: RunItemOut[] };

export function loadRuns(): LoadedRun[] {
  const files = walk("runs");
  const metas = files.filter((p) => p.endsWith(".meta.json"));
  return metas.map((metaPath) => {
    const meta = parse(RunMeta, metaPath, readFileSync(join(repoRoot, metaPath), "utf8"));
    const itemsPath = metaPath.replace(/\.meta\.json$/, ".items.jsonl");
    const itemsRaw = readFileSync(join(repoRoot, itemsPath), "utf8");
    const items = itemsRaw
      .split("\n")
      .filter((l) => l.trim().length > 0)
      .map((l, idx) => {
        const doc = JSON.parse(l);
        const result = RunItem.safeParse(doc);
        if (!result.success) {
          throw new Error(`${itemsPath} [line ${idx + 1}]: ${result.error.issues[0]?.message}`);
        }
        return result.data;
      });
    return { meta, items };
  });
}
