// Suite loader — reads suites/<track>/<suite-id>/{suite.yaml,items/*.jsonl}
// and computes `suite_hash` byte-exactly like legacy/extract_suites.py
// (sha256 over the suite.yaml bytes, then each items/*.jsonl sorted by
// filename; every entry in items/ participates, not just *.jsonl).
// Docs: docs/plan/04-data-model.md (suite shape), packages/schema/src/suite.ts.
import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { SuiteItem, type SuiteItemOut, SuiteMeta, type SuiteMetaOut } from "@zsib/schema";
import { parse as parseYaml } from "yaml";

// <repo root>, resolved from this file so cwd never matters (same convention
// as bench/src/render/index.ts).
export const repoRoot = join(import.meta.dir, "..", "..", "..");
export const suitesRoot = join(repoRoot, "suites");

export type SuiteFile = {
  lang: string; // "<lang>.jsonl" stem
  path: string;
  items: SuiteItemOut[];
};

export type LoadedSuite = {
  id: string; // "<track>/<suite-id>", cross-checked against suite.yaml
  dir: string; // absolute suites/<track>/<suite-id>
  meta: SuiteMetaOut;
  suite_hash: string; // "sha256:…" (the run-meta field name, docs/plan/04)
  /** alias of suite_hash (bench/src/cli.ts reads suite.hash) */
  hash: string;
  items: SuiteItemOut[]; // all files concatenated in sorted-filename order
  files: SuiteFile[];
};

export function computeSuiteHash(dir: string): string {
  const hash = createHash("sha256");
  hash.update(readFileSync(join(dir, "suite.yaml")));
  const itemsDir = join(dir, "items");
  // Legacy sorts every entry of items/ (Python sorted(os.listdir)) and hashes
  // the raw bytes — no newline normalization, no extension filter.
  for (const name of readdirSync(itemsDir).sort()) {
    hash.update(readFileSync(join(itemsDir, name)));
  }
  return `sha256:${hash.digest("hex")}`;
}

/** Suite ids available for a track: subdirectories of suites/<track>/ that
 *  contain a suite.yaml, sorted. */
export function listSuites(track: string, root: string = suitesRoot): string[] {
  const trackRoot = join(root, track);
  if (!existsSync(trackRoot)) return [];
  return readdirSync(trackRoot, { withFileTypes: true })
    .filter((e) => e.isDirectory() && existsSync(join(trackRoot, e.name, "suite.yaml")))
    .map((e) => e.name)
    .sort();
}

/** The registry's per-track handle onto the suites tree. */
export type SuitesNamespace = {
  track: string;
  /** suites/<track> directory */
  root: string;
  /** suite ids with a suite.yaml under this track, sorted */
  list: () => string[];
  /** load "<suite-id>" within this track */
  load: (suiteId: string) => LoadedSuite;
};

export function suitesFor(track: string, root: string = suitesRoot): SuitesNamespace {
  return {
    track,
    root: join(root, track),
    list: () => listSuites(track, root),
    load: (suiteId: string) => loadSuite(`${track}/${suiteId}`, { root }),
  };
}

/** Split "<track>/<suite-id>" — rejects path traversal so ids stay inside
 *  suites/<track>/<suite-id>/. */
function splitSuiteId(id: string): { track: string; suiteId: string } {
  const slash = id.indexOf("/");
  if (slash <= 0 || slash === id.length - 1) {
    throw new Error(`suite id must be "<track>/<suite-id>", got "${id}"`);
  }
  const track = id.slice(0, slash);
  const suiteId = id.slice(slash + 1);
  for (const part of [track, suiteId]) {
    if (part === ".." || part.includes("/") || part.includes("\\")) {
      throw new Error(`invalid suite id "${id}"`);
    }
  }
  return { track, suiteId };
}

/** Load one suite: parse + validate suite.yaml (SuiteMeta) and every
 *  items/<lang>.jsonl line (SuiteItem), then compute suite_hash. Items keep
 *  their authoring order (file order, files sorted by filename). */
export function loadSuite(id: string, opts: { root?: string } = {}): LoadedSuite {
  const { track, suiteId } = splitSuiteId(id);
  const root = opts.root ?? suitesRoot;
  const dir = join(root, track, suiteId);
  const metaPath = join(dir, "suite.yaml");
  if (!existsSync(metaPath)) {
    throw new Error(`suite not found: ${id} (no suite.yaml at ${dir})`);
  }

  const meta = SuiteMeta.parse(parseYaml(readFileSync(metaPath, "utf8")));
  if (meta.track !== track) {
    throw new Error(`suite.yaml track "${meta.track}" does not match directory track "${track}"`);
  }
  if (meta.id !== id) {
    throw new Error(`suite.yaml id "${meta.id}" does not match requested id "${id}"`);
  }

  const itemsDir = join(dir, "items");
  if (!existsSync(itemsDir)) {
    throw new Error(`suite ${id} has no items/ directory (expected ${itemsDir})`);
  }

  const files: SuiteFile[] = [];
  const items: SuiteItemOut[] = [];
  for (const name of readdirSync(itemsDir).sort()) {
    if (!name.endsWith(".jsonl")) continue; // parsed only; the hash covers all entries
    const path = join(itemsDir, name);
    const lang = name.slice(0, -".jsonl".length);
    const fileItems: SuiteItemOut[] = [];
    const lines = readFileSync(path, "utf8").split("\n");
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i]?.trim() ?? "";
      if (line === "") continue; // legacy files end with a trailing newline
      let raw: unknown;
      try {
        raw = JSON.parse(line);
      } catch (cause) {
        throw new Error(`${name}:${i + 1} is not valid JSON`, { cause });
      }
      try {
        fileItems.push(SuiteItem.parse(raw));
      } catch (cause) {
        throw new Error(`${name}:${i + 1} does not match SuiteItem`, { cause });
      }
    }
    files.push({ lang, path, items: fileItems });
    items.push(...fileItems);
  }

  const hash = computeSuiteHash(dir);
  return {
    id,
    dir,
    meta,
    suite_hash: hash,
    hash,
    items,
    files,
  };
}
