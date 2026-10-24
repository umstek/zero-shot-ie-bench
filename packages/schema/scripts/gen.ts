// Regenerates catalog/schema/*.json from the Zod definitions in src/.
// `bun run --cwd packages/schema gen` from the root. CI fails if these drift.
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { z } from "zod";
import * as catalog from "../src/catalog.ts";
import * as protocol from "../src/protocol.ts";
import * as run from "../src/run.ts";
import * as suite from "../src/suite.ts";

// One schema file per DATA FILE shape (see scripts/validate.ts for the walk).
const files: Record<string, z.ZodType> = {
  "model.json": catalog.Model,
  "provider.json": catalog.Provider,
  "machine.json": catalog.Machine,
  "track.json": z.array(catalog.Track), // catalog/tracks.yaml holds the list
  "language.json": z.array(catalog.Language), // catalog/languages.yaml
  "suite.json": suite.SuiteMeta,
  "suite-item.json": suite.SuiteItem, // per line of suites/**/items/*.jsonl
  "run-meta.json": run.RunMeta,
  "run-item.json": run.RunItem, // per line of runs/**/*.items.jsonl
  "runner-info.json": protocol.RunnerInfo,
  "track-request.json": protocol.TrackRequest,
  "track-response.json": protocol.TrackResponse,
};

const outDir = join(import.meta.dir, "..", "..", "..", "catalog", "schema");
mkdirSync(outDir, { recursive: true });

for (const [name, schema] of Object.entries(files)) {
  const json = z.toJSONSchema(schema, { target: "draft-2020-12" });
  const body = JSON.stringify(json, null, 2);
  writeFileSync(join(outDir, name), `${body}\n`);
  console.log(`wrote catalog/schema/${name}`);
}
