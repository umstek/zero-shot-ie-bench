// Run writing — the append-only heart of the bench (docs/plan/04).
// A run is identified by the tuple in docs/plan/02; runs/ is never edited
// after write (CI-enforced). Deferred runs (R2) record WHY they didn't run.
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { RunItemOut, RunMetaOut } from "@zsib/schema";
import { RunItem, RunMeta } from "@zsib/schema";

const repoRoot = join(import.meta.dir, "..", "..", "..");

export type RunIdentity = {
  deployment: string;
  model_version: string;
  suite: string;
  suite_version: number;
  suite_hash: string;
  params: Record<string, unknown>;
};

export function newRunId(prefix: string): string {
  // time-sortable, collision-safe without dependencies
  const rand = crypto.getRandomValues(new BigUint64Array(1))[0]!.toString(36);
  return `${prefix}-${Date.now().toString(36)}-${rand}`;
}

export async function gitSha(): Promise<string> {
  return (await Bun.$`git rev-parse HEAD`.cwd(repoRoot).text()).trim();
}

function runDir(track: string, startedAt: Date): string {
  const yyyy = startedAt.getUTCFullYear();
  const mm = String(startedAt.getUTCMonth() + 1).padStart(2, "0");
  return join(repoRoot, "runs", track, `${yyyy}-${mm}`);
}

/** Append-only writer: refuses to overwrite an existing run file. */
export class RunWriter {
  readonly runId: string;
  private readonly dir: string;
  private readonly items: RunItemOut[] = [];

  constructor(
    readonly identity: RunIdentity,
    readonly where: RunMetaOut["where"],
    runId = newRunId(identity.suite.split("/")[1] ?? "run"),
    rootOverride?: string, // tests write to a temp tree, never the real runs/
  ) {
    this.runId = runId;
    this.dir = rootOverride
      ? join(rootOverride, identity.suite.split("/")[0] ?? "unknown")
      : runDir(identity.suite.split("/")[0] ?? "unknown", new Date());
  }

  push(item: RunItemOut): void {
    this.items.push(item);
  }

  /** read-only view for totals computation */
  snapshot(): readonly RunItemOut[] {
    return this.items;
  }

  async write(meta: {
    status: "ok" | "partial" | "failed" | "deferred";
    deferred_reason?: string | null;
    served_version?: string | null;
    totals: RunMetaOut["totals"];
    source?: string;
  }): Promise<{ metaPath: string; itemsPath: string }> {
    mkdirSync(this.dir, { recursive: true });
    const metaPath = join(this.dir, `${this.runId}.meta.json`);
    const itemsPath = join(this.dir, `${this.runId}.items.jsonl`);
    if (existsSync(metaPath) || existsSync(itemsPath)) {
      throw new Error(`run ${this.runId} already exists — runs are append-only`);
    }
    const now = new Date().toISOString();
    const runMeta: RunMetaOut = {
      run_id: this.runId,
      status: meta.status,
      deferred_reason: meta.deferred_reason ?? null,
      deployment: this.identity.deployment,
      model_version: this.identity.model_version,
      served_version: meta.served_version ?? null,
      where: this.where,
      suite: this.identity.suite,
      suite_version: this.identity.suite_version,
      suite_hash: this.identity.suite_hash,
      params: this.identity.params,
      params_hash: null,
      bench: { version: "0.1.0", git_sha: await gitSha() },
      started_at: now,
      finished_at: now,
      totals: meta.totals,
      source: meta.source ?? "bench",
    };
    RunMeta.parse(runMeta); // refuse to write anything off-schema
    for (const item of this.items) RunItem.parse(item);
    writeFileSync(metaPath, `${JSON.stringify(runMeta, null, 1)}\n`, {
      encoding: "utf-8",
      flag: "wx",
    });
    writeFileSync(itemsPath, this.items.map((i) => JSON.stringify(i)).join("\n") + "\n", {
      encoding: "utf-8",
      flag: "wx",
    });
    return { metaPath, itemsPath };
  }

  /** a deferred run: no items, the reason is the record (R2) */
  static async writeDeferred(
    identity: RunIdentity,
    where: RunMetaOut["where"],
    reason: string,
    rootOverride?: string,
  ): Promise<{ metaPath: string; itemsPath: string }> {
    const writer = new RunWriter(identity, where, newRunId("deferred"), rootOverride);
    return writer.write({
      status: "deferred",
      deferred_reason: reason,
      totals: { items: 0, correct: null, cost_usd: 0, p50_ms: null, p95_ms: null, errors: 0 },
    });
  }
}
