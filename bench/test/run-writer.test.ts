import { describe, expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { newRunId, RunWriter } from "../src/run/writer.ts";

const identity = {
  deployment: "test.local",
  model_version: "test@unversioned",
  suite: "decision/mixed-pool",
  suite_version: 1,
  suite_hash: `sha256:${"0".repeat(64)}`,
  params: {},
};

const tmpRoot = () => mkdtempSync(join(tmpdir(), "zsib-runs-"));

import { join } from "node:path";

describe("RunWriter", () => {
  test("run ids are unique and time-sortable-prefixed", () => {
    const a = newRunId("mixed");
    const b = newRunId("mixed");
    expect(a).not.toBe(b);
    expect(a.startsWith("mixed-")).toBe(true);
  });

  test("writes meta + items append-only and refuses overwrites", async () => {
    const root = tmpRoot();
    const w = new RunWriter(
      identity,
      { kind: "local", machine: "seoul", runtime: {} },
      "testrun-a",
      root,
    );
    w.push({
      item_id: "mp-001",
      lang: "en",
      prediction: "positive",
      gold: "positive",
      correct: true,
      score: null,
      prob: 0.9,
      latency_ms: 12,
      compute_ms: 8,
      cost_usd: 0,
      usage: null,
      error: null,
    });
    const { metaPath, itemsPath } = await w.write({
      status: "ok",
      totals: { items: 1, correct: 1, cost_usd: 0, p50_ms: 12, p95_ms: 12, errors: 0 },
    });
    const meta = JSON.parse(await Bun.file(metaPath).text());
    expect(meta.run_id).toBe("testrun-a");
    expect(meta.source).toBe("bench");
    expect(meta.bench.git_sha).toMatch(/^[0-9a-f]{7,40}$/);
    const lines = (await Bun.file(itemsPath).text()).trim().split("\n");
    expect(lines).toHaveLength(1);
    expect(JSON.parse(lines[0] ?? "{}").correct).toBe(true);

    // append-only: a second writer with the same id, same root, must refuse
    const dup = new RunWriter(
      identity,
      { kind: "local", machine: "seoul", runtime: {} },
      "testrun-a",
      root,
    );
    await expect(
      dup.write({
        status: "ok",
        totals: { items: 0, correct: null, cost_usd: 0, p50_ms: null, p95_ms: null, errors: 0 },
      }),
    ).rejects.toThrow(/append-only/);
  });
});
