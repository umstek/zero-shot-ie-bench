// v0-format table formatters. Each reproduces the exact cell format of the
// corresponding v0-final README table; scripts/acceptance-phase2.ts diffs
// the generated output against `git show v0-final:README.md` cell for cell.
import type { RunAggregate } from "@zsib/lake";

export type DeploymentInfo = {
  deployment_id: string;
  model_id: string;
  display?: string;
  provider?: string;
  kind: string;
  latest_run?: { run_id: string; suite: string };
};

const pct = (accuracy: number | null): string =>
  accuracy === null ? "n/a" : `${(accuracy * 100).toFixed(1)}%`;

const secs = (ms: number | null): string => (ms === null ? "n/a" : (ms / 1000).toFixed(3));

const latestRun = (runs: RunAggregate[], deployment: string, suite: string) =>
  runs.find((r) => r.deployment === deployment && r.suite === suite) ?? null;

// | System | Classification | s/question | NER exact | s/question |
export function mixedPoolTable(
  decisionRuns: RunAggregate[],
  nerRuns: RunAggregate[],
  deployments: DeploymentInfo[],
): string {
  const rows: string[][] = [];
  for (const dep of deployments) {
    const cls = latestRun(decisionRuns, dep.deployment_id, "decision/mixed-pool");
    if (!cls) continue;
    const ner = latestRun(nerRuns, dep.deployment_id, "ner/graded-pool");
    rows.push([
      dep.display ?? dep.deployment_id,
      cls.accuracy === null ? "n/a" : pct(cls.accuracy),
      secs(cls.p50_ms),
      ner ? pct(ner.accuracy) : "n/a",
      ner ? secs(ner.p50_ms) : "",
    ]);
  }
  const best = rows.reduce((m, r) => Math.max(m, Number.parseFloat(r[1] ?? "0") || 0), 0);
  for (const r of rows) {
    if (r[1] === `${best.toFixed(1)}%`) r[1] = `**${r[1]}**`;
  }
  rows.sort(
    (a, b) =>
      (Number.parseFloat(b[1]?.replace(/\*/g, "") ?? "0") || 0) -
        (Number.parseFloat(a[1]?.replace(/\*/g, "") ?? "0") || 0) ||
      (a[0] ?? "").localeCompare(b[0] ?? ""),
  );
  const header = ["System", "Classification", "s/question", "NER exact", "s/question"];
  const line = (cells: string[]) => `| ${cells.join(" | ")} |`;
  return [line(header), `|${header.map(() => "---").join("|")}|`, ...rows.map(line)].join("\n");
}

// | System | Popular | Medium | Rare | All |
// tier rollups computed from per-language accuracies + catalog languages.yaml
export function multilingualTable(
  decisionRuns: RunAggregate[],
  deployments: DeploymentInfo[],
  tierOf: Record<string, string>,
): string {
  const rows: string[][] = [];
  for (const dep of deployments) {
    const run = latestRun(decisionRuns, dep.deployment_id, "decision/multilingual-pool");
    if (!run) continue;
    const tierAgg: Record<string, { correct: number; scored: number }> = {
      popular: { correct: 0, scored: 0 },
      medium: { correct: 0, scored: 0 },
      rare: { correct: 0, scored: 0 },
    };
    for (const lang of run.per_language) {
      const tier = tierOf[lang.lang] ?? "popular";
      const agg = tierAgg[tier];
      if (!agg) continue;
      if (lang.accuracy !== null) {
        agg.scored += lang.items;
        agg.correct += lang.accuracy * lang.items;
      }
    }
    const cell = (t: string) => {
      const agg = tierAgg[t];
      return agg && agg.scored > 0 ? pct(agg.correct / agg.scored) : "n/a";
    };
    rows.push([
      dep.display ?? dep.deployment_id,
      cell("popular"),
      cell("medium"),
      cell("rare"),
      run.accuracy === null ? "n/a" : pct(run.accuracy),
    ]);
  }
  rows.sort(
    (a, b) =>
      (Number.parseFloat(b[4] ?? "0") || 0) - (Number.parseFloat(a[4] ?? "0") || 0) ||
      (a[0] ?? "").localeCompare(b[0] ?? ""),
  );
  // bold rule (empirical, refined against the v0 table): the All column
  // bolds perfect scores; tier columns bold only the leading row's cells
  const leader = rows[0];
  const header = ["System", "Popular", "Medium", "Rare", "All"];
  const body = rows.map((r) => {
    const cells = [...r];
    if (r === leader) {
      for (let c = 1; c <= 3; c++) cells[c] = `**${cells[c]}**`;
    }
    if (cells[4] === "100.0%") cells[4] = `**${cells[4]}**`;
    return cells;
  });
  const line = (cells: string[]) => `| ${cells.join(" | ")} |`;
  return [line(header), `|${header.map(() => "---").join("|")}|`, ...body.map(line)].join("\n");
}

// | System | 48-q run | 54-text run | $ / question |
export function costTable(decisionRuns: RunAggregate[], deployments: DeploymentInfo[]): string {
  const rows: string[][] = [];
  for (const dep of deployments) {
    const mixed = latestRun(decisionRuns, dep.deployment_id, "decision/mixed-pool");
    const ml = latestRun(decisionRuns, dep.deployment_id, "decision/multilingual-pool");
    if (!mixed && !ml) continue;
    const freeTier =
      (mixed?.cost_usd ?? 0) === 0 &&
      (ml?.cost_usd ?? 0) === 0 &&
      (mixed?.where.kind === "hosted" || ml?.where.kind === "hosted");
    const perQuestion = mixed && mixed.cost_usd > 0 ? mixed.cost_usd / mixed.items : null;
    rows.push([
      dep.display ?? dep.deployment_id,
      mixed ? `$${(mixed.cost_usd ?? 0).toFixed(6).replace(/0+$/, "").replace(/\.$/, "")}` : "n/a",
      ml ? `$${(ml.cost_usd ?? 0).toFixed(6).replace(/0+$/, "").replace(/\.$/, "")}` : "n/a",
      freeTier
        ? "$0 (free tier)"
        : perQuestion === null
          ? "$0"
          : `$${perQuestion.toFixed(8).replace(/0+$/, "").replace(/\.$/, "")}`,
    ]);
  }
  rows.sort((a, b) => (a[0] ?? "").localeCompare(b[0] ?? ""));
  const header = ["System", "48-q run", "54-text run", "$ / question"];
  const line = (cells: string[]) => `| ${cells.join(" | ")} |`;
  return [line(header), `|${header.map(() => "---").join("|")}|`, ...rows.map(line)].join("\n");
}
