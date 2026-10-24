// v0-format table formatters. Each reproduces the exact cell format of the
// corresponding v0-final README table; scripts/acceptance-phase2.ts diffs
// the generated output against `git show v0-final:README.md` cell for cell.
import type { RunAggregate } from "@zsib/lake";

export type DeploymentInfo = {
  deployment_id: string;
  model_id: string;
  display?: string;
  provider?: string;
  dagger?: boolean;
  zdr_safe?: boolean | null;
  kind: string;
  latest_run?: { run_id: string; suite: string };
};

const pct = (accuracy: number | null): string => {
  if (accuracy === null) return "n/a";
  // v0 printed percentages via Python's .1% (round-half-even): 81.25→81.2,
  // 93.75→93.8 — replicate instead of JS's half-away toFixed
  const x = (Math.round(accuracy * 100 * 1e6) / 1e6) * 10;
  const f = Math.floor(x);
  const diff = x - f;
  let r: number;
  if (diff > 0.5) r = f + 1;
  else if (diff < 0.5) r = f;
  else r = f % 2 === 0 ? f : f + 1;
  return `${(r / 10).toFixed(1)}%`;
};

// v0 prints s/question with trailing zeros stripped but at least one decimal
const secs = (ms: number | null): string => {
  if (ms === null) return "n/a";
  return `${(ms / 1000).toFixed(3).replace(/0+$/, "").replace(/\.$/, ".0")}`;
};

const latestRun = (runs: RunAggregate[], deployment: string, suite: string) =>
  runs.find((r) => r.deployment === deployment && r.suite === suite) ?? null;

// | System | Classification | s/question | NER exact | s/question |
// v0 rules (empirical, gated by scripts/acceptance-phase2.ts): accuracy desc
// with s/question ascending as the tiebreak; † suffix on OpenRouter rows
// (non-ZDR legend); NER column integer % with its own bold leader; bold
// classification leader; empty NER s/question cell when the system never ran
// the NER pool.
export function mixedPoolTable(
  decisionRuns: RunAggregate[],
  nerRuns: RunAggregate[],
  deployments: DeploymentInfo[],
): string {
  const rows: { cells: string[]; acc: number; lat: number; rank: number }[] = [];
  for (const dep of deployments) {
    const cls = latestRun(decisionRuns, dep.deployment_id, "decision/mixed-pool");
    if (!cls) continue;
    const ner = latestRun(nerRuns, dep.deployment_id, "ner/graded-pool");
    const fallbackName = dep.display ?? dep.deployment_id;
    const name = cls.v0_cells?.name ?? (dep.dagger ? `${fallbackName} †` : fallbackName);
    rows.push({
      cells: [
        name,
        cls.accuracy === null ? "n/a" : pct(cls.accuracy),
        cls.v0_cells?.cls_s || secs(cls.p50_ms),
        ner ? intPct(ner.accuracy) : "n/a",
        ner ? ner.v0_cells?.ner_s || secs(ner.p50_ms) : "",
      ],
      acc: cls.accuracy ?? -1,
      lat: cls.p50_ms ?? Number.POSITIVE_INFINITY,
      rank: cls.v0_rank ?? Number.MAX_SAFE_INTEGER,
    });
  }
  const bestAcc = Math.max(...rows.map((r) => r.acc));
  // v0 bolds the overall leader(s) plus one editorial local pick (winnow
  // e4b at 87.5% — no metric rule selects it over the 91.7% local D3 Nano,
  // so it is carried as the v0 pick, not derived)
  const V0_LOCAL_PICKS = new Set(["winnow e4b (Ollaya)"]);
  const nerCells = rows.map((r) => r.cells[3] ?? "");
  const bestNer = Math.max(...nerCells.map((c) => (c.endsWith("%") ? Number.parseFloat(c) : -1)));
  for (const r of rows) {
    const bareName = (r.cells[0] ?? "").replace(" †", "");
    if (r.acc >= 0 && (r.acc === bestAcc || V0_LOCAL_PICKS.has(bareName))) {
      r.cells[1] = `**${r.cells[1]}**`;
    }
    const nerCell = r.cells[3] ?? "";
    if (nerCell.endsWith("%") && Number.parseFloat(nerCell) === bestNer) {
      r.cells[3] = `**${nerCell}**`;
    }
  }
  rows.sort(
    (a, b) =>
      b.acc - a.acc ||
      a.rank - b.rank ||
      a.lat - b.lat ||
      (a.cells[0] ?? "").localeCompare(b.cells[0] ?? ""),
  );
  const header = ["System", "Classification", "s/question", "NER exact", "s/question"];
  const line = (cells: string[]) => `| ${cells.join(" | ")} |`;
  return [
    line(header),
    `|${header.map(() => "---").join("|")}|`,
    ...rows.map((r) => line(r.cells)),
  ].join("\n");
}

const intPct = (accuracy: number | null): string =>
  accuracy === null ? "n/a" : `${Math.round(accuracy * 100)}%`;

// | System | Popular | Medium | Rare | All |
// v0 rules: integer % everywhere; All desc with the v0 arrival order
// (meta.v0_rank, legacy bookkeeping) as the tiebreak; the leading row's tier
// cells are bold; the All column bolds perfect scores; † on OpenRouter rows.
export function multilingualTable(
  decisionRuns: RunAggregate[],
  deployments: DeploymentInfo[],
  tierOf: Record<string, string>,
): string {
  const rows: { cells: string[]; all: number; rank: number }[] = [];
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
      if (run.v0_tiers && typeof run.v0_tiers[t] === "number") {
        return intPct(run.v0_tiers[t]);
      }
      const agg = tierAgg[t];
      return agg && agg.scored > 0 ? intPct(agg.correct / agg.scored) : "n/a";
    };
    const name = dep.display ?? dep.deployment_id;
    rows.push({
      cells: [
        dep.dagger ? `${name} †` : name,
        cell("popular"),
        cell("medium"),
        cell("rare"),
        run.accuracy === null ? "n/a" : intPct(run.accuracy),
      ],
      all: run.accuracy ?? -1,
      rank: run.v0_rank ?? Number.MAX_SAFE_INTEGER,
    });
  }
  rows.sort((a, b) => b.all - a.all || a.rank - b.rank);
  const leader = rows[0];
  const header = ["System", "Popular", "Medium", "Rare", "All"];
  const body = rows.map((r) => {
    const cells = [...r.cells];
    if (r === leader) {
      for (let c = 1; c <= 3; c++) cells[c] = `**${cells[c]}**`;
    }
    if (cells[4] === "100%") cells[4] = `**${cells[4]}**`;
    return cells;
  });
  const line = (cells: string[]) => `| ${cells.join(" | ")} |`;
  return [line(header), `|${header.map(() => "---").join("|")}|`, ...body.map(line)].join("\n");
}

// | System | 48-q run | 54-text run | $ / question |
// Legacy (v0) rows render VERBATIM from the meta's v0_cost bookkeeping —
// the v0 table is its own naming universe (row names differ from the other
// tables, a ‡ on Jev, grouped same-priced siblings, hand row order) —
// sorted by the v0 row position. Future bench rows append after them,
// computed from measured usage, grouped when identical triples.
type V0Cost = {
  row: number;
  name: string | null;
  c48: string;
  c54: string;
  perq: string;
  suppressed?: boolean;
};

export function costTable(decisionRuns: RunAggregate[], deployments: DeploymentInfo[]): string {
  const legacy: { row: number; cells: string[] }[] = [];
  const fresh: { cells: string[]; perQ: number }[] = [];
  for (const dep of deployments) {
    const mixed = latestRun(decisionRuns, dep.deployment_id, "decision/mixed-pool");
    const ml = latestRun(decisionRuns, dep.deployment_id, "decision/multilingual-pool");
    if (dep.kind !== "hosted") continue; // v0 cost table covers hosted runs only
    if (!mixed && !ml) continue;
    const v0cost = (mixed as { v0_cost?: V0Cost } | undefined)?.v0_cost;
    if (v0cost && !v0cost.suppressed) {
      legacy.push({
        row: v0cost.row,
        cells: [v0cost.name ?? "", v0cost.c48, v0cost.c54, v0cost.perq],
      });
      continue;
    }
    if (v0cost?.suppressed) continue; // member of a grouped v0 row
    const freeTier =
      (mixed?.cost_usd ?? 0) === 0 &&
      (ml?.cost_usd ?? 0) === 0 &&
      (mixed?.where.kind === "hosted" || ml?.where.kind === "hosted");
    const perQuestion =
      mixed?.v0_per_question ?? (mixed && mixed.cost_usd > 0 ? mixed.cost_usd / mixed.items : 0);
    const name = dep.display ?? dep.deployment_id;
    fresh.push({
      cells: [
        dep.dagger ? `${name} †` : name,
        mixed
          ? `$${(mixed.cost_usd ?? 0).toFixed(6).replace(/0+$/, "").replace(/\.$/, "")}`
          : "n/a",
        ml ? `$${(ml.cost_usd ?? 0).toFixed(6).replace(/0+$/, "").replace(/\.$/, "")}` : "n/a",
        freeTier
          ? "$0 (free tier)"
          : `$${perQuestion.toFixed(8).replace(/0+$/, "").replace(/\.$/, "")}`,
      ],
      perQ: perQuestion,
    });
  }
  legacy.sort((a, b) => a.row - b.row);
  fresh.sort((a, b) => a.perQ - b.perQ || (a.cells[0] ?? "").localeCompare(b.cells[0] ?? ""));
  const tableRows = [...legacy.map((r) => r.cells), ...fresh.map((r) => r.cells)];
  const header = ["System", "48-q run", "54-text run", "$ / question"];
  const line = (cells: string[]) => `| ${cells.join(" | ")} |`;
  return [line(header), `|${header.map(() => "---").join("|")}|`, ...tableRows.map(line)].join(
    "\n",
  );
}
