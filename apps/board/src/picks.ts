import { fmtCostPer1k, fmtDate, fmtMs, fmtPct, platformOf } from "./lib/format";
import { langLabel, RARE_LANGS } from "./lib/languages";
import type { Deployment, RunWithTrack } from "./types";

// Picks answer concrete questions from the latest OK decision-track run per
// deployment. All ranking is in-memory over the built JSON — no backend.

export interface PickRow {
  deployment: Deployment;
  run: RunWithTrack;
  headline: string;
  detail: string;
}

export interface Pick {
  id: string;
  question: string;
  note: string;
  rows: PickRow[];
}

const TOP_N = 5;

function startedAt(run: RunWithTrack): number {
  return Date.parse(run.started_at);
}

/** Latest usable run per (deployment, SUITE) — suites are not comparable, so
 *  a fresh mixed-pool run must never shadow a multilingual result (and
 *  vice versa). Failed/deferred runs never rank. */
export function latestRunByDeploymentAndSuite(runs: RunWithTrack[]): Map<string, RunWithTrack> {
  const latest = new Map<string, RunWithTrack>(); // key: deployment + "\u0000" + suite
  for (const run of runs) {
    if (run.track !== "decision" || run.status === "failed" || run.status === "deferred") continue;
    if (run.accuracy == null) continue;
    const key = `${run.deployment}\u0000${run.suite}`;
    const prev = latest.get(key);
    if (!prev || startedAt(run) > startedAt(prev)) latest.set(key, run);
  }
  return latest;
}

interface Candidate {
  deployment: Deployment;
  run: RunWithTrack;
}

function candidates(
  latest: Map<string, RunWithTrack>,
  byId: Map<string, Deployment>,
  suite?: string,
): Candidate[] {
  const rows: Candidate[] = [];
  for (const [key, run] of latest) {
    if (suite && run.suite !== suite) continue;
    const depId = key.split("\u0000")[0] ?? "";
    const deployment = byId.get(depId);
    if (deployment) rows.push({ deployment, run });
  }
  return rows;
}

function rareLanguageScore(run: RunWithTrack): { mean: number; parts: string[] } | null {
  const measured: { lang: string; acc: number }[] = [];
  for (const pl of run.per_language ?? []) {
    if (RARE_LANGS.includes(pl.lang) && pl.accuracy != null) {
      measured.push({ lang: pl.lang, acc: pl.accuracy });
    }
  }
  if (measured.length < 2) return null; // one rare language is not a profile
  const mean = measured.reduce((sum, m) => sum + m.acc, 0) / measured.length;
  const parts = measured
    .sort((a, b) => a.lang.localeCompare(b.lang))
    .map((m) => `${langLabel(m.lang)} ${fmtPct(m.acc)}`);
  return { mean, parts };
}

export function buildPicks(deployments: Deployment[], runs: RunWithTrack[]): Pick[] {
  const latest = latestRunByDeploymentAndSuite(runs);
  const byId = new Map(deployments.map((d): [string, Deployment] => [d.deployment_id, d]));
  const mixedPool = candidates(latest, byId, "decision/mixed-pool");
  const anyDecision = candidates(latest, byId);

  // Cheapest ≥90%: the advertised threshold is enforced BEFORE ranking;
  // cost per item (run cost ÷ items), accuracy as tiebreak.
  const cheapest = mixedPool
    .filter((c) => (c.run.accuracy ?? 0) >= 0.9)
    .map((c) => ({ ...c, perItem: c.run.cost_usd / Math.max(c.run.items, 1) }))
    .sort((a, b) => a.perItem - b.perItem || (b.run.accuracy ?? 0) - (a.run.accuracy ?? 0))
    .slice(0, TOP_N)
    .map<PickRow>((c) => ({
      deployment: c.deployment,
      run: c.run,
      headline: `${fmtPct(c.run.accuracy)} · ${fmtCostPer1k(c.perItem)}`,
      detail: `${platformOf(c.deployment)} · ${c.run.suite} · ${c.run.items} items · ${fmtDate(
        c.run.started_at,
      )}`,
    }));

  // Fastest local on CPU: catalog requires.accel === "cpu", ranked by measured p50.
  const fastestLocal = mixedPool
    .filter(
      (c) =>
        c.deployment.kind === "local" &&
        c.deployment.requires?.accel === "cpu" &&
        c.run.p50_ms != null,
    )
    .sort((a, b) => (a.run.p50_ms ?? 0) - (b.run.p50_ms ?? 0))
    .slice(0, TOP_N)
    .map<PickRow>((c) => {
      const req = c.deployment.requires;
      const gb = (v: number | null | undefined) => (v == null ? "?" : String(v));
      const hw = req ? `${gb(req.vram_gb)} GB VRAM / ${gb(req.ram_gb)} GB RAM needed` : null;
      return {
        deployment: c.deployment,
        run: c.run,
        headline: `${fmtMs(c.run.p50_ms)} p50 · ${fmtPct(c.run.accuracy)}`,
        detail: [
          `machine ${c.run.where.machine ?? "?"}`,
          hw,
          c.run.suite,
          fmtDate(c.run.started_at),
        ]
          .filter((part) => part != null && part !== "")
          .join(" · "),
      };
    });

  // Best rare-language: mean accuracy over the rare tier (≥2 languages measured).
  const rare = candidates(latest, byId, "decision/multilingual-pool")
    .map((c) => ({ ...c, score: rareLanguageScore(c.run) }))
    .filter((c) => c.score != null)
    .sort((a, b) => (b.score?.mean ?? 0) - (a.score?.mean ?? 0))
    .slice(0, TOP_N)
    .map<PickRow>((c) => ({
      deployment: c.deployment,
      run: c.run,
      headline: `${fmtPct(c.score?.mean)} over rare languages`,
      detail: `${c.score?.parts.join(" · ")} · ${c.run.suite} · ${fmtDate(c.run.started_at)}`,
    }));

  // Best open-weights.
  const openWeights = mixedPool
    .filter((c) => c.deployment.open_weights)
    .sort((a, b) => (b.run.accuracy ?? 0) - (a.run.accuracy ?? 0))
    .slice(0, TOP_N)
    .map<PickRow>((c) => ({
      deployment: c.deployment,
      run: c.run,
      headline: fmtPct(c.run.accuracy),
      detail: `${c.deployment.license} · ${platformOf(c.deployment)} · ${c.run.suite} · ${fmtDate(
        c.run.started_at,
      )}`,
    }));

  // ZDR-safe only.
  const zdrSafe = mixedPool
    .filter((c) => c.deployment.zdr_safe === true)
    .sort((a, b) => (b.run.accuracy ?? 0) - (a.run.accuracy ?? 0))
    .slice(0, TOP_N)
    .map<PickRow>((c) => ({
      deployment: c.deployment,
      run: c.run,
      headline: fmtPct(c.run.accuracy),
      detail: `${platformOf(c.deployment)} · ${c.run.suite} · ${fmtDate(c.run.started_at)}`,
    }));

  return [
    {
      id: "cheapest",
      question: "Which deployment is the cheapest that still scores ≥90%?",
      note: "Cost per item = run cost ÷ items. Legacy local runs carry a $0 amortised cost, so free tiers and unpriced local models sort first.",
      rows: cheapest,
    },
    {
      id: "fastest-local-cpu",
      question: "What is the fastest local model on CPU?",
      note: "Local deployments whose catalog entry requires accel: cpu, ranked by the run's measured p50 latency (compute_ms measured inside the runner).",
      rows: fastestLocal,
    },
    {
      id: "best-rare-language",
      question: "Which model holds up best on rare languages?",
      note: "Mean accuracy over the rare tier (Sinhala, Icelandic, Welsh) from the run's per-language stats. Deployments with fewer than two measured rare languages are excluded.",
      rows: rare,
    },
    {
      id: "best-open-weights",
      question: "What is the best open-weights deployment?",
      note: "Open-weights deployments only, ranked by latest decision-track accuracy.",
      rows: openWeights,
    },
    {
      id: "zdr-safe",
      question: "What is the best ZDR-safe deployment?",
      note: "Zero-data-retention-safe deployments only (catalog flag zdr_safe). Entries without the flag are treated as unknown and excluded.",
      rows: zdrSafe,
    },
  ];
}
