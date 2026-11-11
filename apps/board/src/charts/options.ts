import type { EChartsCoreOption } from "echarts/core";
import { measuredPerItem } from "../lib/cost";
import { fmtCostPer1k, fmtDate, fmtMs, fmtPct, platformOf } from "../lib/format";
import { langLabel, orderLangs } from "../lib/languages";
import { latestMixedPoolRun } from "../lib/latest";
import type { Deployment, DeploymentKind, RunWithTrack } from "../types";

// Pure option builders — no echarts imports beyond types, so the chart modules
// stay in echart.tsx and tree-shaking has a single registration point.

interface HistoryPoint {
  value: [number, number];
  modelVersion: string;
  runId: string;
  suite: string;
}

interface SeriesPoint {
  seriesName?: string;
  value?: [number, number];
  data?: Partial<HistoryPoint>;
}

/** Accuracy (0–100) over time, one series per deployment, with version markers. */
export function historyLineOption(
  runs: RunWithTrack[],
  deployments: Deployment[],
): EChartsCoreOption {
  const byDep = new Map<string, RunWithTrack[]>();
  for (const run of runs) {
    const list = byDep.get(run.deployment);
    if (list) list.push(run);
    else byDep.set(run.deployment, [run]);
  }
  const display = new Map(deployments.map((d): [string, string] => [d.deployment_id, d.display]));

  const series = [...byDep.entries()].map(([depId, depRuns]) => {
    const sorted = [...depRuns].sort((a, b) => Date.parse(a.started_at) - Date.parse(b.started_at));
    const points: HistoryPoint[] = [];
    const versions = new Set<string>();
    for (const run of sorted) {
      if (run.accuracy == null) continue;
      versions.add(run.model_version);
      points.push({
        value: [Date.parse(run.started_at), Math.round(run.accuracy * 1000) / 10],
        modelVersion: run.model_version,
        runId: run.run_id,
        suite: run.suite,
      });
    }
    const bounds = [...points].sort((a, b) => a.value[0] - b.value[0]);
    const markers: { coord: [number, number]; name: string }[] = [];
    for (let i = 1; i < bounds.length; i++) {
      const prev = bounds[i - 1];
      const curr = bounds[i];
      if (prev && curr && prev.modelVersion !== curr.modelVersion) {
        markers.push({ coord: curr.value, name: curr.modelVersion });
      }
    }
    return {
      name: display.get(depId) ?? depId,
      type: "line" as const,
      symbolSize: 8,
      connectNulls: true,
      data: points,
      markPoint:
        versions.size > 1
          ? {
              symbol: "diamond",
              symbolSize: 12,
              itemStyle: { color: "#d97706" },
              label: {
                position: "top" as const,
                fontSize: 10,
                formatter: (p: { name?: string }) => p.name ?? "",
              },
              data: markers,
            }
          : undefined,
    };
  });

  return {
    tooltip: {
      trigger: "axis",
      formatter: (params: unknown) => {
        const list = (Array.isArray(params) ? params : [params]) as SeriesPoint[];
        return list
          .filter((p) => p.value != null)
          .map((p) => {
            const ts = p.value?.[0] ?? 0;
            const acc = p.value?.[1] ?? 0;
            return [
              `${p.seriesName ?? ""} · ${fmtDate(new Date(ts).toISOString())}`,
              `${fmtPct(acc / 100)} · ${p.data?.modelVersion ?? ""}`,
              `${p.data?.suite ?? ""} · run ${p.data?.runId ?? ""}`,
            ].join("<br/>");
          })
          .join("<br/><br/>");
      },
    },
    legend: { top: 0, type: "scroll" },
    grid: { left: 44, right: 24, top: 40, bottom: 56 },
    xAxis: {
      type: "time",
      axisLabel: { formatter: (v: number) => fmtDate(new Date(v).toISOString()) },
    },
    yAxis: { type: "value", min: 0, max: 100, axisLabel: { formatter: "{value}%" } },
    series,
  };
}

/** per_language × runs heatmap: one row per run, one column per language. */
export function languageHeatmapOption(
  runs: RunWithTrack[],
  deployments: Deployment[],
): EChartsCoreOption {
  const usable = runs
    .filter((r) => (r.per_language ?? []).some((p) => p.accuracy != null))
    .sort((a, b) => Date.parse(b.started_at) - Date.parse(a.started_at)); // newest first
  const display = new Map(deployments.map((d): [string, string] => [d.deployment_id, d.display]));

  const present = new Set<string>();
  for (const run of usable) {
    for (const pl of run.per_language ?? []) {
      if (pl.accuracy != null) present.add(pl.lang);
    }
  }
  const langs = orderLangs(present);
  const rows = usable.map(
    (r) =>
      `${display.get(r.deployment) ?? r.deployment} · ${fmtDate(r.started_at)}${
        r.track === "ner" ? " (ner)" : ""
      }`,
  );

  const cells: [number, number, number][] = [];
  for (let y = 0; y < usable.length; y++) {
    const run = usable[y];
    if (!run) continue;
    for (let x = 0; x < langs.length; x++) {
      const lang = langs[x];
      const pl = (run.per_language ?? []).find((p) => p.lang === lang);
      if (lang && pl?.accuracy != null) {
        cells.push([x, y, Math.round(pl.accuracy * 1000) / 10]);
      }
    }
  }

  return {
    tooltip: {
      formatter: (params: unknown) => {
        const p = params as { value?: [number, number, number] };
        const v = p.value;
        if (!v) return "";
        const run = usable[v[1]];
        const lang = langs[v[0]];
        if (!run || !lang) return "";
        const pl = (run.per_language ?? []).find((x) => x.lang === lang);
        return `${run.run_id}<br/>${langLabel(lang)}: ${fmtPct(pl?.accuracy)} (${pl?.correct ?? "?"}/${
          pl?.items ?? "?"
        } items)`;
      },
    },
    grid: { left: 220, right: 110, top: 10, bottom: 24 },
    xAxis: { type: "category", data: langs.map(langLabel), splitArea: { show: true } },
    yAxis: { type: "category", data: rows, splitArea: { show: true }, inverse: true },
    visualMap: {
      min: 0,
      max: 100,
      calculable: false,
      orient: "vertical",
      right: 8,
      top: "center",
      text: ["100%", "0%"],
      inRange: { color: ["#f8fafc", "#bfdbfe", "#3b82f6", "#1d4ed8"] },
    },
    series: [
      {
        type: "heatmap" as const,
        data: cells,
        label: {
          show: true,
          formatter: (p: unknown) => {
            const q = p as { value?: [number, number, number] };
            return q.value ? `${q.value[2]}%` : "";
          },
        },
      },
    ],
  };
}

// Compare scatters: one point per deployment over its latest usable decision
// run, hosted/local split for the speed chart, measured spend only for price.

/** One plotted deployment — shared by both compare charts. */
export interface ComparePoint {
  deployment: string;
  display: string;
  /** provider (hosted) or runner (local) */
  platform: string;
  kind: DeploymentKind;
  /** 0–1 */
  accuracy: number;
  /** run cost ÷ items (v0 bookkeeping figure when present) */
  perItem: number | null;
  p50Ms: number | null;
  /** provenance, shown in tooltips so every number is self-describing */
  suite: string;
  startedAt: string;
}

interface ScatterPoint {
  value: [number, number];
  display: string;
  platform: string;
  perItem: number | null;
  p50Ms: number | null;
  suite: string;
  startedAt: string;
}

interface ScatterDatum {
  seriesName?: string;
  value?: [number, number];
  data?: Partial<ScatterPoint>;
}

function toPoint(dep: Deployment, run: RunWithTrack): ComparePoint {
  return {
    deployment: dep.deployment_id,
    display: dep.display,
    platform: platformOf(dep),
    kind: dep.kind,
    accuracy: run.accuracy ?? 0,
    perItem: measuredPerItem(run),
    p50Ms: run.p50_ms,
    suite: run.suite,
    startedAt: run.started_at,
  };
}

/** Speed vs accuracy points: every deployment whose canonical mixed-pool run
 *  also has a measured p50 (a log x-axis cannot place the rest). */
export function speedPoints(runs: RunWithTrack[], deployments: Deployment[]): ComparePoint[] {
  const byId = new Map(deployments.map((d): [string, Deployment] => [d.deployment_id, d]));
  const points: ComparePoint[] = [];
  for (const [depId, run] of latestMixedPoolRun(runs)) {
    const dep = byId.get(depId);
    if (!dep || run.p50_ms == null) continue;
    points.push(toPoint(dep, run));
  }
  return points.sort((a, b) => a.display.localeCompare(b.display));
}

/** Price vs accuracy points: hosted deployments whose canonical mixed-pool run
 *  carries measured positive spend only (free-tier/unpriced never rank on
 *  price — same rule as the cheapest ≥90% pick). */
export function pricePoints(runs: RunWithTrack[], deployments: Deployment[]): ComparePoint[] {
  const byId = new Map(deployments.map((d): [string, Deployment] => [d.deployment_id, d]));
  const points: ComparePoint[] = [];
  for (const [depId, run] of latestMixedPoolRun(runs)) {
    const dep = byId.get(depId);
    if (dep?.kind !== "hosted" || measuredPerItem(run) == null) continue;
    points.push(toPoint(dep, run));
  }
  return points.sort((a, b) => a.display.localeCompare(b.display));
}

function scatterData(points: ComparePoint[], x: (p: ComparePoint) => number): ScatterPoint[] {
  const data: ScatterPoint[] = [];
  for (const p of points) {
    data.push({
      value: [x(p), Math.round(p.accuracy * 1000) / 10],
      display: p.display,
      platform: p.platform,
      perItem: p.perItem,
      p50Ms: p.p50Ms,
      suite: p.suite,
      startedAt: p.startedAt,
    });
  }
  return data;
}

/** Non-dominated points: no other point is both lower on x (faster / cheaper)
 *  and at least as accurate. Sorted by x ascending with accuracy strictly
 *  rising along the line — the best-available trade-off staircase. */
export function paretoFrontier(
  points: ComparePoint[],
  x: (p: ComparePoint) => number,
): ComparePoint[] {
  const sorted = [...points].sort((a, b) => x(a) - x(b) || b.accuracy - a.accuracy);
  const frontier: ComparePoint[] = [];
  let best = Number.NEGATIVE_INFINITY;
  for (const p of sorted) {
    if (p.accuracy > best) {
      frontier.push(p);
      best = p.accuracy;
    }
  }
  return frontier;
}

/** The dashed frontier line drawn over a compare scatter. */
function frontierSeries(points: ComparePoint[], x: (p: ComparePoint) => number) {
  return {
    name: "Pareto frontier",
    type: "line" as const,
    symbolSize: 6,
    lineStyle: { type: "dashed" as const, width: 1.5, color: "#64748b" },
    itemStyle: { color: "#64748b" },
    data: scatterData(paretoFrontier(points, x), x),
  };
}

/** p50 latency (log ms) vs accuracy, one series per deployment kind. */
export function speedScatterOption(points: ComparePoint[]): EChartsCoreOption {
  const data = (kind: DeploymentKind) =>
    scatterData(
      points.filter((p) => p.kind === kind && p.p50Ms != null),
      (p) => p.p50Ms ?? 0,
    );
  return {
    tooltip: {
      trigger: "item",
      formatter: (params: unknown) => {
        const p = params as ScatterDatum;
        if (!p.data) return "";
        return [
          `${p.data.display ?? ""} · ${p.data.platform ?? ""}`,
          `accuracy ${fmtPct((p.value?.[1] ?? 0) / 100)} · p50 ${fmtMs(p.data.p50Ms)}`,
          `${p.data.suite ?? ""} · ${fmtDate(p.data.startedAt ?? "")}`,
        ].join("<br/>");
      },
    },
    legend: { top: 0 },
    grid: { left: 44, right: 24, top: 40, bottom: 62 },
    xAxis: {
      type: "log",
      name: "p50 latency (log scale)",
      nameLocation: "middle",
      nameGap: 32,
      axisLabel: { formatter: (v: number) => fmtMs(v) },
    },
    yAxis: { type: "value", min: 0, max: 100, axisLabel: { formatter: "{value}%" } },
    series: [
      { name: "hosted", type: "scatter" as const, symbolSize: 10, data: data("hosted") },
      { name: "local", type: "scatter" as const, symbolSize: 10, data: data("local") },
      frontierSeries(points, (p) => p.p50Ms ?? 0),
    ],
  };
}

/** $ per 1k items (log) vs accuracy, measured spend only — a single hosted
 *  series (local runs carry no metered cost). */
export function priceScatterOption(points: ComparePoint[]): EChartsCoreOption {
  const usable = points.filter((p) => p.perItem != null);
  const data = scatterData(usable, (p) => (p.perItem ?? 0) * 1000);
  return {
    tooltip: {
      trigger: "item",
      formatter: (params: unknown) => {
        const p = params as ScatterDatum;
        if (!p.data) return "";
        return [
          `${p.data.display ?? ""} · ${p.data.platform ?? ""}`,
          `${fmtCostPer1k(p.data.perItem ?? 0)} · accuracy ${fmtPct((p.value?.[1] ?? 0) / 100)}`,
          `${p.data.suite ?? ""} · ${fmtDate(p.data.startedAt ?? "")}`,
        ].join("<br/>");
      },
    },
    legend: { top: 0 },
    grid: { left: 44, right: 24, top: 40, bottom: 62 },
    xAxis: {
      type: "log",
      name: "$ per 1k items (log scale)",
      nameLocation: "middle",
      nameGap: 32,
      axisLabel: {
        formatter: (v: number) => `$${v < 0.01 ? v.toFixed(4) : v.toFixed(2)}`,
      },
    },
    yAxis: { type: "value", min: 0, max: 100, axisLabel: { formatter: "{value}%" } },
    series: [
      { name: "hosted", type: "scatter" as const, symbolSize: 10, data },
      frontierSeries(usable, (p) => (p.perItem ?? 0) * 1000),
    ],
  };
}
