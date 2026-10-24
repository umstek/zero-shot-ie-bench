import type { EChartsCoreOption } from "echarts/core";
import { fmtDate, fmtPct } from "../lib/format";
import { langLabel, orderLangs } from "../lib/languages";
import type { Deployment, RunWithTrack } from "../types";

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
