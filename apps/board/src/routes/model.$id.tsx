import { Link, useParams } from "@tanstack/react-router";
import { useMemo } from "react";
import { EChart } from "../charts/echart";
import { historyLineOption, languageHeatmapOption } from "../charts/options";
import { fmtCostPer1k, fmtDate, fmtMs, fmtPct } from "../lib/format";
import { useBoardData } from "../queries/data";

// /model/<model_id> — splat route, because model ids contain a slash
// (e.g. fastino/gliner2.5-base) and one model can own several deployments.

export function ModelPage() {
  const params = useParams({ from: "/model/$" });
  const modelId = params._splat ?? "";
  const { deployments, runs } = useBoardData();

  const modelDeployments = useMemo(
    () => deployments.filter((d) => d.model_id === modelId),
    [deployments, modelId],
  );

  const modelRuns = useMemo(() => {
    const ids = new Set(modelDeployments.map((d) => d.deployment_id));
    return runs
      .filter((r) => ids.has(r.deployment))
      .sort((a, b) => Date.parse(a.started_at) - Date.parse(b.started_at));
  }, [runs, modelDeployments]);

  // Hooks stay above the early return so the hook order never changes.
  const lineOption = useMemo(
    () => historyLineOption(modelRuns, modelDeployments),
    [modelRuns, modelDeployments],
  );
  const heatOption = useMemo(
    () => languageHeatmapOption(modelRuns, modelDeployments),
    [modelRuns, modelDeployments],
  );

  if (modelDeployments.length === 0) {
    return (
      <section>
        <h1>Unknown model</h1>
        <p className="muted">
          No deployment in the lake for <code>{modelId}</code>.{" "}
          <Link to="/explore">Back to the explorer.</Link>
        </p>
      </section>
    );
  }

  const first = modelDeployments[0];

  return (
    <section>
      <header className="page-head">
        <p className="muted mono">{modelId}</p>
        <h1>{first?.display ?? modelId}</h1>
        <p>
          {first && `${first.org} · `}
          {modelDeployments.length} deployment{modelDeployments.length === 1 ? "" : "s"} ·{" "}
          {modelRuns.length} run{modelRuns.length === 1 ? "" : "s"}
        </p>
      </header>

      <div className="card">
        <h2>Deployments</h2>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Deployment</th>
                <th>Kind</th>
                <th>Provider / runner</th>
                <th>Version</th>
                <th>ZDR</th>
                <th>License</th>
                <th>Latest acc</th>
              </tr>
            </thead>
            <tbody>
              {modelDeployments.map((d) => (
                <tr key={d.deployment_id}>
                  <td className="mono">{d.deployment_id}</td>
                  <td>{d.kind}</td>
                  <td>{d.provider ?? d.runner ?? "—"}</td>
                  <td className="mono">{d.version}</td>
                  <td>{d.zdr_safe == null ? "unknown" : d.zdr_safe ? "safe" : "no"}</td>
                  <td>{d.license}</td>
                  <td>{fmtPct(d.latest_run?.accuracy)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <h2>Accuracy history</h2>
        <p className="muted">
          One point per OK run. Diamonds mark a model-version boundary; hover for the run id.
        </p>
        <EChart option={lineOption} height={320} />
      </div>

      <div className="card">
        <h2>Language heatmap</h2>
        <p className="muted">
          per_language accuracy for every run of this model (newest first). Grey cells were not
          measured.
        </p>
        {modelRuns.some((r) => (r.per_language ?? []).some((p) => p.accuracy != null)) ? (
          <EChart option={heatOption} height={Math.max(180, modelRuns.length * 34 + 120)} />
        ) : (
          <p className="muted">No per-language stats recorded for this model.</p>
        )}
      </div>

      <div className="card">
        <h2>Run history</h2>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Run</th>
                <th>Track</th>
                <th>Suite</th>
                <th>Date</th>
                <th>Status</th>
                <th>Accuracy</th>
                <th>Items</th>
                <th>Cost / 1k</th>
                <th>p50</th>
              </tr>
            </thead>
            <tbody>
              {[...modelRuns].reverse().map((r) => (
                <tr key={r.run_id}>
                  <td className="mono">{r.run_id}</td>
                  <td>{r.track}</td>
                  <td>{r.suite}</td>
                  <td>{fmtDate(r.started_at)}</td>
                  <td>{r.status}</td>
                  <td>{fmtPct(r.accuracy)}</td>
                  <td>{r.items}</td>
                  <td>{fmtCostPer1k(r.cost_usd / Math.max(r.items, 1))}</td>
                  <td>{fmtMs(r.p50_ms)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
