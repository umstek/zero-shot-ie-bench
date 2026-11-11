import { Link } from "@tanstack/react-router";
import { useMemo } from "react";
import { EChart } from "../charts/echart";
import { type NerEntry, nerF1BarOption } from "../charts/options";
import { fmtDate, fmtMs, fmtPct, platformOf } from "../lib/format";
import { latestSuiteRun } from "../lib/latest";
import { useBoardData } from "../queries/data";
import type { Deployment } from "../types";

// /ner — the extraction track: strict micro-F1 on ner/graded-pool (18 items,
// span-triple gold). One row per deployment, its canonical latest scored run —
// the same selection rule every other board view uses.

export function NerPage() {
  const { deployments, runs } = useBoardData();

  const latest = useMemo(() => latestSuiteRun(runs, "ner/graded-pool"), [runs]);
  const byId = useMemo(
    () => new Map(deployments.map((d): [string, Deployment] => [d.deployment_id, d])),
    [deployments],
  );

  const rows = useMemo(() => {
    const out: { dep: Deployment; entry: NerEntry }[] = [];
    for (const [depId, run] of latest) {
      const dep = byId.get(depId);
      if (!dep) continue;
      out.push({
        dep,
        entry: {
          display: dep.display,
          platform: platformOf(dep),
          kind: dep.kind,
          f1: run.accuracy ?? 0,
          p50Ms: run.p50_ms,
          suite: run.suite,
          startedAt: run.started_at,
        },
      });
    }
    return out.sort(
      (a, b) => b.entry.f1 - a.entry.f1 || a.entry.display.localeCompare(b.entry.display),
    );
  }, [latest, byId]);

  const unbenched = useMemo(
    () =>
      deployments
        .filter((d) => (d.claims?.tracks ?? []).includes("ner") && !latest.has(d.deployment_id))
        .map((d) => d.display)
        .sort((a, b) => a.localeCompare(b)),
    [deployments, latest],
  );

  const option = useMemo(() => nerF1BarOption(rows.map((r) => r.entry)), [rows]);

  return (
    <section>
      <header className="page-head">
        <h1>NER</h1>
        <p>
          Span extraction on <code>ner/graded-pool</code> — 18 items with span-triple gold, strict
          micro-F1 (a predicted span counts only on an exact label-and-boundary set match, the v0
          rule). One row per deployment: its latest scored run, the same selection rule as every
          other board view.
        </p>
      </header>

      <div className="card">
        <h2>Strict micro-F1</h2>
        <p className="muted">
          Highest first. Hover a bar for its p50 and run date; every run on this suite measured zero
          spend, so there is no price view here.
        </p>
        {rows.length === 0 ? (
          <p className="muted">No data.</p>
        ) : (
          <EChart option={option} height={340} />
        )}
      </div>

      <div className="card">
        <h2>Runs</h2>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Kind</th>
                <th>Provider / runner</th>
                <th>Micro-F1 (strict)</th>
                <th>p50</th>
                <th>Run date</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(({ dep, entry }) => (
                <tr key={dep.deployment_id}>
                  <td>
                    <Link to="/model/$" params={{ _splat: dep.model_id }}>
                      {dep.display}
                    </Link>
                  </td>
                  <td>{dep.kind}</td>
                  <td>{entry.platform}</td>
                  <td>{fmtPct(entry.f1)}</td>
                  <td>{fmtMs(entry.p50Ms)}</td>
                  <td>{fmtDate(entry.startedAt)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {unbenched.length > 0 && (
          <p className="muted">Claim NER, not benched on this suite: {unbenched.join(" · ")}</p>
        )}
      </div>
    </section>
  );
}
