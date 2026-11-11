import { useMemo } from "react";
import { EChart } from "../charts/echart";
import {
  languageSupportOption,
  pricePoints,
  priceScatterOption,
  speedPoints,
  speedScatterOption,
} from "../charts/options";
import { latestSuiteRun } from "../lib/latest";
import { useBoardData } from "../queries/data";

// /compare — cross-deployment charts over each deployment's canonical runs.

export function ComparePage() {
  const { deployments, runs } = useBoardData();

  const speed = useMemo(() => speedPoints(runs, deployments), [runs, deployments]);
  const price = useMemo(() => pricePoints(runs, deployments), [runs, deployments]);
  const speedOption = useMemo(() => speedScatterOption(speed), [speed]);
  const priceOption = useMemo(() => priceScatterOption(price), [price]);

  const langs = useMemo(() => latestSuiteRun(runs, "decision/multilingual-pool"), [runs]);
  const langOption = useMemo(() => languageSupportOption(langs, deployments), [langs, deployments]);

  return (
    <section>
      <header className="page-head">
        <h1>Charts</h1>
        <p>
          Cross-deployment comparisons over each deployment's latest scored decision/mixed-pool run
          — the same selection the explorer and picks use. Hover a point for its run.
        </p>
      </header>

      <div className="card">
        <h2>Speed vs accuracy</h2>
        <p className="muted">
          Measured p50 latency (log scale) against accuracy; one point per deployment, split by
          hosted/local. The dashed Pareto frontier marks systems where nothing is both faster and at
          least as accurate. Runs without a measured p50 can't be plotted.
        </p>
        {speed.length === 0 ? (
          <p className="muted">No data.</p>
        ) : (
          <EChart option={speedOption} height={360} />
        )}
      </div>

      <div className="card">
        <h2>Price vs accuracy</h2>
        <p className="muted">
          Measured spend per 1k items (log scale) against accuracy, hosted deployments with a
          positive measured cost only — free-tier and unpriced systems are excluded. The dashed
          Pareto frontier marks systems where nothing is both cheaper and at least as accurate.
        </p>
        {price.length === 0 ? (
          <p className="muted">No data.</p>
        ) : (
          <EChart option={priceOption} height={360} />
        )}
      </div>

      <div className="card">
        <h2>Language support</h2>
        <p className="muted">
          Per-language accuracy, one row per deployment — its latest scored decision/
          multilingual-pool run, rows sorted by mean accuracy. Blank cells were not measured; hover
          a cell for its counts and run date.
        </p>
        {langs.size === 0 ? (
          <p className="muted">No data.</p>
        ) : (
          <EChart option={langOption} height={Math.max(360, langs.size * 22 + 130)} />
        )}
      </div>
    </section>
  );
}
