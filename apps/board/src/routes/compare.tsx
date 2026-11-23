import { useMemo } from "react";
import { EChart } from "../charts/echart";
import {
  costSpeedScatterOption,
  languageSupportOption,
  pricePoints,
  priceScatterOption,
  rankedBarOption,
  spectrumBarOption,
  speedPoints,
  speedScatterOption,
} from "../charts/options";
import { fmtMs, fmtPct } from "../lib/format";
import { latestSuiteRun } from "../lib/latest";
import { useBoardData } from "../queries/data";

// /compare — cross-deployment charts over each deployment's canonical runs.

export function ComparePage() {
  const { deployments, runs, spectrum } = useBoardData();

  const speed = useMemo(() => speedPoints(runs, deployments), [runs, deployments]);
  const price = useMemo(() => pricePoints(runs, deployments), [runs, deployments]);
  const speedOption = useMemo(() => speedScatterOption(speed), [speed]);
  const priceOption = useMemo(() => priceScatterOption(price), [price]);
  const costSpeedOption = useMemo(() => costSpeedScatterOption(price), [price]);
  const accBarOption = useMemo(
    () =>
      rankedBarOption(speed, {
        value: (p) => p.accuracy,
        label: "accuracy",
        higherFirst: true,
        format: fmtPct,
        top: 20,
      }),
    [speed],
  );
  const p50BarOption = useMemo(
    () =>
      rankedBarOption(speed, {
        value: (p) => p.p50Ms,
        label: "p50",
        higherFirst: false,
        format: (v) => fmtMs(v),
        top: 20,
      }),
    [speed],
  );

  const langs = useMemo(() => latestSuiteRun(runs, "decision/multilingual-pool"), [runs]);
  const langOption = useMemo(() => languageSupportOption(langs, deployments), [langs, deployments]);
  const spectrumOption = useMemo(() => spectrumBarOption(spectrum.items), [spectrum]);

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
        <h2>Cost vs speed</h2>
        <p className="muted">
          Measured spend per 1k items against p50 latency (both log scales), hosted deployments with
          a measured price and p50 only — the cheap-and-fast quadrant. The dashed Pareto frontier
          marks systems nothing else is both cheaper and faster than; accuracy rides in the tooltip.
        </p>
        {price.length === 0 ? (
          <p className="muted">No data.</p>
        ) : (
          <EChart option={costSpeedOption} height={360} />
        )}
      </div>

      <div className="card">
        <h2>Ranked — top 20</h2>
        <p className="muted">
          The same canonical runs as the scatters, as ranked bars: most accurate first, then fastest
          first. Everything beyond the top 20 lives in the explorer, sortable per column.
        </p>
        <EChart option={accBarOption} height={520} />
        <EChart option={p50BarOption} height={520} />
      </div>

      <div className="card">
        <h2>Item difficulty</h2>
        <p className="muted">
          Per-question accuracy across the same canonical runs ({spectrum.systems} systems), hardest
          first — the bar is the share of answering systems that got the item right, so short bars
          stumped almost everyone. Only systems that actually answered an item count; adapter errors
          are not attempts. Hover for correct/attempt counts.
        </p>
        {spectrum.items.length === 0 ? (
          <p className="muted">No data.</p>
        ) : (
          <EChart
            option={spectrumOption}
            height={Math.max(320, spectrum.items.length * 12 + 100)}
          />
        )}
      </div>

      <div className="card">
        <h2>Language support</h2>
        <p className="muted">
          Per-language accuracy, one row per deployment — its latest scored decision/
          multilingual-pool run, rows sorted by mean accuracy. Colors only; hover a cell for the
          percentage, counts and run date. Blank cells were not measured.
        </p>
        {langs.size === 0 ? (
          <p className="muted">No data.</p>
        ) : (
          <EChart option={langOption} height={Math.max(320, langs.size * 12 + 120)} />
        )}
      </div>
    </section>
  );
}
