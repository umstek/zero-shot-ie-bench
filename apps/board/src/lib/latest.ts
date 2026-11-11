import type { Run, RunWithTrack } from "../types";

/** The board's canonical run for cross-deployment comparison: the latest
 *  scored decision/mixed-pool run per deployment. Failed/deferred runs never
 *  qualify; same-timestamp ties break on run_id so the pick is deterministic.
 *
 *  Explorer, picks and both compare charts all derive their numbers from this
 *  selection so they can never disagree. deployments.latest_run (latest run of
 *  ANY suite) is deliberately NOT used for comparison — mixing suites fuses
 *  numbers from different runs onto one row. */
export function latestMixedPoolRun<T extends Run | RunWithTrack>(runs: T[]): Map<string, T> {
  const latest = new Map<string, T>();
  for (const run of runs) {
    if (run.suite !== "decision/mixed-pool") continue;
    if (run.status === "failed" || run.status === "deferred") continue;
    if (run.accuracy == null) continue;
    const prev = latest.get(run.deployment);
    const newer =
      !prev ||
      Date.parse(run.started_at) > Date.parse(prev.started_at) ||
      (Date.parse(run.started_at) === Date.parse(prev.started_at) && run.run_id > prev.run_id);
    if (newer) latest.set(run.deployment, run);
  }
  return latest;
}
