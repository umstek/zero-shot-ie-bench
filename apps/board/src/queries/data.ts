import { use } from "react";
import type { Deployment, Run, RunWithTrack } from "../types";

// Promise cache keyed by URL — survives re-renders; re-created on HMR, which is fine.
const cache = new Map<string, Promise<unknown>>();

function fetchJson<T>(path: string): Promise<T> {
  const cached = cache.get(path);
  if (cached) return cached as Promise<T>;
  const p = fetch(path).then((res) => {
    if (!res.ok) throw new Error(`${path}: HTTP ${res.status} — is build/*.json present?`);
    return res.json() as Promise<T>;
  });
  cache.set(path, p);
  return p;
}

export interface BoardData {
  deployments: Deployment[];
  /** decision + ner runs merged, each tagged with its track. */
  runs: RunWithTrack[];
}

async function loadBoardData(): Promise<BoardData> {
  const [deployments, decisionRuns, nerRuns] = await Promise.all([
    fetchJson<Deployment[]>("/data/deployments.json"),
    fetchJson<Run[]>("/data/decision.runs.json"),
    fetchJson<Run[]>("/data/ner.runs.json"),
  ]);
  const runs: RunWithTrack[] = [
    ...decisionRuns.map((r) => ({ ...r, track: "decision" as const })),
    ...nerRuns.map((r) => ({ ...r, track: "ner" as const })),
  ];
  return { deployments, runs };
}

let boardPromise: Promise<BoardData> | null = null;

export function boardData(): Promise<BoardData> {
  boardPromise ??= loadBoardData();
  return boardPromise;
}

/** Suspends until the lake JSON is loaded — wrap the route tree in <Suspense>. */
export function useBoardData(): BoardData {
  return use(boardData());
}
