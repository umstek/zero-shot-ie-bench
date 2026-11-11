import { use } from "react";
import type { HostedDeployment, ProviderRow } from "../types";

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

export interface CloudData {
  deployments: HostedDeployment[];
  providers: ProviderRow[];
}

async function loadCloudData(): Promise<CloudData> {
  const [allDeployments, providers] = await Promise.all([
    fetchJson<HostedDeployment[]>("/data/deployments.json"),
    fetchJson<ProviderRow[]>("/data/providers.json"),
  ]);
  const deployments = allDeployments
    .filter((d) => d.kind === "hosted")
    .sort((a, b) => (a.display ?? a.deployment_id).localeCompare(b.display ?? b.deployment_id));
  return { deployments, providers };
}

let cloudPromise: Promise<CloudData> | null = null;

export function cloudData(): Promise<CloudData> {
  cloudPromise ??= loadCloudData();
  return cloudPromise;
}

/** Suspends until the lake JSON is loaded — wrap the tree in <Suspense>. */
export function useCloudData(): CloudData {
  return use(cloudData());
}
