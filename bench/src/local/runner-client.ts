// HTTP client for runners/<island> — the local side of the protocol
// (docs/plan/02). Latency is measured client-side here; the server's own
// compute_ms arrives on the response and both are recorded on run items.
import {
  RunnerInfo,
  type RunnerInfoOut,
  type TrackRequestOut,
  TrackResponse,
  type TrackResponseOut,
} from "@zsib/schema";

export type RunnerEndpoint = {
  baseUrl: string; // e.g. http://127.0.0.1:8030
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
};

export class RunnerUnavailableError extends Error {
  constructor(
    message: string,
    readonly cause?: unknown,
  ) {
    super(message);
    this.name = "RunnerUnavailableError";
  }
}

export async function fetchInfo(end: RunnerEndpoint): Promise<RunnerInfoOut> {
  const doFetch = end.fetchImpl ?? fetch;
  try {
    const res = await doFetch(`${end.baseUrl}/info`, {
      signal: AbortSignal.timeout(end.timeoutMs ?? 10_000),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return RunnerInfo.parse(await res.json());
  } catch (err) {
    throw new RunnerUnavailableError(
      `runner at ${end.baseUrl} did not answer /info: ${err instanceof Error ? err.message : err}`,
      err,
    );
  }
}

export async function sendTrack(
  end: RunnerEndpoint,
  track: string,
  req: TrackRequestOut,
): Promise<{ response: TrackResponseOut; latencyMs: number }> {
  const doFetch = end.fetchImpl ?? fetch;
  const t0 = performance.now();
  let res: Response;
  try {
    res = await doFetch(`${end.baseUrl}/v1/${track}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(req),
      signal: AbortSignal.timeout(end.timeoutMs ?? 120_000),
    });
  } catch (err) {
    throw new RunnerUnavailableError(
      `runner at ${end.baseUrl} failed on /v1/${track}: ${err instanceof Error ? err.message : err}`,
      err,
    );
  }
  const latencyMs = performance.now() - t0;
  if (!res.ok) {
    throw new RunnerUnavailableError(`runner HTTP ${res.status} on /v1/${track}`);
  }
  return { response: TrackResponse.parse(await res.json()), latencyMs };
}

// /info host fingerprints are matched to catalog machines by hostname
// (catalog/machines/*.yaml fingerprint_match) — R3: a run names its machine
export function matchMachine(
  info: RunnerInfoOut,
  machines: Array<{ id: string; fingerprint_match: { hostname: string | null } }>,
): string | null {
  const byHost = new Map(
    machines
      .filter((m) => m.fingerprint_match.hostname)
      .map((m) => [m.fingerprint_match.hostname?.toLowerCase() ?? "", m.id]),
  );
  return byHost.get(info.host.hostname.toLowerCase()) ?? null;
}
