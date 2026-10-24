// Provider adapter registry — catalog deployment.adapter ("openrouter.systemone",
// "workers-ai.clef", …) resolves to a module here. One adapter per API
// *contract*, not per model (docs/plan/03). Each adapter exposes:
//   export const adapter: Adapter  — with the contract's request builders
// Adapters never touch the catalog or runs; the bench CLI drives them.
import type { TrackRequestOut, TrackResponseOut } from "@zsib/schema";

export type AdapterContext = {
  baseUrl: string;
  /** resolved secrets — never logged, never written to runs */
  auth: Record<string, string>;
  endpoint: string;
  /** the wire `model` slug when it differs from the endpoint (e.g. the
   *  pinned typesafe api_slug jev-1.14.0 while the endpoint is a URL) */
  model?: string;
  fetchImpl?: typeof fetch;
};

export type Adapter = {
  id: string; // "<provider>.<contract>"
  /** send one track request to one deployment; returns track-typed results.
   *  Throws AdapterError on transport/HTTP failure so the bench can retry
   *  or record the item error. */
  send: (req: TrackRequestOut, ctx: AdapterContext) => Promise<TrackResponseOut>;
};

export class AdapterError extends Error {
  constructor(
    message: string,
    readonly status: number | null,
    readonly retryAfterMs: number | null,
  ) {
    super(message);
    this.name = "AdapterError";
  }
}
