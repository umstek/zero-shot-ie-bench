// Structural types of the lake output the cloud tryer reads (build/*.json,
// served/copied under /data by the vite plugin). Additive lake fields this
// app does not use (runs, feasibility, families…) are simply absent here.

/** One hosted deployment row of build/deployments.json (kind: "hosted"). */
export type HostedDeployment = {
  deployment_id: string;
  model_id: string;
  display?: string;
  zdr_safe: boolean | null;
  family: string;
  org: string;
  status: string;
  kind: "hosted";
  version: string;
  provider: string;
  /** catalog deployment.adapter — resolves through bench/src/providers */
  adapter: string;
  /** the wire endpoint: an API path/URL or a provider model slug */
  endpoint: string;
  /** the pinned wire model slug, when the endpoint is a URL (see ask.ts) */
  api_slug?: string;
};

/** One row of build/providers.json — the BYOK surface of a provider. */
export type ProviderRow = {
  id: string;
  name: string;
  base_url: string | null;
  /** env var NAMES the adapter reads — names only, never values */
  auth: Array<string> | null;
};
