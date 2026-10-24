// Server-side key resolution — this module must never be imported by client
// code. Values are read from process.env ONLY (the operator exports them
// before `bun run dev`; no .env file is loaded for this app) and are placed
// into the AdapterContext for the duration of one adapter call. Keys are
// never logged, never cached into a response, and never sent to the browser.
//
// Mirrors bench/src/cli.ts authFor: the env var names each provider's
// adapter reads. The catalog providers' `auth` field is the source of truth
// (catalog/providers/<slug>.yaml); the fallback table below matches cli.ts
// for providers that ship without it.
const FALLBACK_PROVIDER_KEYS: Record<string, string[]> = {
  openrouter: ["OPENROUTER_API_KEY"],
  typesafe: ["TYPESAFE_API_KEY"],
  fastino: ["FASTINO_API_KEY"],
  typellm: ["TYPELLM_API_KEY"],
  "cloudflare-workers-ai": ["CLOUDFLARE_AUTH_TOKEN", "CLOUDFLARE_ACCOUNT_ID"],
};

export type ProviderAuth = { id: string; auth: Array<string> | null };

/** Env var NAMES the provider's adapter reads — safe to show in the UI. */
export function expectedKeyNames(provider: ProviderAuth): Array<string> {
  return provider.auth ?? FALLBACK_PROVIDER_KEYS[provider.id] ?? [];
}

/** Resolved secrets for one adapter call. Only present values are included,
 *  exactly like bench/src/cli.ts. */
export function authFor(
  provider: ProviderAuth,
  env: Record<string, string | undefined> = process.env as Record<string, string | undefined>,
): Record<string, string> {
  const out: Record<string, string> = {};
  for (const name of expectedKeyNames(provider)) {
    const value = env[name];
    if (value) out[name] = value;
  }
  return out;
}

/** Which of the provider's keys are missing from the environment — reported
 *  as NAMES only so the UI can badge a deployment as "needs X". */
export function missingKeyNames(
  provider: ProviderAuth,
  env: Record<string, string | undefined> = process.env as Record<string, string | undefined>,
): Array<string> {
  return expectedKeyNames(provider).filter((name) => !env[name]);
}
