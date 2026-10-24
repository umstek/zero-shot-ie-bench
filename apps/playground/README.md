# @zsib/playground

A try-it UI for one question across **both kinds of deployment**
([docs/plan/02](../../docs/plan/02-architecture.md), "Hosted side"): a
**hosted** catalog deployment (called in-process through the same
`bench/src/providers` adapters the bench CLI drives — a provider is wired once)
or a **live local runner** island (proxied server-side over the runner
protocol, so the browser never needs CORS).

`Bun.serve` + React via **HTML imports** — no bundler, no framework extras.

## Run

```sh
# once, from the repo root (links the workspace + installs react)
bun install

# export the keys for the providers you want to try (server-side only), e.g.
export OPENROUTER_API_KEY=sk-or-...
bun run --cwd apps/playground dev       # → http://localhost:4010
```

`PORT` overrides the port. API keys are read from **`process.env` only**.

### Keys note (read this)

- Keys live on the **server**. The browser only ever receives predictions,
  probabilities, latencies and usage — never an auth value. `GET
  /api/deployments` reports the env var **names** a deployment needs (e.g.
  `needs TYPESAFE_API_KEY`), never values.
- No `.env` file is loaded by this app. Bun auto-loads `.env` from the
  **current working directory**, so a `apps/playground/.env` (or a repo-root
  `.env` when starting from the root) would be picked up by Bun itself — but
  the operator contract is: **export keys in the shell**.
  `legacy/.env` is **not** auto-loaded in production (it only loads if you `cd
  legacy/`, which you shouldn't for this app).

| Provider              | Env vars                                         |
| --------------------- | ------------------------------------------------ |
| openrouter            | `OPENROUTER_API_KEY`                             |
| typesafe              | `TYPESAFE_API_KEY`                               |
| fastino               | `FASTINO_API_KEY`                                |
| typellm               | `TYPELLM_API_KEY`                                |
| cloudflare-workers-ai | `CLOUDFLARE_AUTH_TOKEN`, `CLOUDFLARE_ACCOUNT_ID` |

(The catalog `providers/*.yaml` `auth:` field is the source of truth; the
table above mirrors `bench/src/cli.ts` as fallback.)

### Local runners

Start a runner island (e.g. `runners/von`, default `http://127.0.0.1:8030`),
then either type its base URL into the UI or export the same var the bench CLI
uses so it shows as configured:

```sh
export ZSIB_RUNNER_VON=http://127.0.0.1:8030   # island name, dashes → underscores
```

## API surface

| Route                  | Body / params                                                                                                                                                                         | Returns                                                                                                                                                                                                                          |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GET /`                | —                                                                                                                                                                                     | the React page                                                                                                                                                                                                                   |
| `GET /api/deployments` | —                                                                                                                                                                                     | hosted deployments from the catalog (`id`, `modelId`, `adapter`, `endpoint`, `version`, `zdrSafe`, key **names** + configured flags)                                                                                             |
| `POST /api/ask`        | `{deploymentId?, model?, state, question:{type:"choice",options[]}}` — `deploymentId` omitted → first deployment whose keys are all configured; `model` overrides the wire model slug | `{deployment, adapter, model, prediction, prob, error, latencyMs, computeMs, usage}`                                                                                                                                             |
| `GET /api/runners`     | `?baseUrl=<url>` probes that runner; `?runner=<island>` probes the island's `ZSIB_RUNNER_*` URL; no param → islands only                                                              | catalog runner islands (env var, configured URL, models) + optional probe of `/info` (reachable, served models, host fingerprint)                                                                                                |
| `POST /api/runner-ask` | `{baseUrl, model, state, question:{type:"choice",options[]}}`                                                                                                                         | `{baseUrl, model, prediction, prob, error, latencyMs, computeMs, usage}` — proxies to the runner's `/v1/decision` via `bench/src/local/runner-client` (server-side call, latency measured around the client call like the bench) |

Errors: `400` for malformed bodies, `502` for `AdapterError`/`RunnerUnavailableError`
from the upstream call, `500` otherwise — always `{error: string}`.

## Notes

- `bench` imports ride **relative paths** (`../../../bench/src/...`) because
  `bench/package.json` declares no `exports` field yet; once it does, switch
  to `@zsib/bench/...` subpaths. `@zsib/lake` / `@zsib/schema` are normal
  workspace imports.
- `state` is the item's `input.text`; the question maps to the suite's
  decision-track shape. Runners get track `decision` (`/v1/decision`).
- `POST /api/runner-ask` fetches whatever `baseUrl` the client sends — by
  design (it is how you try any island), and acceptable because this is a
  localhost dev tool. Don't expose it publicly without an allowlist.
- Typecheck: `bun run --cwd apps/playground typecheck`. Lint: `bunx biome
  check apps/playground` (covered by the root `bun run lint`).
