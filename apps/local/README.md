# @zsib/local

The live-local-runner tryer (split out of the old playground). It lists the
catalog's **runner islands** and proxies decision questions to a live runner
over the runner protocol ([docs/plan/02](../../docs/plan/02-architecture.md)),
so the browser never needs CORS. Local-only: **this app holds no credentials
of any kind** — no provider keys, no `.env` lookups, no secrets. Hosted
deployments (bring your own key) moved to `apps/cloud`.

`Bun.serve` + React via **HTML imports** — no bundler, no framework extras.

## Run

```sh
# once, from the repo root (links the workspace + installs react)
bun install

bun run --cwd apps/local dev       # → http://localhost:4010
```

`PORT` overrides the port. The only env vars read are the `ZSIB_RUNNER_*`
base URLs (the same vars the bench CLI uses) — never a provider key.

### Local runners

Start a runner island (e.g. `runners/von`, default `http://127.0.0.1:8030`),
then either type its base URL into the UI or export the same var the bench CLI
uses so it shows as configured:

```sh
export ZSIB_RUNNER_VON=http://127.0.0.1:8030   # island name, dashes → underscores
```

## API surface

| Route                  | Body / params                                                                                                            | Returns                                                                                                                                                                                                                          |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GET /`                | —                                                                                                                        | the React page                                                                                                                                                                                                                   |
| `GET /api/runners`     | `?baseUrl=<url>` probes that runner; `?runner=<island>` probes the island's `ZSIB_RUNNER_*` URL; no param → islands only | catalog runner islands (env var, configured URL, models) + optional probe of `/info` (reachable, served models, host fingerprint)                                                                                                |
| `POST /api/runner-ask` | `{baseUrl, model, state, question:{type:"choice",options[]}}`                                                            | `{baseUrl, model, prediction, prob, error, latencyMs, computeMs, usage}` — proxies to the runner's `/v1/decision` via `bench/src/local/runner-client` (server-side call, latency measured around the client call like the bench) |

Errors: `400` for malformed bodies, `502` for `RunnerUnavailableError`
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
- Cloud models with your own API keys: `apps/cloud` (static, browser-side).
- Typecheck: `bun run --cwd apps/local typecheck`. Lint: `bunx biome
  check apps/local` (covered by the root `bun run lint`).
