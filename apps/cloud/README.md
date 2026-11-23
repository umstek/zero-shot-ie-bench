# @zsib/cloud

The BYOK tryer for the bench's **hosted** deployments: **Vite 8 + React 19**,
one static page, zero backend. You paste **your own** API key for the selected
deployment's provider; the bench adapter (`bench/src/providers`) runs **in
your browser** and sends the request straight to the provider.

**Your key stays in this browser tab** — it is sent only to the provider,
never to us (there is no us: this page is static). Keys live in React state
only: no `localStorage`, no logging, never in the URL. Reloading clears them
by design.

Local runners have their own app: `apps/local` (proxied server-side, no keys).

## Commands

Run from the repo root (Bun workspaces; this app is covered by the `apps/*` glob):

```sh
bun install                       # once, at the repo root
bun run lake                      # build/*.json first — the app reads it at build time
bun run --cwd apps/cloud dev      # http://localhost:5173
bun run --cwd apps/cloud build    # → apps/cloud/dist/
bun run --cwd apps/cloud preview  # serve the built dist/
bun run --cwd apps/cloud typecheck
```

## Data expectations

Same mechanism as `apps/board`: the app fetches `./data/*.json` at runtime.

- **dev**: a Vite middleware in `vite.config.ts` serves `../../build/**` under
  `/data/**`.
- **build**: `vite build` copies the `build/**` JSON tree into `dist/data/`
  via the same plugin's `closeBundle` hook.

| File                     | Used by                                                             |
| ------------------------ | ------------------------------------------------------------------- |
| `build/deployments.json` | the deployment picker (hosted rows: adapter, endpoint, pinned slug) |
| `build/providers.json`   | per-provider `base_url` + the env var **names** the adapter reads   |

`providers.json` and the hosted fields carry **names only** — the user's keys
are typed into the browser at ask time and never leave it.

## CORS reality

The adapter call is a plain `fetch` from the browser. OpenRouter allows direct
browser calls; other providers may not send CORS headers. When the call fails
as a network error (`fetch` `TypeError`), the UI shows a hint that the
provider may not allow direct browser calls and may work through the local
app or CLI instead. No provider is pre-excluded.

## Notes

- `bench` imports ride **relative paths** (`../../../bench/src/providers/…`,
  the same trick the old playground's server used) because
  `bench/package.json` declares no `exports` field yet. Both modules are
  browser-safe: plain `fetch`, no `node:`/`fs`/`process`/`Bun` imports; their
  `@zsib/schema` imports are type-only and erased at build.
- Wire-model rule (same as `bench/src/cli.ts`): a URL endpoint selects by the
  pinned `api_slug` from the catalog; a slug endpoint **is** the wire model.
- Deploy target: any static host / Cloudflare Pages (`base: "./"`, relative —
  no router, so no SPA fallback needed).
- Typecheck: `bun run --cwd apps/cloud typecheck`. Lint: covered by the root
  `bun run lint`.
