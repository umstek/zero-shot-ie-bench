# Static Hosting (Cloudflare)

The benchmark board (`apps/board`) and browser BYOK tryer (`apps/cloud`) are built and served together as a single static site via Cloudflare Workers Static Assets and the `cf` CLI ([`cloudflare.config.ts`](../cloudflare.config.ts)).

## Domains and Routing

- **Production:** `https://zsib.umstek.com` (Worker `zsib`)
  - `/` (plus `/explore`, `/compare`, `/ner`, `/model/*`) → `apps/board`
  - `/try/` → `apps/cloud`
- **Staging:** `https://zsib.staging.umstek.com` (Worker `zsib-staging`, same unified layout)
- **301 Redirect aliases** → `https://zsib.umstek.com` (preserving path and query string):
  - `zero-shot-ie-bench.umstek.com`
  - `zsieb.umstek.com`
  - `ie-bench.umstek.com`
  - `iebench.umstek.com`

## Build and Asset Configuration

- **Build command:** `bun install && bun run build:site`
- **Output directory:** `dist-site`
- **Cloudflare config:** [`cloudflare.config.ts`](../cloudflare.config.ts) (`worker.assets.htmlHandling: "auto-trailing-slash"`, `worker.assets.notFoundHandling: "single-page-application"`, mode-switched Worker name and custom domain).

`bun run build:site` builds the data lake (`bun run lake`), builds and prerenders `apps/board`, builds `apps/cloud`, and runs `scripts/build-site.ts` to assemble `dist-site/` along with `_headers`, `robots.txt`, and `sitemap.xml` (using `SITE_ORIGIN`, defaulting to `https://zsib.umstek.com`).

When `SITE_ORIGIN` is anything other than `https://zsib.umstek.com` (such as staging), `scripts/build-site.ts` automatically emits `Disallow: /` in `robots.txt` and `X-Robots-Tag: noindex, nofollow` in `_headers`.

> **Runtime note:** `cf` requires Node.js (`^22.23.0 || ^24.22.0 || ^26.12.0`) to load `cloudflare.config.ts` — always invoke it via `npx cf ...` (not `bunx cf`, since loading `cloudflare.config.ts` is not supported on Bun). Cloudflare Workers Builds defaults to Node.js 24.

## Cloudflare Workers Builds

- **Production (`zsib`, branch `main`):**
  - **Build command:** `bun install && bun run build:site`
  - **Deploy command:** `npx cf deploy ./dist-site`
  - **Domain:** `zsib.umstek.com` (declared in `cloudflare.config.ts`)
- **Staging (`zsib-staging`, connected to your staging branch or manual trigger):**
  - **Build command:** `bun install && bun run build:site`
  - **Deploy command:** `npx cf deploy ./dist-site --mode staging`
  - **Build environment variable:** `SITE_ORIGIN=https://zsib.staging.umstek.com`
  - **Domain:** `zsib.staging.umstek.com` (declared in `cloudflare.config.ts`)

## Cloudflare Redirect Rule

Ensure the 4 alias hostnames have **Proxied** DNS records in the `umstek.com` zone (for example, `AAAA` `100::` with Proxied enabled) so Cloudflare receives requests for them, then configure a Single Redirect Rule in the `umstek.com` zone to forward all alias hostnames to the canonical domain:

- **Filter expression:**
  ```text
  (http.host in {"zero-shot-ie-bench.umstek.com" "zsieb.umstek.com" "ie-bench.umstek.com" "iebench.umstek.com"})
  ```
- **Target URL (Dynamic expression):**
  ```text
  concat("https://zsib.umstek.com", http.request.uri.path)
  ```
- **Status code:** `301`
- **Preserve query string:** Enabled
