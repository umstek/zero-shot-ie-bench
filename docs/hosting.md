# Static Hosting (Cloudflare)

The benchmark board (`apps/board`) and browser BYOK tryer (`apps/cloud`) are built and served together as a single static site.

## Domains and Routing

- **Production:** `https://zsib.umstek.com`
  - `/` (plus `/explore`, `/compare`, `/ner`, `/model/*`) → `apps/board`
  - `/try/` → `apps/cloud`
- **Staging:** `https://zsib.staging.umstek.com` (same unified layout: board at `/`, cloud tryer at `/try/`)
- **301 Redirect aliases** → `https://zsib.umstek.com` (preserving path and query string):
  - `zero-shot-ie-bench.umstek.com`
  - `zsieb.umstek.com`
  - `ie-bench.umstek.com`
  - `iebench.umstek.com`

## Build and Asset Configuration

- **Build command:** `bun install && bun run build:site`
- **Output directory:** `dist-site`
- **Wrangler config:** [`wrangler.jsonc`](../wrangler.jsonc) (`name: "zsib"`, `assets.directory: "./dist-site"`, `html_handling: "auto-trailing-slash"`, `not_found_handling: "single-page-application"`).

`bun run build:site` builds the data lake (`bun run lake`), builds and prerenders `apps/board`, builds `apps/cloud`, and runs `scripts/build-site.ts` to assemble `dist-site/` along with `_headers`, `robots.txt`, and `sitemap.xml` (using `SITE_ORIGIN`, defaulting to `https://zsib.umstek.com`).

## Cloudflare Redirect Rule

Configure a Single Redirect Rule in the `umstek.com` zone to forward all alias hostnames to the canonical domain:

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
