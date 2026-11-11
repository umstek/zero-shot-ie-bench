// Build-time prerenderer: renders every route to static HTML so crawlers see
// real content (the board's stated purpose is an SEO-friendly static app).
// Runs after `vite build` — reads dist/index.html as the shell, renders each
// route with react-dom/server over a memory-history router, and writes
// dist/<path>/index.html. dist/index.html stays as the SPA fallback for any
// path not prerendered. The browser entry hydrates the same markup.
//
// Run from apps/board: `bun scripts/prerender.ts` (wired into `npm run build`).

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createMemoryHistory, RouterProvider } from "@tanstack/react-router";
import { createElement } from "react";
import { renderToReadableStream } from "react-dom/server";

const here = dirname(fileURLToPath(import.meta.url));
const distDir = resolve(here, "../dist");

// The board fetches lake JSON with root-relative URLs (/data/...); resolve
// those against the built dist tree so the server render sees real data.
const realFetch = globalThis.fetch;
globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
  const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
  if (url.startsWith("/data/")) {
    const file = join(distDir, ...url.replaceAll("\\", "/").split("/").filter(Boolean));
    return new Response(await Bun.file(file), { headers: { "content-type": "application/json" } });
  }
  return realFetch(input, init);
}) as typeof fetch;

// Import after the fetch patch: module code may capture references eagerly.
const { createAppRouter, ROUTE_META } = await import("../src/router");

type Deployment = { model_id: string; display: string };
const deployments: Deployment[] = await Bun.file(join(distDir, "data/deployments.json")).json();
const modelIds = [...new Set(deployments.map((d) => d.model_id))].sort();
const displayOf = new Map(deployments.map((d): [string, string] => [d.model_id, d.display]));

const shell = await Bun.file(join(distDir, "index.html")).text();
const fallbackDescription =
  shell.match(/<meta\s+name="description"\s+content="([^"]*)"/)?.[1] ?? "";

function metaFor(path: string): { title: string; description: string } {
  if (path.startsWith("/model/")) {
    const display = displayOf.get(path.slice("/model/".length)) ?? path;
    return {
      title: `${display} — zsib board`,
      description: `Benchmark runs for ${display}: accuracy, latency and measured price across the decision and NER pools.`,
    };
  }
  return ROUTE_META[path] ?? { title: "zsib — decision board", description: fallbackDescription };
}

// $ in replacement strings is a pattern — always replace through functions.
function renderPage(path: string, html: string): string {
  const meta = metaFor(path);
  return shell
    .replace(/<title>[^<]*<\/title>/, () => `<title>${meta.title}</title>`)
    .replace(
      /<meta\s+name="description"\s+content="[^"]*"/,
      () => `<meta name="description" content="${meta.description}"`,
    )
    .replace(/<div id="root"><\/div>/, () => `<div id="root">${html}</div>`);
}

const routes = ["/", "/explore", "/compare", "/ner", ...modelIds.map((m) => `/model/${m}`)];

let rendered = 0;
for (const path of routes) {
  const router = createAppRouter(createMemoryHistory({ initialEntries: [path] }));
  await router.load(); // resolve matches before rendering, like the SSR docs
  const stream = await renderToReadableStream(createElement(RouterProvider, { router }));
  await stream.allReady; // full content before we snapshot — no fallback HTML
  const html = await new Response(stream).text();
  if (html.length < 500 || html.includes("Switched to client rendering")) {
    throw new Error(`prerender: ${path} rendered badly (${html.length} bytes) — see errors above`);
  }

  const out = resolve(distDir, `.${path}`, "index.html");
  await mkdir(dirname(out), { recursive: true });
  await writeFile(out, renderPage(path, html));
  rendered += 1;
}

console.log(`[board] prerendered ${rendered} route(s) into dist`);
