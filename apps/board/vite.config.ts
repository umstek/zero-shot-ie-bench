import { existsSync, readFileSync, statSync } from "node:fs";
import { copyFile, mkdir, readdir } from "node:fs/promises";
import { dirname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

const here = dirname(fileURLToPath(import.meta.url));
const buildDir = resolve(here, "../../build");

// The board reads the lake output (build/*.json) under /data with plain fetch().
// - dev: a middleware maps /data/** onto ../../build/** (no publicDir symlink).
// - build: the JSON tree is copied into dist/data so the site stays fully static.
function lakeData(): Plugin {
  return {
    name: "zsib-lake-data",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = (req.url ?? "").split("?")[0] ?? "";
        if (!url.startsWith("/data/")) return next();
        const rel = decodeURIComponent(url.slice("/data/".length)).replaceAll("\\", "/");
        const file = resolve(buildDir, rel);
        if (!file.startsWith(buildDir + sep) || !existsSync(file) || !statSync(file).isFile()) {
          res.statusCode = 404;
          res.end(`not found: ${rel} — run \`bun run lake\` at the repo root to build the lake`);
          return;
        }
        res.setHeader("Content-Type", "application/json; charset=utf-8");
        res.setHeader("Cache-Control", "no-store");
        res.end(readFileSync(file));
      });
    },
    async closeBundle() {
      if (!existsSync(buildDir)) {
        console.warn(
          "[board] ../../build does not exist — dist/data will be empty. Run `bun run lake` at the repo root first.",
        );
        return;
      }
      const copied = await copyJsonTree(buildDir, resolve(here, "dist", "data"));
      console.log(`[board] copied ${copied} data file(s) from build/ into dist/data`);
    },
  };
}

async function copyJsonTree(src: string, dest: string): Promise<number> {
  let count = 0;
  await mkdir(dest, { recursive: true });
  for (const entry of await readdir(src, { withFileTypes: true })) {
    const s = join(src, entry.name);
    const d = join(dest, entry.name);
    if (entry.isDirectory()) {
      count += await copyJsonTree(s, d);
    } else if (entry.name.endsWith(".json")) {
      await copyFile(s, d);
      count += 1;
    }
  }
  return count;
}

export default defineConfig({
  // Absolute base: browser-history deep links (/model/<org>/<name>) must keep
  // resolving /assets/* from the host root. For a project-pages deploy, override
  // this to the repo sub-path (and add an SPA fallback for history deep links).
  base: "/",
  plugins: [react(), lakeData()],
});
