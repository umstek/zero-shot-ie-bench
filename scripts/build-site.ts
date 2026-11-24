// Assemble the unified static Cloudflare site into dist-site/:
//   /       -> apps/board/dist/**
//   /try/   -> apps/cloud/dist/**
// Plus Cloudflare _headers, robots.txt, and a dynamically discovered sitemap.xml.
import { existsSync, statSync } from "node:fs";
import { cp, mkdir, readdir, rm, writeFile } from "node:fs/promises";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const DEFAULT_SITE_ORIGIN = "https://zsib.umstek.com";

const CRITICAL_OUTPUT_FILES = [
  "index.html",
  "try/index.html",
  "try/data/deployments.json",
  "data/deployments.json",
  "_headers",
  "robots.txt",
  "sitemap.xml",
] as const;

const HEADERS_CONTENT = [
  "/assets/*",
  "  Cache-Control: public, max-age=31536000, immutable",
  "",
  "/try/assets/*",
  "  Cache-Control: public, max-age=31536000, immutable",
  "",
  "/",
  "  Cache-Control: public, max-age=0, must-revalidate",
  "",
  "/index.html",
  "  Cache-Control: public, max-age=0, must-revalidate",
  "",
  "/try/",
  "  Cache-Control: public, max-age=0, must-revalidate",
  "",
  "/try/index.html",
  "  Cache-Control: public, max-age=0, must-revalidate",
  "",
  "/data/*",
  "  Cache-Control: public, max-age=0, must-revalidate",
  "",
  "/try/data/*",
  "  Cache-Control: public, max-age=0, must-revalidate",
  "",
].join("\n");

export interface AssembleSiteOptions {
  boardDist: string;
  cloudDist: string;
  outDir: string;
  siteOrigin?: string;
}

export interface AssembleSiteResult {
  outDir: string;
  siteOrigin: string;
  routes: string[];
}

function normalizeOrigin(origin?: string): string {
  const raw = origin?.trim();
  const base = raw && raw.length > 0 ? raw : DEFAULT_SITE_ORIGIN;
  return base.replace(/\/+$/, "");
}

function escapeXml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}

async function discoverRoutes(rootDir: string, relDir = ""): Promise<string[]> {
  const dir = relDir.length > 0 ? join(rootDir, relDir) : rootDir;
  const routes: string[] = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      const childRel = relDir.length > 0 ? `${relDir}/${entry.name}` : entry.name;
      routes.push(...(await discoverRoutes(rootDir, childRel)));
    } else if (entry.isFile() && entry.name === "index.html") {
      routes.push(relDir.length > 0 ? `/${relDir}/` : "/");
    }
  }
  return relDir.length === 0 ? routes.sort() : routes;
}

function renderRobotsTxt(siteOrigin: string): string {
  return `User-agent: *\nAllow: /\nSitemap: ${siteOrigin}/sitemap.xml\n`;
}

function renderSitemapXml(siteOrigin: string, routes: string[]): string {
  const urls = routes.map(
    (route) => `  <url>\n    <loc>${escapeXml(`${siteOrigin}${route}`)}</loc>\n  </url>`,
  );
  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ...urls,
    "</urlset>",
    "",
  ].join("\n");
}

function assertDirExists(dirPath: string, label: string): void {
  if (!existsSync(dirPath) || !statSync(dirPath).isDirectory()) {
    throw new Error(`build-site: missing ${label} directory at ${dirPath}`);
  }
}

function verifyCriticalFiles(outDir: string): void {
  const missing = CRITICAL_OUTPUT_FILES.filter((rel) => {
    const filePath = join(outDir, rel);
    return !existsSync(filePath) || !statSync(filePath).isFile();
  });
  if (missing.length > 0) {
    throw new Error(`build-site: missing critical output file(s): ${missing.join(", ")}`);
  }
}

/** Assemble the unified static site directory from built board and cloud bundles. */
export async function assembleSite(options: AssembleSiteOptions): Promise<AssembleSiteResult> {
  const boardDist = resolve(options.boardDist);
  const cloudDist = resolve(options.cloudDist);
  const outDir = resolve(options.outDir);
  const siteOrigin = normalizeOrigin(options.siteOrigin);

  assertDirExists(boardDist, "boardDist");
  assertDirExists(cloudDist, "cloudDist");

  await rm(outDir, { recursive: true, force: true });
  await mkdir(outDir, { recursive: true });

  await cp(boardDist, outDir, { recursive: true });
  await cp(cloudDist, join(outDir, "try"), { recursive: true });

  await writeFile(join(outDir, "_headers"), HEADERS_CONTENT, "utf8");
  await writeFile(join(outDir, "robots.txt"), renderRobotsTxt(siteOrigin), "utf8");

  const routes = await discoverRoutes(outDir);
  await writeFile(join(outDir, "sitemap.xml"), renderSitemapXml(siteOrigin, routes), "utf8");

  verifyCriticalFiles(outDir);
  return { outDir, siteOrigin, routes };
}

if (import.meta.main) {
  const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
  const result = await assembleSite({
    boardDist: join(repoRoot, "apps", "board", "dist"),
    cloudDist: join(repoRoot, "apps", "cloud", "dist"),
    outDir: join(repoRoot, "dist-site"),
    siteOrigin: process.env.SITE_ORIGIN,
  });
  const displayDir = relative(repoRoot, result.outDir) || ".";
  console.log(
    `[site] assembled ${displayDir} (${result.routes.length} routes, origin ${result.siteOrigin})`,
  );
}
