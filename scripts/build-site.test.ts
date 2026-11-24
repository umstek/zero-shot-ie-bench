import { afterEach, describe, expect, test } from "bun:test";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { assembleSite } from "./build-site";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "zsib-site-"));
  tempDirs.push(dir);
  return dir;
}

async function writeFixture(baseDir: string, files: Record<string, string>): Promise<void> {
  for (const [relPath, content] of Object.entries(files)) {
    const target = join(baseDir, relPath);
    await mkdir(dirname(target), { recursive: true });
    await writeFile(target, content, "utf8");
  }
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
});

describe("assembleSite", () => {
  test("merges board and cloud dists, writes _headers, robots.txt, and sorted XML-escaped sitemap.xml", async () => {
    const root = await makeTempDir();
    const boardDist = join(root, "board-dist");
    const cloudDist = join(root, "cloud-dist");
    const outDir = join(root, "dist-site");

    await writeFixture(boardDist, {
      "index.html": "<html>board</html>",
      "assets/board-abc123.js": "console.log('board');",
      "data/deployments.json": "[]",
      "explore/index.html": "<html>explore</html>",
      "compare/index.html": "<html>compare</html>",
      "ner/index.html": "<html>ner</html>",
      "model/acme/zero&one/index.html": "<html>model</html>",
    });

    await writeFixture(cloudDist, {
      "index.html": "<html>cloud</html>",
      "assets/cloud-xyz789.js": "console.log('cloud');",
      "data/deployments.json": "[]",
    });

    // Pre-existing stale file should be removed when outDir is cleaned.
    await writeFixture(outDir, { "stale.txt": "old" });

    const result = await assembleSite({ boardDist, cloudDist, outDir });

    expect(existsSync(join(outDir, "stale.txt"))).toBe(false);
    expect(await readFile(join(outDir, "index.html"), "utf8")).toBe("<html>board</html>");
    expect(await readFile(join(outDir, "assets/board-abc123.js"), "utf8")).toBe(
      "console.log('board');",
    );
    expect(await readFile(join(outDir, "try/index.html"), "utf8")).toBe("<html>cloud</html>");
    expect(await readFile(join(outDir, "try/assets/cloud-xyz789.js"), "utf8")).toBe(
      "console.log('cloud');",
    );
    expect(await readFile(join(outDir, "try/data/deployments.json"), "utf8")).toBe("[]");
    expect(await readFile(join(outDir, "data/deployments.json"), "utf8")).toBe("[]");

    expect(result.allowIndexing).toBe(true);
    const headers = await readFile(join(outDir, "_headers"), "utf8");
    expect(headers).toContain("/assets/*\n  Cache-Control: public, max-age=31536000, immutable");
    expect(headers).toContain(
      "/try/assets/*\n  Cache-Control: public, max-age=31536000, immutable",
    );
    for (const path of ["/", "/index.html", "/try/", "/try/index.html", "/data/*", "/try/data/*"]) {
      expect(headers).toContain(`${path}\n  Cache-Control: public, max-age=0, must-revalidate`);
    }
    expect(headers).not.toContain("X-Robots-Tag");

    const robots = await readFile(join(outDir, "robots.txt"), "utf8");
    expect(robots).toBe("User-agent: *\nAllow: /\nSitemap: https://zsib.umstek.com/sitemap.xml\n");

    const expectedRoutes = [
      "/",
      "/compare/",
      "/explore/",
      "/model/acme/zero&one/",
      "/ner/",
      "/try/",
    ];
    expect(result.routes).toEqual(expectedRoutes);

    const sitemap = await readFile(join(outDir, "sitemap.xml"), "utf8");
    const locs = [...sitemap.matchAll(/<loc>([^<]+)<\/loc>/g)].map((m) => m[1]);
    expect(locs).toEqual([
      "https://zsib.umstek.com/",
      "https://zsib.umstek.com/compare/",
      "https://zsib.umstek.com/explore/",
      "https://zsib.umstek.com/model/acme/zero&amp;one/",
      "https://zsib.umstek.com/ner/",
      "https://zsib.umstek.com/try/",
    ]);
  });

  test("normalizes custom siteOrigin by stripping trailing slashes", async () => {
    const root = await makeTempDir();
    const boardDist = join(root, "board-dist");
    const cloudDist = join(root, "cloud-dist");
    const outDir = join(root, "dist-site");

    await writeFixture(boardDist, {
      "index.html": "<html>board</html>",
      "data/deployments.json": "[]",
    });
    await writeFixture(cloudDist, {
      "index.html": "<html>cloud</html>",
      "data/deployments.json": "[]",
    });

    const result = await assembleSite({
      boardDist,
      cloudDist,
      outDir,
      siteOrigin: "https://zsib.staging.umstek.com///",
    });

    expect(result.siteOrigin).toBe("https://zsib.staging.umstek.com");
    expect(result.allowIndexing).toBe(false);
    expect(await readFile(join(outDir, "robots.txt"), "utf8")).toBe("User-agent: *\nDisallow: /\n");
    expect(await readFile(join(outDir, "_headers"), "utf8")).toContain(
      "/*\n  X-Robots-Tag: noindex, nofollow",
    );
    const sitemap = await readFile(join(outDir, "sitemap.xml"), "utf8");
    expect(sitemap).toContain("<loc>https://zsib.staging.umstek.com/</loc>");
    expect(sitemap).toContain("<loc>https://zsib.staging.umstek.com/try/</loc>");
  });

  test("throws when critical output files are missing", async () => {
    const root = await makeTempDir();
    const boardDist = join(root, "board-dist");
    const cloudDist = join(root, "cloud-dist");
    const outDir = join(root, "dist-site");

    await writeFixture(boardDist, { "index.html": "<html>board</html>" });
    await writeFixture(cloudDist, { "index.html": "<html>cloud</html>" });

    await expect(assembleSite({ boardDist, cloudDist, outDir })).rejects.toThrow(
      /missing critical output file\(s\)/,
    );
  });
});
