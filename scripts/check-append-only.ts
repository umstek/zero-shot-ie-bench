// CI append-only gate (docs/plan/02 invariant 3): no PR may modify or
// delete an existing runs/ file. Run from repo root on a PR branch:
//   bun run scripts/check-append-only.ts <merge-base-sha>
// Files under runs/ that already exist in the base must be byte-identical.
import { join } from "node:path";

const repoRoot = join(import.meta.dir, "..");
const base = process.argv[2];
if (!base) {
  console.error("usage: check-append-only.ts <merge-base-sha>");
  process.exit(2);
}

const diff = await Bun.$`git diff --name-status ${base}...HEAD -- runs/`.cwd(repoRoot).text();
const violations = diff
  .split("\n")
  .map((line) => line.trim())
  .filter((line) => line.length > 0)
  // status M/D on an existing run file breaks append-only; A (add) is the
  // only legal change, plus renames away from runs/
  .filter((line) => /^(M|D|R)/.test(line));

if (violations.length > 0) {
  console.error("FAIL runs/ is append-only — these changes are not allowed:");
  for (const v of violations) console.error(`  ${v}`);
  process.exit(1);
}
console.log("OK runs/ append-only (adds only)");
