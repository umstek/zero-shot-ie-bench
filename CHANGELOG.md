# Changelog

One entry per release or arrival round. The run history itself lives in
`runs/` (append-only); every table below is generated from the lake by
`bun run render` — nothing here is a hand-edited number.

## 2026-11-12 — board polish round

- **`apps/board` prerenders every route at build time** — 85 pages (picks,
  explorer, charts, NER, 81 model pages) with per-route `<title>`/description
  shared between the prerenderer and SPA navigation, and the browser entry now
  hydrates instead of re-rendering, so crawlers see real content and the page
  doesn't blank on load. A data-as-of stamp in the footer shows the newest run
  date.
- Charts page gained three cards over the same canonical latest-scored
  mixed-pool runs as the explorer: a **cost-vs-speed** scatter with its own
  Pareto frontier, **ranked top-20** accuracy/latency bars, and an
  **item-difficulty** spectrum fed by a new `build/spectrum.json` (per-question
  accuracy across systems; adapter failures are not attempts).
- The lake records **measured browser-CORS verdicts per provider** (dated,
  credential-free OPTIONS preflights against the real POST paths; only
  OpenRouter currently passes) and `apps/cloud` badges its picker accordingly.
- **Capability matrix ported from v0** into 49 model catalogs as an optional
  11-key schema block; families the v0 app never charted stay unset.
- The NER trio (GLiNER-relex, ReLiK, GLiREL) was attempted on
  `ner/graded-pool` and **deferred** — the runner islands' Python environments
  are gone from this machine; the append-only deferred rows record why.
- CI builds the board bundle + prerender as a gate; determinism remains
  unbuilt (needs a new flat suite first).

## 2026-11-11 — app split

- `apps/playground` split into two apps and deleted. **`apps/local`** keeps the
  live-local-runner half (`/api/runners`, `/api/runner-ask`) and holds no
  credentials of any kind. **`apps/cloud`** is the new hosted half: a static
  Vite + React page where the bench adapters run in the browser with the user's
  own keys — React state only, nothing stored or logged; a network failure
  surfaces a CORS hint. The lake now emits `adapter`/`endpoint`/`api_slug` on
  hosted deployment rows plus `build/providers.json` (base_url + env var
  names, never values) to feed it.

## 2026-10-24 — restructure (phases 0–4)

- **Phase 0**: v0 frozen at tag `v0-final`; the flat repo moved under
  `legacy/` (deleted at the end of this restructure).
- **Phase 1**: Bun + TypeScript 7 monorepo spine; `packages/schema` (Zod 4)
  generating `catalog/schema/`; the whole v0 system catalog extracted —
  81 systems / 78 benched / 38 families, matching the v0-final README
  headline exactly; providers, tracks, languages, `machines/seoul.yaml`.
- **Phase 2**: suites (mixed-pool 48, gate-12, multilingual 6×9 parallel,
  canaries, NER 18) with deterministic hashes; 171 `legacy-v0` runs
  imported from the v0 results; `packages/lake` (plain TS) + `bench render`
  regenerating the README tables — cell-for-cell identical to v0-final
  (193 rows). v0 editorial quirks are carried as documented bookkeeping
  (arrival-order ranks, verbatim latency cells, half-even percentages,
  per-endpoint † flags, one table-vs-data drift).
- **Phase 3**: `runners/_protocol` + six islands (gliner,
  systemone-local, von, relik, small-decisions, nanodiff), one
  independent uv project each, Pyrefly clean, weight-gated tests; real
  weights exercised on von, relik and small-decisions' engines.
- **Phase 4**: the TS bench — six provider adapters, decision/NER scorers,
  `run`/`canary`/`doctor` CLI, deferred-run feasibility path, bench-owned
  retries. **First rebench history**: 10 of 11 acceptance targets re-ran
  the mixed pool within noise of v0 (elephant-1 exactly 89.6% through its
  free-tier pacing); clef 401 — no `CLOUDFLARE_AUTH_TOKEN` on the account
  (v0 rode a cf CLI session), a truthful item-error state awaiting a
  token. First drift data points recorded: d1-ultra −4.2pp, glide −8.3pp,
  typellm −8.3pp, solar-decide +2.1pp, gpt-6-luna +2.1pp.

## v0 history (2026-09-20 → 2026-10-24)

The narrative record of the v0 era — arrival rounds, trial gates, culled
systems — is preserved verbatim at tag `v0-final` (README + TRIALS.md).
Highlights: 81 systems across 38 families benched from 2026-09-20
(7-system first run) through 2026-10-24 (Elephant-1, GPT-6 Luna
Decisions); the 12-question trial gate with one recorded override
(jev-1.14.0, 2026-10-17); the multilingual pool at 54 texts × 9
languages with popular/medium/rare tiers; the elephant-1 free-tier and
luna ZDR-toggle notes. Every number is citable at the tag.
