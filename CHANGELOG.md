# Changelog

One entry per release or arrival round. The run history itself lives in
`runs/` (append-only); every table below is generated from the lake by
`bun run render` — nothing here is a hand-edited number.

## v1.0.1 (2026-11-25)

- Overview page (`apps/board`) and `README.md` now explain the model types covered (decision models, zero-shot classifiers, rerankers, NER & relation extractors), input modalities (text benchmarked; image/audio/video vendor-claimed), and a benchmark glossary ([PR #57](https://github.com/umstek/zero-shot-ie-bench/pull/57)).
- Added `AGENTS.md` checklist for adding or updating models, versions, and tracks.

## v1.0.0 (2026-11-24)

- First `v1.0.0` release after the TypeScript + Bun monorepo restructure (`packages/schema`, `packages/lake`, `bench`, `runners/*`, `apps/board`, `apps/cloud`, `apps/local`).
- Unified static Cloudflare deployment at [zsib.umstek.com](https://zsib.umstek.com) (board with a new Overview introduction landing page at `/` plus `/picks`, `/explore`, `/compare`, `/ner`, `/model/*`, and browser BYOK tryer at `/try/`) with 301 redirect aliases and `noindex` staging at [zsib.staging.umstek.com](https://zsib.staging.umstek.com).
- Migrated hosting to Cloudflare's `cf` CLI (`cloudflare.config.ts`) and GitHub Actions (`main` → staging, published GitHub Releases → production).

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

## 2026-11-11 — new-system arrival round

- **Hosted catalog** (+5 deployments, benched same day): Morph Reflex,
  Liquid D1.1, Drex v2.0, Drex Max Finance, Span-02 — all on the existing
  `openrouter.systemone` adapter. A sixth arrival, TypeSafe's Jev Router,
  was pulled back: the OpenRouter web listing resolves but the API route
  400s (stub) — tracked in docs/pending-access.md.
- **Gate override #2** (recorded, per the adding-a-model recipe): the
  gate-12 screening was skipped for this round at the user's call —
  arrivals went straight to mixed-pool + multilingual-pool. Results —
  mixed: reflex 45/48, d1.1 44/48, drex-v2.0 / drex-max-finance / span-02
  43/48; multilingual: reflex / d1.1 / drex-max-finance 54/54, drex-v2.0 /
  span-02 53/54; 0 errors anywhere, p50 604–856 ms. span-02's open weights
  (ungated, 15.8B) settled its params_b.
- **Local islands** (+3 servable): laya-multilingual (gliner island, one
  registry row — laya==0.4.14 loads it generically), decider-2b
  (systemone-local; decider.py refactored to per-checkpoint machinery),
  decima:agent (small-decisions Ollaya table; needs the 0.13.0+ daemon to
  run). GLiDE-0.4B enters the catalog with its island module deferred (its
  Python API ships inside the HF repo, not on PyPI — see
  docs/pending-access.md).
- **docs/pending-access.md**: the keys/access list (10 vendors + the
  Jev Router stub) plus engineering deferrals and next-round CPU-small
  candidates from the 2026-11-11 research fan-out. HF gating check across
  all 36 researched repos: only MoLeMo-Lab/mojev is gated (auto) — and it
  is already benched via a pinned commit; nothing this round needs an
  unlock.

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
