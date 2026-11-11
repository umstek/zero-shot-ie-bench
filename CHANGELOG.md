# Changelog

One entry per release or arrival round. The run history itself lives in
`runs/` (append-only); every table below is generated from the lake by
`bun run render` — nothing here is a hand-edited number.

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
