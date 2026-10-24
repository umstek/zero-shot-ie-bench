# runner-protocol

Shared library imported by every `runners/<island>` as a path dependency
(`[tool.uv.sources] runner-protocol = { path = "../_protocol", editable = true }`).

It provides:

- `create_app(...)` — FastAPI factory exposing `GET /healthz`, `GET /info`,
  `POST /v1/<track>` (wire contract from `catalog/schema/*.json`), with an
  optional Gradio UI mounted at `/` in the same process.
- `host_fingerprint()` — CPU / RAM / GPU / OS fingerprint for `/info`
  (stdlib only, no psutil).
- `schema_models` — Pydantic models **generated** from
  `catalog/schema/{runner-info,track-request,track-response}.json`
  (Zod 4 source of truth: `packages/schema/src/protocol.ts`). Regenerate with
  `uv run python tools/generate_schema_models.py` — never edit by hand.

Independent uv project, own venv, Python 3.13. CI tests the protocol with a
stub model only — no weights, no network.
