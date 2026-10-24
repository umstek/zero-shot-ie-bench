"""The gliner island's FastAPI app: the runner protocol over the registry.

The protocol (runners/_protocol, imported as the `runner_protocol` path
dependency) owns /healthz, /info and the POST /v1/<track> routes; this
module only wires the island's name, its model table (models/__init__.py)
and the per-item predict dispatch.

`host_fingerprint` is re-exported for operators: `python -m gliner_runner.server`
prints it before serving so a machine can be matched to its catalog
machine_id without scraping /info.

(The import package is `gliner_runner`, not `gliner`: the PyPI classic
gliner package — the relex checkpoint's runtime — owns that top-level name.)

Start: uv run uvicorn gliner_runner.server:app --port 8009
"""

from __future__ import annotations

from runner_protocol import create_app, host_fingerprint  # noqa: F401 (re-exported)

from gliner_runner.models import SERVED_MODELS, predict

RUNNER = "gliner"

# GLiREL is relation-only and relex also claims relation, but the protocol's
# route set here is the island's bench surface: /v1/relation arrives when
# the shared route set grows (see README "Known gap").
TRACKS = ["ner", "classify", "decision"]

app = create_app(runner=RUNNER, models=SERVED_MODELS, predict=predict,
                 tracks=TRACKS, ui=None)


def main() -> None:
    """`python -m gliner_runner.server`: serve on :8009 after printing the
    host fingerprint the bench matches against catalog/machines/."""
    import json

    print(json.dumps(host_fingerprint(), indent=2))
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8009)


if __name__ == "__main__":
    main()
