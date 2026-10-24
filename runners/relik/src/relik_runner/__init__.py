"""relik runner island — relik-ie ReLiK on the relik 1.0.7 CPU stack.

Serves the retriever-reader relation-extraction pipeline (relik-ie/
relik-relation-extraction-small, the benched catalog deployment) and its
joint NER+relation sibling (relik-ie/relik-relation-extraction-ner-small)
behind the shared runner protocol: `POST /v1/ner` and `POST /v1/relation`.
Pipelines load lazily from the HF cache on the first predict — never at
import, never in tests unless already cached.

This is the repo's only Python 3.11 island: relik 1.0.7 pins torch==2.3.1,
which has no 3.13 wheels. Importing `relik` at all needs the Windows csv
shim carried over from legacy engines/relik_client.py — it lives in
`relik_runner.relik` and installs at that module's import.

Entry points: `relik_runner.relik` (the engine, ported from legacy
engines/relik_client.py + the serialization of legacy demos/relik_demo.py)
and `relik_runner.server` (the FastAPI app, `uvicorn relik_runner.server:app`).
"""

__all__ = ["relik", "server"]
