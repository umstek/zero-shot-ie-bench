"""One-shot relik runner for the web UI — executes inside .venv-relik.

relik needs its own dependency set (torch 2.3.1, faiss-cpu), so the app's
main venv keeps transformers 4.57.6 and the relik tab spawns this script
in `.venv-relik`, talking JSON over stdin/stdout.

Importing engines.relik_client first installs the Windows csv shim that
relik's import otherwise trips over (SapienzaNLP/relik#39).

stdin:  {"texts": [str, ...]}
stdout: {"results": [{"text": str,
                      "triplets": [{"subject": {start, end, label, text},
                                    "relation": str,
                                    "object": {start, end, label, text},
                                    "confidence": float}, ...],
                      "spans": [{"start", "end", "label", "text"}, ...]},
                     ...]}
        or {"error": "Type: message"}
"""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
import os
import sys

# the shim must run before relik is imported anywhere in this process
import engines.relik_client  # noqa: F401  (installs the csv field-size shim)


def span_to_dict(span) -> dict:
    """Serialize a relik Span (NamedTuple) via its own fields."""
    return {"start": span.start, "end": span.end,
            "label": span.label, "text": span.text}


def main() -> None:
    payload = json.loads(sys.stdin.read())
    try:
        from engines.relik_client import load_relik

        # relik prints its banner and binds its loggers to stdout at load
        # time, and its dataloader children inherit the OS-level stdout —
        # a sys.stdout redirect catches neither, so fd 1 moves onto stderr
        # for load + inference to keep this script's stdout pure JSON.
        saved_stdout = os.dup(1)
        try:
            os.dup2(2, 1)
            relik = load_relik()
            # one batched call for every line in the same process
            outs = relik(payload["texts"])
        finally:
            os.dup2(saved_stdout, 1)
            os.close(saved_stdout)
        if not isinstance(outs, list):
            outs = [outs]
        results = []
        for text, out in zip(payload["texts"], outs):
            results.append({
                "text": text,
                "triplets": [{
                    "subject": span_to_dict(t.subject),
                    "relation": t.label,
                    "object": span_to_dict(t.object),
                    "confidence": t.confidence,
                } for t in out.triplets],
                "spans": [span_to_dict(s) for s in out.spans],
            })
        # ASCII-escaped JSON survives Windows pipes using legacy code pages.
        print(json.dumps({"results": results}))
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))


if __name__ == "__main__":
    main()
