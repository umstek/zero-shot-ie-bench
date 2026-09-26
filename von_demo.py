"""One-shot von runner for the web UI — executes inside .venv-von.

von needs transformers 5.x while the app's main venv pins 4.57.6, so the
von tab spawns this script there and talks JSON over stdin/stdout.

stdin:  {"texts": [str, ...], "instructions": str,
         "choices": {label: description-or-null}}
stdout: {"results": [{"choice": str, "probabilities": {label: float},
                      "confidence": float}, ...]}
        or {"error": "Type: message"}
"""

import json
import sys


def main() -> None:
    payload = json.loads(sys.stdin.read())
    try:
        from engines.von_client import load_von_decider

        decide = load_von_decider()
        results = []
        for text in payload["texts"]:
            res = decide(state=text, choices=payload["choices"],
                         instructions=payload["instructions"])
            results.append({
                "choice": res.choice,
                "probabilities": res.probabilities,
                "confidence": res.confidence,
            })
        # ASCII-escaped JSON survives Windows pipes using legacy code pages.
        print(json.dumps({"results": results}))
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))


if __name__ == "__main__":
    main()
