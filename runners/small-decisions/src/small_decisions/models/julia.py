"""SupersonicLabs' Julia 1 (typed-decision model, 144M) — ported, NOT
registered on /info (no catalog yaml exists yet; there is no catalog
identity to serve — models/__init__ keeps it out of MODELS until one
lands).

Ported from legacy/engines/julia_client.py: the runtime package (`julia`)
ships inside the SupersonicLabs/Julia-1 Hugging Face repo itself; the
complete snapshot (fp32 weights, 550.5 MiB, plus the runtime) is cloned
next to the repo (JULIA_HOME overrides) and imported via sys.path — the
house pattern. Upstream's pyproject pins transformers >=5.0,<5.1 while
this island keeps 5.17.0; the one real drift is handled in load():

- julia.router.encoder.specialize_decision_encoder monkey-patches an
  optional fast forward over ModernBertModel._update_attention_mask, an
  API transformers 5.17 removed — without the guard every predict()
  raises AttributeError. The patch is skipped so the stock forward runs;
  the runtime's own parity test asserts both paths produce exactly equal
  outputs (atol=0). Verified on 5.17: end-to-end inference is sound.

Upstream only reads JULIA_CPU_THREADS on its CUDA engine path — the CPU
FastEngine this snapshot runs never calls torch.set_num_threads — so
load() applies it itself before load_model(): 16 torch threads by default
(torch's own default on the 16-core bench machine), JULIA_CPU_THREADS
overrides when set.

Catalog identity when one lands: MODEL_ID "SupersonicLabs/Julia-1".
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any

from . import _base, _systemone

MODEL_ID = "SupersonicLabs/Julia-1"

# No catalog yaml yet — the version id below is the placeholder shape a
# future entry would take; unused while ENTRY is not registered.
VERSION_CANDIDATE = "julia-1@unversioned"
RUNTIME = {"engine": "torch", "dtype": "fp32", "quant": None, "device": "cpu"}

_ENGINE: list[Any] = []
_LOCK = threading.Lock()


def model_home() -> str:
    """Snapshot directory: JULIA_HOME, default ../Julia-1 from the repo root."""
    return os.environ.get(
        "JULIA_HOME",
        str(_base.REPO_DIR.parent / "Julia-1"),
    )


def available() -> bool:
    """The snapshot (its julia/ runtime package) is locally present."""
    return os.path.isdir(os.path.join(model_home(), "julia"))


def _pop_stale_modules() -> None:
    """Drop any earlier snapshot's modules so a re-load with a different
    JULIA_HOME imports the fresh one (sys.modules hygiene before the
    import below)."""
    for stale in ("julia", "julia.router", "julia.router.encoder"):
        sys.modules.pop(stale, None)


def load_engine(device: str = "cpu", **kwargs: Any) -> Any:
    """julia.load_model over the local snapshot with the 5.17 compat guard.

    Heavy imports stay inside (offline tests import this module freely).
    strict_encoding=True and head_length=512 follow the model card; the
    remaining defaults are the runtime's (max_length resolves to the
    encoder's 8192-token context)."""
    import torch

    # upstream reads JULIA_CPU_THREADS only on its CUDA engine path, so
    # the CPU loader applies it here (see the module docstring)
    torch.set_num_threads(int(os.environ.get("JULIA_CPU_THREADS", "16")))

    home = model_home()
    if not os.path.isdir(os.path.join(home, "julia")):
        raise FileNotFoundError(
            f"no Julia-1 snapshot (julia/ runtime package) at {home} — clone "
            "SupersonicLabs/Julia-1 there or point JULIA_HOME at the directory"
        )
    # unconditional insert so a stale sibling cannot win the import
    while home in sys.path:
        sys.path.remove(home)
    sys.path.insert(0, home)
    _pop_stale_modules()

    import julia.router.encoder as _julia_encoder  # pyrefly: ignore[missing-import] — snapshot-resolved

    _julia_encoder.specialize_decision_encoder = lambda model: False  # noqa: E731 — the 5.17 guard

    import julia  # pyrefly: ignore[missing-import] — snapshot-resolved

    return julia.load_model(home, device=device, strict_encoding=True, head_length=512, **kwargs)


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    """One decision item: {input, question} -> {prediction, prob}.

    Julia's answer rows carry `max_probability` (the full softmax — no
    separate confidence field), which answer_to_result ignores in favour
    of the chosen label's own probability; same number by construction."""
    question = _systemone.normalize_question(payload.get("question"))
    state = _systemone.state_text(payload.get("input"))
    with _LOCK:
        if not _ENGINE:
            _ENGINE.append(load_engine())
        engine = _ENGINE[0]
    answers = engine.predict(state, {"q": question})
    return _systemone.answer_to_result(question["type"], answers["answers"]["q"])
