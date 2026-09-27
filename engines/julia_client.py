"""Loader for Julia 1, SupersonicLabs' typed-decision model (local snapshot).

The runtime package (`julia`) ships inside the SupersonicLabs/Julia-1 Hugging
Face repo itself; the complete snapshot (fp32 weights, 550.5 MiB, plus the
runtime) is cloned to ../Julia-1 (JULIA_HOME overrides) and installed into
.venv-von --no-deps, because its pyproject pins transformers >=5.0,<5.1
while .venv-von keeps 5.17.0. The one real drift is handled here:

- julia.router.encoder.specialize_decision_encoder monkey-patches an
  optional fast forward over ModernBertModel._update_attention_mask, an
  API transformers 5.17 removed - without the guard every predict()
  raises AttributeError. The patch is skipped so the stock forward runs;
  the runtime's own parity test asserts both paths produce exactly equal
  outputs (atol=0). Verified on 5.17: the runtime's top-level tests/
  suite (typed API, context) passes and end-to-end inference is sound;
  the julia/router/tests suite does not run here - every failure is the
  same removed API (those tests exercise the specialized forward
  directly) or the Bend native backends, whose .so needs a Linux build.

The runtime reads JULIA_CPU_THREADS (torch.set_num_threads, default 4)
when the engine loads, so set it before Python starts - the web-UI tab
passes JULIA_CPU_THREADS=16 to the process it spawns.
"""

import os

MODEL_ID = "SupersonicLabs/Julia-1"


def model_home() -> str:
    """Snapshot directory: JULIA_HOME, default ../Julia-1 from this repo."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.environ.get("JULIA_HOME",
                          os.path.abspath(os.path.join(root, os.pardir,
                                                       "Julia-1")))


def load_engine(device: str = "cpu", **kwargs):
    """julia.load_model over the local snapshot with the 5.17 compat guard.

    Heavy imports stay inside (offline tests import this module freely).
    strict_encoding=True and head_length=512 follow the model card; the
    remaining defaults are the runtime's (max_length resolves to the
    encoder's 8192-token context)."""
    import julia.router.encoder as _julia_encoder

    _julia_encoder.specialize_decision_encoder = lambda model: False

    import julia

    return julia.load_model(model_home(), device=device,
                            strict_encoding=True, head_length=512, **kwargs)
