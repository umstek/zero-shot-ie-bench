"""Vendored subset of BY571/nanoDiff (MIT): only Config and NanoDiff,
which is all the released code/eval_calibration.py loading path needs.
config.py and model.py are unmodified copies @ main; the imports below
are the one change - relative, so the package works inside nanodiff_engine
without a sys.path hack."""
from .config import Config
from .model import NanoDiff

__all__ = ["Config", "NanoDiff"]
