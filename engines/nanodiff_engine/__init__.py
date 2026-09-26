"""Vendored inference path for the nanodiff 350M system.

Sources (both MIT, stated on the pngwn model card and the BY571 repo):
- nanodiff/  — NanoDiff model class, verbatim from github.com/BY571/nanoDiff
  (bidirectional LLaMA-style transformer, pure torch).
- decision_format.py — typed-decision single-choice serialization, verbatim
  from the pngwn/nanodiff-350m-typed-decisions release (code/decision_format.py).
- runner.py — checkpoint loading + the released scoring path (answer slot
  [MASK]ed, one bidirectional forward, softmax over option-letter token ids),
  adapted from the release's code/eval_calibration.py.

The benchmarked checkpoint is the lam1 (proper-scoring) arm:
pngwn/nanodiff-350m-typed-decisions-lam1, step-3000 artifact.
"""
import sys

from . import nanodiff as _nanodiff

# the checkpoint pickle and the release's own modules import the package
# by its upstream name; alias it so loading works from anywhere
sys.modules.setdefault("nanodiff", _nanodiff)
