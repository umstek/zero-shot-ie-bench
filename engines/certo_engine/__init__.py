# Vendored inference path for altslate/certo-decision-model: files infer.py
# and model_generic.py copied verbatim from the certo GitHub repo
# (github.com/AltSlate-Labs/certo), revision
# 5ae9dc40ec015e77606e4e7ec53aee87f89577d8 (2026-09-21), MIT licensed (repo
# LICENSE). The checkpoint card documents no PyPI package - usage is
# "from infer import DecisionModel" against a clone of that repo - so the
# two-file inference surface is vendored here instead (model_generic.py is
# included unchanged; infer.py differs only in the relative import noted in
# its header).
"""Certo: calibrated non-generative decision model (ModernBERT-large +
per-option query/scoring head). One forward pass, no text generation."""
from .infer import DecisionModel  # noqa: F401
