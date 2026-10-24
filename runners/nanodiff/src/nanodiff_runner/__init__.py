"""The nanodiff runner island: the diffusion decision engine (350M) on the
shared runner protocol.

Package layout (the vendored code keeps its upstream names for
provenance; engine.py is the adapted legacy runner):

- decision_format.py  — the typed-decision single/multi-choice
  serialization, verbatim from the pngwn release
  (code/decision_format.py)
- nanodiff/           — Config + NanoDiff, verbatim from
  github.com/BY571/nanoDiff (bidirectional LLaMA-style transformer)
- engine.py           — checkpoint loading + the released scoring path,
  ported from legacy/engines/nanodiff_engine/runner.py
- server.py           — create_app wiring + the wire predict glue

Importing this package never downloads or loads weights: the checkpoints
(hf_hub_download + torch.load) happen inside engine.load_model, lazily on
the first predict.
"""
