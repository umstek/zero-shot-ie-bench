"""gliner runner island — the GLiNER lineage on the transformers 4.57.x CPU stack.

Serves the local checkpoints the catalog wires to `runner: gliner`
(catalog/models/{knowledgator,jackboyla,convaiinnovations}): GLiFormer
base/large, GLiREL, GLiNER-relex, GLiClass ×3 and Laya (local, router,
typed-decisions). The GLiNER2.5 (gliner2) and GLiNER-X (gliner[stanza])
checkpoints are NOT served here: the former is absent from the island's
catalog slice, the latter is culled and needs its own transformers-5
resolve (legacy .venv-glinerx).

Every model loads lazily: weights are pulled from the HF cache on first
use, never at import. Tests never download weights — they skip unless
the checkpoint is already in the local HF cache.
"""

__all__ = ["server"]
