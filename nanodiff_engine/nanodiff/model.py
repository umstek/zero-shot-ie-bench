"""nanoDiff model: a bidirectional LLaMA-style transformer.

The ONLY architectural difference from an autoregressive GPT is that attention is
bidirectional (no causal mask): a diffusion LM must see the whole partially-masked
sequence to fill in the masked positions.

We also adopt LLaDA's *time-free* parameterization: the timestep `t` is NOT fed to
the network. Because unmasked tokens already pin down the clean data, and masked
tokens are predicted purely from context, the optimal denoiser is time-invariant
(LLaDA, Eq. 11). This deletes all the timestep-embedding machinery you would see
in an image diffusion model — the model is genuinely just a transformer.

Components are otherwise standard modern-LLM: RMSNorm, SwiGLU, RoPE, pre-norm
residual blocks.
"""
import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# SDPA backend lock. On NVIDIA sm_121 (DGX Spark GB10) we measured that the
# EFFICIENT and CUDNN scaled-dot-product-attention backends produce >1%
# relative numerical drift vs FLASH on bf16 (max_diff ~0.14 on logits, well
# above the ~0.01 bf16 noise floor). Restrict SDPA to the two backends that
# are correctness-verified on every target we care about — FLASH on CUDA,
# MATH as the safe fallback for everything else.
try:
    from torch.nn.attention import sdpa_kernel as _sdpa_kernel, SDPBackend

    def _attention_ctx():
        return _sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH])
except ImportError:
    def _attention_ctx():
        return contextlib.nullcontext()


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


def precompute_rope(head_dim: int, max_seq_len: int, theta: float):
    """Precompute the rotary position embedding cos/sin tables, shape (T, head_dim)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    freqs = torch.outer(positions, inv_freq)          # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)           # (T, head_dim)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    """Rotate query/key vectors. x: (B, n_head, T, head_dim)."""
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    """Multi-head self-attention — bidirectional (this is the diffusion-LM change)."""

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = config.dropout

    def forward(self, x, cos, sin, return_kv=False):
        """Standard forward. When `return_kv=True`, also return the RoPE-rotated
        K and V tensors so the caller can seed a Fast-dLLM cache."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # is_causal=False  <-- the heart of a diffusion LM: every token attends to
        # every other token, so the model can use right-context to fill in [MASK]s.
        with _attention_ctx():
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=False,
                dropout_p=self.dropout if self.training else 0.0,
            )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return (y, k, v) if return_kv else y

    def forward_active(self, x_active, cos_active, sin_active, kv_cache, active_range):
        """Fast-dLLM forward: compute attention only over the active range, reusing
        cached K/V for the non-active positions.

        Args:
            x_active:    (B, A, C)               — input embeddings ONLY for the
                                                   active positions [s0, s1)
            cos_active:  (1, 1, A, head_dim)     — RoPE rotations for [s0, s1)
            sin_active:  (1, 1, A, head_dim)     — RoPE rotations for [s0, s1)
            kv_cache:    (K, V) tuple,
                         each (B, n_head, T, head_dim)  — already RoPE-rotated,
                         covers the full sequence [0, T). We OVERWRITE [s0:s1].
            active_range: (s0, s1)

        Returns:
            y_active:    (B, A, C)               — attention output for active positions
            new_kv:      (K', V') with the same full-T shape, updated at [s0:s1]
        """
        B, A, C = x_active.shape
        s0, s1 = active_range
        K_cached, V_cached = kv_cache

        # Q, K, V for the active range only — single qkv linear over A tokens
        # instead of T. That's where the cache pays for itself.
        q, k_new, v_new = self.qkv(x_active).split(C, dim=2)
        q     = q.view    (B, A, self.n_head, self.head_dim).transpose(1, 2)
        k_new = k_new.view(B, A, self.n_head, self.head_dim).transpose(1, 2)
        v_new = v_new.view(B, A, self.n_head, self.head_dim).transpose(1, 2)
        q     = apply_rope(q,     cos_active, sin_active)
        k_new = apply_rope(k_new, cos_active, sin_active)

        # Merge fresh K/V at [s0:s1] into the cached full-T tensors. Clone so the
        # caller's cache isn't mutated (cheap relative to attention; see commit
        # message for the alternative in-place strategy).
        K = K_cached.clone()
        V = V_cached.clone()
        K[:, :, s0:s1, :] = k_new
        V[:, :, s0:s1, :] = v_new

        # Attention: A queries × T keys/values. The output is only for the
        # active positions, which is the second source of cache savings.
        with _attention_ctx():
            y = F.scaled_dot_product_attention(
                q, K, V, is_causal=False,
                dropout_p=self.dropout if self.training else 0.0,
            )
        y = y.transpose(1, 2).contiguous().view(B, A, C)
        return self.proj(y), K, V


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (the LLaMA MLP)."""

    def __init__(self, config):
        super().__init__()
        hidden = int(config.mlp_ratio * config.n_embd * 2 / 3)
        hidden = 64 * ((hidden + 63) // 64)            # round up to a multiple of 64
        self.w1 = nn.Linear(config.n_embd, hidden, bias=config.bias)   # gate
        self.w3 = nn.Linear(config.n_embd, hidden, bias=config.bias)   # value
        self.w2 = nn.Linear(hidden, config.n_embd, bias=config.bias)   # down-projection
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = Attention(config)
        self.mlp_norm = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos, sin, return_kv=False):
        """Standard block forward. When `return_kv=True`, also return the
        attention layer's (K, V) — used to populate the Fast-dLLM cache."""
        if return_kv:
            attn_out, k, v = self.attn(self.attn_norm(x), cos, sin, return_kv=True)
            x = x + attn_out
            x = x + self.mlp(self.mlp_norm(x))
            return x, (k, v)
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x

    def forward_active(self, x_active, cos_active, sin_active, kv_cache, active_range):
        """Fast-dLLM decode: operate only on the active range, reusing kv_cache."""
        attn_out, K_new, V_new = self.attn.forward_active(
            self.attn_norm(x_active), cos_active, sin_active, kv_cache, active_range
        )
        x_active = x_active + attn_out
        x_active = x_active + self.mlp(self.mlp_norm(x_active))
        return x_active, (K_new, V_new)


class NanoDiff(nn.Module):
    """The full model: token embedding -> N bidirectional blocks -> logits."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_rope(head_dim, config.block_size, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 style: scale down the residual-path projections by 1/sqrt(2 * n_layer)
        # so the residual stream variance does not grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        """idx: (B, T) token ids, with [MASK] at corrupted positions -> logits (B, T, V)."""
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence length {T} > block_size"
        x = self.drop(self.tok_emb(idx))
        cos = self.rope_cos[:T].view(1, 1, T, -1).to(x.dtype)
        sin = self.rope_sin[:T].view(1, 1, T, -1).to(x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm(x)
        return self.lm_head(x)

    def forward_prefill(self, idx):
        """Run a full forward AND return per-layer K/V caches.

        Used to seed a Fast-dLLM cache at the start of a block: the prompt and
        all already-committed blocks contribute K/V that won't change during
        the upcoming block's denoising steps, so we compute them once here and
        reuse via `forward_decode` for every subsequent step in the block.

        Returns: (logits (B, T, V), kv_caches: list of (K, V) per layer)
        """
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence length {T} > block_size"
        x = self.drop(self.tok_emb(idx))
        cos = self.rope_cos[:T].view(1, 1, T, -1).to(x.dtype)
        sin = self.rope_sin[:T].view(1, 1, T, -1).to(x.dtype)
        kv_caches = []
        for block in self.blocks:
            x, kv = block(x, cos, sin, return_kv=True)
            kv_caches.append(kv)
        x = self.norm(x)
        return self.lm_head(x), kv_caches

    def forward_decode(self, idx, kv_caches, active_range):
        """Fast-dLLM decode: emit logits ONLY for positions in `active_range`,
        reusing kv_caches for K/V at non-active positions.

        Args:
            idx:         (B, T) — the full current sequence (used to read the
                                  active-region embeddings; non-active positions
                                  are ignored because K/V is cached).
            kv_caches:   list of (K, V) per layer, each (B, n_head, T, head_dim),
                         RoPE-rotated. Returned by a prior `forward_prefill`.
            active_range: (s0, s1) — half-open slice of positions to recompute.

        Returns: (logits_active (B, s1-s0, V), new_kv_caches with [s0:s1] updated)
        """
        B, T = idx.shape
        s0, s1 = active_range
        # Embed only the active range — saves the per-layer qkv/proj/MLP work
        # on the non-active positions (which is the whole point of the cache).
        x_active = self.drop(self.tok_emb(idx[:, s0:s1]))
        cos_active = self.rope_cos[s0:s1].view(1, 1, s1 - s0, -1).to(x_active.dtype)
        sin_active = self.rope_sin[s0:s1].view(1, 1, s1 - s0, -1).to(x_active.dtype)
        new_kv_caches = []
        for block, kv in zip(self.blocks, kv_caches):
            x_active, new_kv = block.forward_active(
                x_active, cos_active, sin_active, kv, (s0, s1)
            )
            new_kv_caches.append(new_kv)
        x_active = self.norm(x_active)
        return self.lm_head(x_active), new_kv_caches

    def get_num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if not self.config.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def configure_optimizers(self, weight_decay, lr, betas, device_type):
        """AdamW with weight decay on 2D+ params only (matrices), not on norms/biases."""
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas,
                                 fused=(device_type == "cuda"))
