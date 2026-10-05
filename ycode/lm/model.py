"""A GPT-style decoder-only transformer, written from scratch in PyTorch.

Architecture: token embeddings -> N x [RMSNorm -> causal self-attention
(rotary position embeddings) -> RMSNorm -> SwiGLU MLP] -> RMSNorm -> LM
head (weights tied to the embeddings). Generation uses a KV cache.

Version 2 adds grouped-query attention (fewer key/value heads: a smaller KV
cache and faster generation) and QK-norm (normalised queries/keys, which keeps
attention logits bounded and allows higher learning rates). Version-1
checkpoints load unchanged: the new fields default to v1 behaviour.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    n_kv_head: int | None = None  # v2: grouped-query attention (None = one KV head per query head)
    qk_norm: bool = False  # v2: RMSNorm on queries and keys
    arch_version: int = 1

    @property
    def kv_heads(self) -> int:
        return self.n_kv_head or self.n_head

    def to_dict(self) -> dict:
        return asdict(self)


_V2 = dict(qk_norm=True, arch_version=2)

# Named sizes. Parameter counts are for an 8k vocabulary.
PRESETS: dict[str, dict] = {
    # version 1
    "tiny": dict(n_layer=2, n_head=2, n_embd=64, block_size=128),       # ~0.6M, for tests
    "small": dict(n_layer=4, n_head=4, n_embd=256, block_size=256),     # ~5M, CPU-friendly
    "base": dict(n_layer=6, n_head=6, n_embd=384, block_size=512),      # ~14M
    "medium": dict(n_layer=8, n_head=8, n_embd=512, block_size=1024),   # ~30M, wants a GPU
    "large": dict(n_layer=12, n_head=12, n_embd=768, block_size=1024),  # ~92M, GPU
    # version 2 (GQA + QK-norm)
    "v2-tiny": dict(n_layer=2, n_head=4, n_kv_head=2, n_embd=64, block_size=128, **_V2),
    "v2-small": dict(n_layer=8, n_head=8, n_kv_head=2, n_embd=256, block_size=512, **_V2),
    "v2-base": dict(n_layer=12, n_head=8, n_kv_head=2, n_embd=512, block_size=1024, **_V2),
    "v2-medium": dict(n_layer=16, n_head=12, n_kv_head=4, n_embd=768, block_size=1024, **_V2),
    "v2-large": dict(n_layer=24, n_head=16, n_kv_head=4, n_embd=1024, block_size=2048, **_V2),
}
# version 3: the ~40M-parameter model (v2 architecture, scaled up, trained with Muon)
_V3 = dict(qk_norm=True, arch_version=3)
PRESETS.update({
    "v3-tiny": dict(n_layer=3, n_head=4, n_kv_head=2, n_embd=64, block_size=128, **_V3),
    "v3-40m": dict(n_layer=13, n_head=8, n_kv_head=2, n_embd=512, block_size=1024, **_V3),
})
# version 4: ~100M parameters. Deep and narrow (34 x 512), which suits small models and lets
# v4 be grown from a trained v3 (13 x 512) without changing its width.
_V4 = dict(qk_norm=True, arch_version=4)
PRESETS.update({
    "v4-tiny": dict(n_layer=5, n_head=4, n_kv_head=2, n_embd=64, block_size=128, **_V4),
    "v4-100m": dict(n_layer=34, n_head=8, n_kv_head=2, n_embd=512, block_size=1024, **_V4),
})
DEFAULT_PRESET = "v2-small"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def rope_tables(head_dim: int, length: int, device=None, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(length, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (T, hd/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, hd); cos/sin: (T, hd/2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        assert cfg.n_head % cfg.kv_heads == 0, "n_head must be a multiple of n_kv_head"
        self.n_head = cfg.n_head
        self.n_kv = cfg.kv_heads
        self.head_dim = cfg.n_embd // cfg.n_head
        assert self.head_dim % 2 == 0, "head dimension must be even for rotary embeddings"
        self.kv_dim = self.n_kv * self.head_dim
        # One fused projection; with n_kv == n_head this is exactly the v1 layout.
        self.qkv = nn.Linear(cfg.n_embd, cfg.n_embd + 2 * self.kv_dim, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = self.k_norm = None
        self.dropout = cfg.dropout

    def _sdpa(self, q, k, v, **kwargs):
        if self.n_kv != self.n_head:
            repeat = self.n_head // self.n_kv
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        return F.scaled_dot_product_attention(q, k, v, **kwargs)

    def forward(self, x, cos, sin, cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split([C, self.kv_dim, self.kv_dim], dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            if cache.get("k") is not None:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v  # cached at n_kv heads: GQA keeps the cache small
        # Causal masking: with a cache, the T new queries see all past keys.
        if cache is not None and k.size(2) != T:
            if T == 1:
                y = self._sdpa(q, k, v)
            else:
                past = k.size(2) - T
                mask = torch.ones(T, k.size(2), dtype=torch.bool, device=x.device).tril(diagonal=past)
                y = self._sdpa(q, k, v, attn_mask=mask)
        else:
            y = self._sdpa(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3)
        hidden = 64 * ((hidden + 63) // 64)
        self.gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.norm1(x), cos, sin, cache)
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying
        cos, sin = rope_tables(cfg.n_embd // cfg.n_head, cfg.block_size)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                loss_mask: torch.Tensor | None = None, caches=None, start_pos: int = 0):
        B, T = idx.shape
        if start_pos + T > self.cfg.block_size:
            raise ValueError(f"sequence length {start_pos + T} exceeds block_size {self.cfg.block_size}")
        cos = self.rope_cos[start_pos: start_pos + T]
        sin = self.rope_sin[start_pos: start_pos + T]
        x = self.drop(self.embed(idx))
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, caches[i] if caches is not None else None)
        x = self.norm(x)
        if targets is None:
            return self.head(x[:, -1:, :]), None
        logits = self.head(x)
        losses = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
        if loss_mask is not None:
            mask = loss_mask.reshape(-1).float()
            loss = (losses * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            loss = losses.mean()
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, *, temperature: float = 0.8,
                 top_k: int | None = 50, top_p: float | None = 0.95, stop_ids: set[int] | None = None,
                 on_token=None, no_repeat_ngram: int = 0) -> torch.Tensor:
        """Sample tokens after ``idx`` (shape (1, T)). Uses a KV cache.

        ``no_repeat_ngram=n`` forbids any token that would repeat an n-token span already
        generated; this breaks degenerate loops without penalising ordinary repetition
        such as indentation or a variable name used twice."""
        self.eval()
        seen: dict[tuple[int, ...], set[int]] = {}
        generated: list[int] = []
        stop_ids = stop_ids or set()
        # Keep the most recent context that leaves room for generation.
        budget = self.cfg.block_size - max_new_tokens
        if budget < 1:
            max_new_tokens = self.cfg.block_size // 2
            budget = self.cfg.block_size - max_new_tokens
        idx = idx[:, -budget:]
        caches = [dict() for _ in self.blocks]
        logits, _ = self(idx, caches=caches, start_pos=0)
        pos = idx.size(1)
        out = idx
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :].float()
            if no_repeat_ngram > 1 and len(generated) >= no_repeat_ngram - 1:
                banned = seen.get(tuple(generated[-(no_repeat_ngram - 1):]))
                if banned:
                    next_logits[:, list(banned)] = float("-inf")
            if temperature <= 0:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k:
                    kth = torch.topk(next_logits, min(top_k, next_logits.size(-1))).values[:, -1, None]
                    next_logits = next_logits.masked_fill(next_logits < kth, float("-inf"))
                probs = F.softmax(next_logits, dim=-1)
                if top_p and top_p < 1.0:
                    sorted_probs, sorted_idx = probs.sort(descending=True)
                    cumulative = sorted_probs.cumsum(-1)
                    sorted_probs[cumulative - sorted_probs > top_p] = 0.0
                    probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
                    probs = probs / probs.sum(-1, keepdim=True)
                next_id = torch.multinomial(probs, 1)
            token = int(next_id)
            generated.append(token)
            if no_repeat_ngram > 1 and len(generated) >= no_repeat_ngram:
                seen.setdefault(tuple(generated[-no_repeat_ngram:-1]), set()).add(token)
            out = torch.cat([out, next_id], dim=1)
            if token in stop_ids:
                break
            if on_token is not None:
                on_token(token)
            if pos >= self.cfg.block_size:
                break
            logits, _ = self(next_id, caches=caches, start_pos=pos)
            pos += 1
        return out
