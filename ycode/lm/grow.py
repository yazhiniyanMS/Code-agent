"""Grow a trained model into a deeper one without losing what it learned.

New blocks are inserted between the existing ones. Each new block starts as a
copy of the block before it, with its two output projections (attention
``proj`` and MLP ``down``) set to zero. Because every block is residual
(``x + f(x)``), a block whose output projections are zero is an exact
identity: the grown model computes exactly the same function as the original
at step 0. Training then teaches the new blocks to add to it.

This reuses all the compute already spent on the smaller model, which on
limited hardware beats training the bigger model from scratch.
"""

from __future__ import annotations

import copy

import torch

from ycode.lm.model import GPT, GPTConfig


def insertion_plan(old_layers: int, new_layers: int) -> list[int]:
    """How many new blocks to insert after each original block (spread evenly)."""
    if new_layers < old_layers:
        raise ValueError("can only grow: new_layers must be >= the current number of layers")
    extra = new_layers - old_layers
    plan = [extra // old_layers] * old_layers
    for i in range(extra % old_layers):
        # Spread the remainder across the whole depth, not just the first blocks.
        plan[(i * old_layers) // (extra % old_layers)] += 1
    return plan


@torch.no_grad()
def grow_depth(model: GPT, new_layers: int, *, arch_version: int | None = None) -> GPT:
    old = model.cfg
    cfg = GPTConfig(**{**old.to_dict(), "n_layer": new_layers,
                       "arch_version": arch_version if arch_version is not None else old.arch_version})
    grown = GPT(cfg)
    grown.embed.weight.copy_(model.embed.weight)  # head is tied to embed
    grown.norm.load_state_dict(model.norm.state_dict())
    blocks = []
    for block, n_new in zip(model.blocks, insertion_plan(old.n_layer, new_layers)):
        blocks.append(copy.deepcopy(block))
        for _ in range(n_new):
            fresh = copy.deepcopy(block)
            fresh.attn.proj.weight.zero_()
            fresh.mlp.down.weight.zero_()
            blocks.append(fresh)
    grown.blocks = torch.nn.ModuleList(blocks)
    return grown


@torch.no_grad()
def grow_width(model: GPT, new_embd: int, *, new_heads: int | None = None, new_kv_heads: int | None = None,
               init_std: float = 0.02, arch_version: int | None = None, seed: int = 0) -> GPT:
    """Widen a model (more channels, attention heads and MLP units) while keeping its function.

    The residual stream gets extra channels that start at exactly zero: everything that *writes*
    into them (embedding columns, attention ``proj`` rows, MLP ``down`` rows) is zero. Everything
    that only *reads* from new channels or feeds new heads / new MLP units gets small random
    weights, so the new capacity receives gradients and can learn. RMSNorm divides by the RMS over
    all channels, so the old gains are rescaled by sqrt(old/new) to cancel the zero channels.
    Head size stays the same, so old query heads keep using their old key/value heads.
    The result matches the original up to RMSNorm's small eps term (differences around 1e-4).
    """
    old = model.cfg
    hd = old.n_embd // old.n_head
    n_head = new_heads or new_embd // hd
    n_kv = new_kv_heads or old.kv_heads * n_head // old.n_head
    if new_embd < old.n_embd or n_head * hd != new_embd:
        raise ValueError(f"new width must be >= {old.n_embd} and a multiple of the head size {hd}")
    if n_head < old.n_head or n_kv < old.kv_heads or n_head % n_kv:
        raise ValueError("cannot shrink heads, and n_head must be a multiple of n_kv_head")
    if (n_head // n_kv) != (old.n_head // old.kv_heads):
        raise ValueError("keep the query-to-key/value head ratio so old heads keep their pairing")
    cfg = GPTConfig(**{**old.to_dict(), "n_embd": new_embd, "n_head": n_head, "n_kv_head": n_kv,
                       "arch_version": arch_version if arch_version is not None else old.arch_version})
    gen = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=gen) * init_std

    wide = GPT(cfg)
    C0, C1 = old.n_embd, new_embd
    scale = (C0 / C1) ** 0.5

    wide.embed.weight.zero_()
    wide.embed.weight[:, :C0] = model.embed.weight  # new channels start at zero (head is tied)
    wide.norm.weight.fill_(scale)
    wide.norm.weight[:C0] = model.norm.weight * scale

    kv0, kv1 = old.kv_heads * hd, n_kv * hd
    for ob, nb in zip(model.blocks, wide.blocks):
        for name in ("norm1", "norm2"):
            getattr(nb, name).weight.fill_(scale)
            getattr(nb, name).weight[:C0] = getattr(ob, name).weight * scale
        # attention: fused qkv rows = [q (C) | k (kv) | v (kv)]
        w_old, w = ob.attn.qkv.weight, nb.attn.qkv.weight
        w.copy_(rand(*w.shape))
        for (o0, o1, n0) in ((0, C0, 0), (C0, C0 + kv0, C1), (C0 + kv0, C0 + 2 * kv0, C1 + kv1)):
            w[n0:n0 + (o1 - o0), :C0] = w_old[o0:o1]
        nb.attn.proj.weight.zero_()
        nb.attn.proj.weight[:C0, :C0] = ob.attn.proj.weight  # new heads' output: zero for now
        if ob.attn.q_norm is not None:
            nb.attn.q_norm.load_state_dict(ob.attn.q_norm.state_dict())
            nb.attn.k_norm.load_state_dict(ob.attn.k_norm.state_dict())
        # MLP: new hidden units read random weights, write nothing yet
        h0 = ob.mlp.gate.weight.shape[0]
        for name in ("gate", "up"):
            w = getattr(nb.mlp, name).weight
            w.copy_(rand(*w.shape))
            w[:h0, :C0] = getattr(ob.mlp, name).weight
        nb.mlp.down.weight.zero_()
        nb.mlp.down.weight[:C0, :h0] = ob.mlp.down.weight
    return wide
