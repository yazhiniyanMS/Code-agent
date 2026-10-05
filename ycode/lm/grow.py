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
