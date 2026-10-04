"""Optimizers for YCode-LM.

Muon (MomentUm Orthogonalized by Newton-Schulz) updates each 2-D weight matrix
with an orthogonalized version of its momentum, so every direction in the
update gets a similar step size. On small transformers it reaches a given loss
with noticeably fewer tokens than AdamW. It is only meant for the hidden
weight matrices; embeddings and norm gains keep using AdamW.

Reference: Keller Jordan et al., "Muon: An optimizer for hidden layers in
neural networks" (2024).
"""

from __future__ import annotations

import torch

_NS_COEFFS = (3.4445, -4.7750, 2.0315)


def orthogonalize(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the nearest semi-orthogonal matrix (U V^T of g's SVD) with a
    quintic Newton-Schulz iteration. Runs in bfloat16 for speed."""
    a, b, c = _NS_COEFFS
    x = g.to(torch.bfloat16)
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    if transposed:
        x = x.mT
    return x


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True,
                 ns_steps: int = 5, weight_decay: float = 0.0) -> None:
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                                      weight_decay=weight_decay))
        for group in self.param_groups:
            for p in group["params"]:
                if p.dim() != 2:
                    raise ValueError("Muon only handles 2-D weight matrices; use AdamW for the rest")

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                buf.lerp_(p.grad, 1 - group["momentum"])
                update = p.grad.lerp(buf, group["momentum"]) if group["nesterov"] else buf
                update = orthogonalize(update, group["ns_steps"]).to(p.dtype)
                # Keep the update RMS comparable across matrix shapes.
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"] * scale)
        return loss


class CombinedOptimizer:
    """Steps several optimizers as one (Muon for matrices + AdamW for the rest)."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        self.optimizers = optimizers
        for opt in optimizers:
            for group in opt.param_groups:
                group.setdefault("base_lr", group["lr"])

    @property
    def param_groups(self):
        return [g for opt in self.optimizers for g in opt.param_groups]

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def set_lr_scale(self, scale: float) -> None:
        for group in self.param_groups:
            group["lr"] = group["base_lr"] * scale

    def state_dict(self) -> dict:
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state: dict) -> None:
        for opt, opt_state in zip(self.optimizers, state["optimizers"]):
            opt.load_state_dict(opt_state)
        for opt in self.optimizers:
            for group in opt.param_groups:
                group.setdefault("base_lr", group["lr"])


def build_optimizer(model: torch.nn.Module, *, kind: str, lr: float, muon_lr: float, weight_decay: float,
                    fused: bool = False) -> CombinedOptimizer:
    """AdamW everywhere, or Muon for the transformer blocks' 2-D weights + AdamW for embeddings/norms."""
    adam_kwargs = dict(betas=(0.9, 0.95), fused=fused)
    if kind == "adamw":
        decay = [p for p in model.parameters() if p.dim() >= 2]
        no_decay = [p for p in model.parameters() if p.dim() < 2]
        return CombinedOptimizer([torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=lr, **adam_kwargs)])
    if kind == "muon":
        matrices, others = [], []
        for name, p in model.named_parameters():
            (matrices if (p.dim() == 2 and name.startswith("blocks.")) else others).append(p)
        return CombinedOptimizer([
            Muon(matrices, lr=muon_lr, weight_decay=0.0),
            torch.optim.AdamW([{"params": [p for p in others if p.dim() >= 2], "weight_decay": weight_decay},
                               {"params": [p for p in others if p.dim() < 2], "weight_decay": 0.0}],
                              lr=lr, **adam_kwargs),
        ])
    raise ValueError(f"unknown optimizer {kind!r}; use 'adamw' or 'muon'")
