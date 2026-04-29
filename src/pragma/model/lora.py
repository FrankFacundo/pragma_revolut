from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Low-rank adapter around a frozen nn.Linear layer."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        adapted = self.dropout(x) @ self.lora_a.t() @ self.lora_b.t()
        return base + adapted * self.scaling


def apply_lora(
    model: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_modules: Iterable[str] = ("qkv", "out_proj", "fc1", "fc2"),
    freeze_base: bool = True,
) -> nn.Module:
    """Replace selected Linear modules with LoRA adapters.

    The default targets match the PRAGMA paper's attention and MLP adaptation.
    """

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False
    targets = set(target_modules)
    _replace_lora(model, targets=targets, rank=rank, alpha=alpha, dropout=dropout)
    for name, param in model.named_parameters():
        if "lora_" in name or name.startswith("classifier."):
            param.requires_grad = True
    return model


def lora_trainable_parameters(model: nn.Module) -> tuple[int, int, float]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / max(1, total)
    return trainable, total, pct


def _replace_lora(
    module: nn.Module,
    *,
    targets: set[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in targets:
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
        else:
            _replace_lora(child, targets=targets, rank=rank, alpha=alpha, dropout=dropout)
