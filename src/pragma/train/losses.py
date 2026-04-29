from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mlm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    valid = labels.ne(-100)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(
        logits.view(-1, logits.shape[-1]).float(),
        labels.view(-1),
        ignore_index=-100,
        label_smoothing=label_smoothing,
    )


def binary_or_multilabel_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits.float(), labels.float())
