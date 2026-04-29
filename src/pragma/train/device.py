from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class RuntimeContext:
    device: torch.device
    device_type: str
    dtype: torch.dtype


def resolve_runtime(device: str | None = None, dtype: str | None = None) -> RuntimeContext:
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    torch_device = torch.device(device)
    device_type = torch_device.type
    if dtype is None:
        if device_type == "cuda" and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
    else:
        torch_dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[dtype]
    if device_type == "mps" and torch_dtype == torch.bfloat16:
        torch_dtype = torch.float32
    return RuntimeContext(device=torch_device, device_type=device_type, dtype=torch_dtype)


def autocast_context(ctx: RuntimeContext):
    if ctx.dtype == torch.float32 or ctx.device_type == "cpu":
        return nullcontext()
    return torch.autocast(device_type=ctx.device_type, dtype=ctx.dtype)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
