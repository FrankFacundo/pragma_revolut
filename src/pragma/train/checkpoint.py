from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from pragma.config import ModelConfig


def save_checkpoint(
    out_dir: str | Path,
    model: torch.nn.Module,
    *,
    config: ModelConfig,
    step: int,
    optimizer: torch.optim.Optimizer | None = None,
    meta: dict[str, Any] | None = None,
    name: str | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = name or f"model_{step:06d}"
    state = {
        key.removeprefix("_orig_mod."): value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }
    model_path = out / f"{stem}.safetensors"
    save_file(state, str(model_path))
    payload = {"step": step, "config": config.to_dict(), "meta": meta or {}}
    (out / f"{stem}.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), out / f"{stem}.optim.pt")
    return model_path


def load_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    strict: bool = False,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = Path(path)
    state = load_file(str(checkpoint), device=str(map_location))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    meta_path = checkpoint.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return {"missing": list(missing), "unexpected": list(unexpected), "meta": meta}


def checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    meta_path = Path(path).with_suffix(".json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    paths = sorted(Path(checkpoint_dir).glob("*.safetensors"))
    return paths[-1] if paths else None
