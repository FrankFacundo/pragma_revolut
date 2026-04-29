from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

Variant = Literal["tiny", "s", "m", "l"]


@dataclass(slots=True)
class ModelConfig:
    """Architecture parameters for the PRAGMA backbone."""

    vocab_size: int
    d_model: int = 192
    d_ffn: int = 768
    profile_layers: int = 1
    event_layers: int = 5
    history_layers: int = 2
    num_heads: int = 3
    dropout: float = 0.1
    max_field_tokens: int = 24
    max_profile_tokens: int = 200
    max_events: int = 6500
    rope_theta: float = 10000.0
    layer_norm_eps: float = 1e-5
    calendar_hidden: int | None = None
    pad_token_id: int = 0
    mask_token_id: int = 1
    unk_token_id: int = 2
    usr_token_id: int = 3
    evt_token_id: int = 4

    @property
    def head_dim(self) -> int:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        return self.d_model // self.num_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        return cls(**data)


@dataclass(slots=True)
class MaskingConfig:
    token_prob: float = 0.15
    event_prob: float = 0.10
    key_prob: float = 0.10
    mask_replace_prob: float = 0.80
    unk_replace_prob: float = 0.10


@dataclass(slots=True)
class TrainingConfig:
    """Shared trainer options for pretraining and downstream adaptation."""

    batch_size: int = 8
    steps: int = 1000
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0
    seed: int = 13
    device: str | None = None
    dtype: str | None = None
    num_workers: int = 0
    log_every: int = 10
    save_every: int = 500
    label_smoothing: float = 0.05
    masking: MaskingConfig = field(default_factory=MaskingConfig)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["masking"] = asdict(self.masking)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        copied = dict(data)
        if isinstance(copied.get("masking"), dict):
            copied["masking"] = MaskingConfig(**copied["masking"])
        return cls(**copied)


def model_config_for_variant(
    variant: Variant,
    *,
    vocab_size: int,
    pad_token_id: int = 0,
    mask_token_id: int = 1,
    unk_token_id: int = 2,
    usr_token_id: int = 3,
    evt_token_id: int = 4,
) -> ModelConfig:
    table: dict[str, dict[str, int]] = {
        "tiny": {
            "d_model": 64,
            "d_ffn": 192,
            "profile_layers": 1,
            "event_layers": 1,
            "history_layers": 1,
            "num_heads": 4,
        },
        "s": {
            "d_model": 192,
            "d_ffn": 768,
            "profile_layers": 1,
            "event_layers": 5,
            "history_layers": 2,
            "num_heads": 3,
        },
        "m": {
            "d_model": 512,
            "d_ffn": 2048,
            "profile_layers": 3,
            "event_layers": 16,
            "history_layers": 6,
            "num_heads": 8,
        },
        "l": {
            "d_model": 1024,
            "d_ffn": 4096,
            "profile_layers": 9,
            "event_layers": 45,
            "history_layers": 18,
            "num_heads": 16,
        },
    }
    if variant not in table:
        raise ValueError(f"Unknown model variant: {variant}")
    return ModelConfig(
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        mask_token_id=mask_token_id,
        unk_token_id=unk_token_id,
        usr_token_id=usr_token_id,
        evt_token_id=evt_token_id,
        **table[variant],
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
