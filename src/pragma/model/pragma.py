from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn

from pragma.config import ModelConfig
from pragma.model.layers import TransformerEncoder, sinusoidal_positions

RepresentationKind = Literal["usr", "last_event", "both"]


class CalendarEncoder(nn.Module):
    def __init__(self, d_model: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or d_model
        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, calendar: torch.Tensor) -> torch.Tensor:
        hour = calendar[..., 0] / 24.0
        weekday = calendar[..., 1] / 7.0
        monthday = (calendar[..., 2].clamp(min=1) - 1.0) / 31.0
        angles = torch.stack([hour, weekday, monthday], dim=-1) * (2.0 * torch.pi)
        features = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return self.net(features.to(dtype=next(self.parameters()).dtype))


class PragmaBackbone(nn.Module):
    """PRAGMA profile/event/history encoder backbone."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.profile_encoder = TransformerEncoder(
            depth=config.profile_layers,
            d_model=config.d_model,
            d_ffn=config.d_ffn,
            num_heads=config.num_heads,
            dropout=config.dropout,
            rope_theta=config.rope_theta,
            use_rope=True,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.event_encoder = TransformerEncoder(
            depth=config.event_layers,
            d_model=config.d_model,
            d_ffn=config.d_ffn,
            num_heads=config.num_heads,
            dropout=config.dropout,
            rope_theta=config.rope_theta,
            use_rope=False,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.calendar_encoder = CalendarEncoder(config.d_model, config.calendar_hidden)
        self.history_encoder = TransformerEncoder(
            depth=config.history_layers,
            d_model=config.d_model,
            d_ffn=config.d_ffn,
            num_heads=config.num_heads,
            dropout=config.dropout,
            rope_theta=config.rope_theta,
            use_rope=True,
            layer_norm_eps=config.layer_norm_eps,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        profile_repr = self._encode_profile(batch)
        event_token_embeddings, event_repr = self._encode_events(batch)
        event_repr = event_repr + self.calendar_encoder(batch["calendar"]) * batch[
            "event_mask"
        ].unsqueeze(-1).to(dtype=event_repr.dtype)
        history_input = torch.cat([profile_repr, event_repr], dim=1)
        history_mask = torch.cat(
            [
                torch.ones(
                    batch["event_mask"].shape[0],
                    1,
                    dtype=torch.bool,
                    device=batch["event_mask"].device,
                ),
                batch["event_mask"],
            ],
            dim=1,
        )
        history_times = torch.cat(
            [torch.zeros_like(batch["event_times"][:, :1]), batch["event_times"]],
            dim=1,
        )
        history = self.history_encoder(
            history_input,
            attention_mask=history_mask,
            rope_positions=history_times,
        )
        return {
            "history": history,
            "history_mask": history_mask,
            "user_embedding": history[:, 0],
            "event_token_embeddings": event_token_embeddings,
            "history_event_embeddings": history[:, 1:],
            "event_embeddings": event_repr,
        }

    def _pair_embeddings(
        self,
        key_ids: torch.Tensor,
        value_ids: torch.Tensor,
        value_positions: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embedding(key_ids) + self.embedding(value_ids)
        x = x + sinusoidal_positions(value_positions, self.config.d_model).to(dtype=x.dtype)
        return self.dropout(x)

    def _encode_profile(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        bsz = batch["profile_key_ids"].shape[0]
        x = self._pair_embeddings(
            batch["profile_key_ids"],
            batch["profile_value_ids"],
            batch["profile_positions"],
        )
        cls = self.embedding(
            torch.full(
                (bsz, 1),
                self.config.usr_token_id,
                dtype=torch.long,
                device=x.device,
            )
        )
        x = torch.cat([cls, x], dim=1)
        mask = torch.cat(
            [
                torch.ones(bsz, 1, dtype=torch.bool, device=x.device),
                batch["profile_mask"],
            ],
            dim=1,
        )
        times = torch.cat(
            [
                torch.zeros(bsz, 1, dtype=batch["profile_times"].dtype, device=x.device),
                batch["profile_times"],
            ],
            dim=1,
        )
        encoded = self.profile_encoder(x, attention_mask=mask, rope_positions=times)
        return encoded[:, :1]

    def _encode_events(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, event_count, token_count = batch["event_key_ids"].shape
        flat_key = batch["event_key_ids"].reshape(bsz * event_count, token_count)
        flat_value = batch["event_value_ids"].reshape(bsz * event_count, token_count)
        flat_pos = batch["event_positions"].reshape(bsz * event_count, token_count)
        flat_mask = batch["event_token_mask"].reshape(bsz * event_count, token_count)
        x = self._pair_embeddings(flat_key, flat_value, flat_pos)
        cls = self.embedding(
            torch.full(
                (bsz * event_count, 1),
                self.config.evt_token_id,
                dtype=torch.long,
                device=x.device,
            )
        )
        x = torch.cat([cls, x], dim=1)
        mask = torch.cat(
            [torch.ones(bsz * event_count, 1, dtype=torch.bool, device=x.device), flat_mask],
            dim=1,
        )
        encoded = self.event_encoder(x, attention_mask=mask)
        token_embeddings = encoded[:, 1:].reshape(
            bsz,
            event_count,
            token_count,
            self.config.d_model,
        )
        event_embeddings = encoded[:, 0].reshape(bsz, event_count, self.config.d_model)
        event_embeddings = event_embeddings * batch["event_mask"].unsqueeze(-1).to(
            dtype=event_embeddings.dtype
        )
        token_embeddings = token_embeddings * batch["event_token_mask"].unsqueeze(-1).to(
            dtype=token_embeddings.dtype
        )
        return token_embeddings, event_embeddings


class PragmaForMaskedModeling(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = PragmaBackbone(config)
        self.mlm_projection = nn.Sequential(
            nn.Linear(config.d_model * 3, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model, eps=config.layer_norm_eps),
        )
        self.mlm_bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = self.backbone(batch)
        token = out["event_token_embeddings"]
        history_event = out["history_event_embeddings"].unsqueeze(2).expand_as(token)
        user = out["user_embedding"][:, None, None, :].expand_as(token)
        fused = torch.cat([token, history_event, user], dim=-1)
        hidden = self.mlm_projection(fused)
        logits = torch.matmul(hidden, self.backbone.embedding.weight.t()) + self.mlm_bias
        out["mlm_logits"] = logits
        return out


class PragmaForSequenceClassification(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        num_labels: int = 1,
        representation: RepresentationKind = "both",
    ) -> None:
        super().__init__()
        self.config = config
        self.representation = representation
        self.backbone = PragmaBackbone(config)
        width = config.d_model * (2 if representation == "both" else 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(width, eps=config.layer_norm_eps),
            nn.Dropout(config.dropout),
            nn.Linear(width, num_labels),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = self.backbone(batch)
        rep = sequence_representation(
            out["history"],
            batch["event_mask"],
            representation=self.representation,
        )
        logits = self.classifier(rep)
        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        out["logits"] = logits
        out["sequence_embedding"] = rep
        return out


def sequence_representation(
    history: torch.Tensor,
    event_mask: torch.Tensor,
    *,
    representation: RepresentationKind,
) -> torch.Tensor:
    usr = history[:, 0]
    if representation == "usr":
        return usr
    event_history = history[:, 1:]
    lengths = event_mask.long().sum(dim=1).clamp(min=1)
    gather_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, event_history.shape[-1])
    last_event = event_history.gather(1, gather_idx).squeeze(1)
    last_event = last_event * (event_mask.any(dim=1).unsqueeze(-1).to(dtype=last_event.dtype))
    if representation == "last_event":
        return last_event
    if representation == "both":
        return torch.cat([usr, last_event], dim=-1)
    raise ValueError(f"Unknown representation: {representation}")


def batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
