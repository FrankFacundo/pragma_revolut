from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pragma.model.rotary import RotaryEmbedding, apply_rotary


def sinusoidal_positions(positions: torch.Tensor, dim: int) -> torch.Tensor:
    device = positions.device
    dtype = torch.float32
    half = dim // 2
    exponent = torch.arange(half, device=device, dtype=dtype) / max(1, half)
    div_term = torch.exp(-math.log(10000.0) * exponent)
    angles = positions.to(dtype=dtype).unsqueeze(-1) * div_term
    out = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if dim % 2:
        out = F.pad(out, (0, 1))
    return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        dropout: float,
        rope_theta: float,
        use_rope: bool,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_rope = use_rope
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.rope = RotaryEmbedding(self.head_dim, theta=rope_theta) if use_rope else None

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        rope_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.use_rope:
            if rope_positions is None:
                raise ValueError("rope_positions are required for this attention block")
            assert self.rope is not None
            cos, sin = self.rope(rope_positions)
            q, k = apply_rotary(q, k, cos, sin)

        sdpa_mask = attention_mask[:, None, None, :]
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = attn.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        out = self.out_proj(out)
        return out * attention_mask.unsqueeze(-1).to(dtype=out.dtype)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, *, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ffn)
        self.fc2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x), approximate="tanh")))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        num_heads: int,
        *,
        dropout: float,
        rope_theta: float,
        use_rope: bool,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            dropout=dropout,
            rope_theta=rope_theta,
            use_rope=use_rope,
        )
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn = FeedForward(d_model, d_ffn, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        rope_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.dropout(
            self.attn(self.norm1(x), attention_mask=attention_mask, rope_positions=rope_positions)
        )
        ffn_out = self.dropout(self.ffn(self.norm2(x)))
        x = x + ffn_out * attention_mask.unsqueeze(-1).to(dtype=x.dtype)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth: int,
        d_model: int,
        d_ffn: int,
        num_heads: int,
        dropout: float,
        rope_theta: float,
        use_rope: bool,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    d_ffn,
                    num_heads,
                    dropout=dropout,
                    rope_theta=rope_theta,
                    use_rope=use_rope,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        rope_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask, rope_positions=rope_positions)
        return self.norm(x) * attention_mask.unsqueeze(-1).to(dtype=x.dtype)
