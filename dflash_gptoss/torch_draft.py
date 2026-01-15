from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _build_rope_cache(
    *,
    seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(dtype=dtype)
    sin = emb.sin().to(dtype=dtype)
    return cos, sin


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # cos/sin: [seq, head_dim]
    # q: [B, H, q_seq, D], k: [B, H, k_seq, D]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1,1,seq,D]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_len = q.shape[-2]
    q_cos = cos[..., -q_len:, :]
    q_sin = sin[..., -q_len:, :]
    q = (q * q_cos) + (_rotate_half(q) * q_sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (x.to(orig_dtype) * self.weight.to(orig_dtype))


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # x: [B, kvH, S, D] -> [B, kvH*n_rep, S, D]
    if n_rep == 1:
        return x
    b, kvh, s, d = x.shape
    x = x[:, :, None, :, :].expand(b, kvh, n_rep, s, d)
    return x.reshape(b, kvh * n_rep, s, d)


class GptOssDFlashAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rope_theta: float,
        attention_bias: bool,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.num_key_value_groups = int(self.num_attention_heads // self.num_key_value_heads)
        self.scaling = float(self.head_dim**-0.5)
        self.rope_theta = float(rope_theta)

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=bool(attention_bias))
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=bool(attention_bias))
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=bool(attention_bias))
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=bool(attention_bias))

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,      # [B, q_len, H]
        target_hidden: torch.Tensor,      # [B, ctx_len, H]
        cos: torch.Tensor,               # [ctx_len+q_len, head_dim]
        sin: torch.Tensor,               # [ctx_len+q_len, head_dim]
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape
        ctx_len = int(target_hidden.shape[1])

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        k_ctx = self.k_proj(target_hidden)
        v_ctx = self.v_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_noise = self.v_proj(hidden_states)

        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        q, k = _apply_rope(q, k, cos, sin)

        k = _repeat_kv(k, self.num_key_value_groups)
        v = _repeat_kv(v, self.num_key_value_groups)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn = attn - attn.amax(dim=-1, keepdim=True)
        probs = F.softmax(attn, dim=-1, dtype=attn.dtype)
        out = torch.matmul(probs, v)  # [B, heads, q_len, D]
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.num_attention_heads * self.head_dim)
        return self.o_proj(out)


class GptOssDFlashMLP(nn.Module):
    def __init__(self, *, hidden_size: int, mlp_ratio: float, hidden_act: str) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(round(self.hidden_size * float(mlp_ratio)))
        self.hidden_act = str(hidden_act)

        self.gate_up = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        if self.hidden_act == "silu":
            h = F.silu(gate) * up
        else:
            h = F.gelu(gate) * up
        return self.down(h)


class GptOssDFlashDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rope_theta: float,
        rms_norm_eps: float,
        attention_bias: bool,
        mlp_ratio: float,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = GptOssDFlashAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            attention_bias=attention_bias,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = GptOssDFlashMLP(hidden_size=hidden_size, mlp_ratio=mlp_ratio, hidden_act=hidden_act)

    def forward(self, *, hidden_states: torch.Tensor, target_hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states, target_hidden=target_hidden, cos=cos, sin=sin)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


@dataclass(frozen=True)
class GptOssDFlashDraftConfig:
    target_model_id: str
    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    attention_bias: bool
    num_hidden_layers: int
    num_target_layers: int
    block_size: int
    mlp_ratio: float
    hidden_act: str = "silu"


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    if int(num_draft_layers) == 1:
        return [int(num_target_layers // 2)]
    start = 1
    end = int(num_target_layers) - 3
    span = end - start
    return [int(round(start + (i * span) / (int(num_draft_layers) - 1))) for i in range(int(num_draft_layers))]


class TorchGptOssDFlashDraftModel(nn.Module):
    def __init__(self, cfg: GptOssDFlashDraftConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.target_layer_ids = build_target_layer_ids(int(cfg.num_target_layers), int(cfg.num_hidden_layers))

        self.layers = nn.ModuleList(
            [
                GptOssDFlashDecoderLayer(
                    hidden_size=int(cfg.hidden_size),
                    num_attention_heads=int(cfg.num_attention_heads),
                    num_key_value_heads=int(cfg.num_key_value_heads),
                    head_dim=int(cfg.head_dim),
                    rope_theta=float(cfg.rope_theta),
                    rms_norm_eps=float(cfg.rms_norm_eps),
                    attention_bias=bool(cfg.attention_bias),
                    mlp_ratio=float(cfg.mlp_ratio),
                    hidden_act=str(cfg.hidden_act),
                )
                for _ in range(int(cfg.num_hidden_layers))
            ]
        )

        self.norm = RMSNorm(int(cfg.hidden_size), eps=float(cfg.rms_norm_eps))
        self.fc = nn.Linear(len(self.target_layer_ids) * int(cfg.hidden_size), int(cfg.hidden_size), bias=False)
        self.hidden_norm = RMSNorm(int(cfg.hidden_size), eps=float(cfg.rms_norm_eps))
        self.mask_embedding = nn.Parameter(torch.zeros(int(cfg.hidden_size)))

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        # target_hidden: [B, ctx_len, K*hidden]
        return self.hidden_norm(self.fc(target_hidden))

    def forward(
        self,
        *,
        noise_embedding: torch.Tensor,  # [B, block, hidden]
        target_hidden: torch.Tensor,    # [B, ctx, K*hidden]
        rope_cos: torch.Tensor,         # [ctx+block, head_dim]
        rope_sin: torch.Tensor,         # [ctx+block, head_dim]
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        target_hidden_proj = self.project_target_hidden(target_hidden)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden_proj,
                cos=rope_cos,
                sin=rope_sin,
            )
        return self.norm(hidden_states)

    @staticmethod
    def build_rope_for_lengths(
        *,
        ctx_len: int,
        block_size: int,
        head_dim: int,
        rope_theta: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total = int(ctx_len) + int(block_size)
        return _build_rope_cache(seq_len=total, head_dim=int(head_dim), theta=float(rope_theta), device=device, dtype=dtype)

