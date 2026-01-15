from __future__ import annotations

from typing import Any, Callable, Optional

from .configuration_gptoss_dflash import GptOssDFlashConfig
from .utils import build_target_layer_ids, extract_context_feature

try:
    import torch
    from torch import nn
    from torch.nn import functional as F

    from transformers.cache_utils import Cache
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
    from transformers.models.gpt_oss.modeling_gpt_oss import (
        GptOssRMSNorm,
        GptOssRotaryEmbedding,
        _apply_rotary_emb,
    )
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Cache = object  # type: ignore[assignment]
    ALL_ATTENTION_FUNCTIONS = {}  # type: ignore[assignment]
    PreTrainedModel = object  # type: ignore[assignment]
    GptOssRMSNorm = object  # type: ignore[assignment]
    GptOssRotaryEmbedding = object  # type: ignore[assignment]
    _apply_rotary_emb = None  # type: ignore[assignment]
    eager_attention_forward = None  # type: ignore[assignment]
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


def apply_rotary_pos_emb_dflash(q, k, cos, sin, unsqueeze_dim: int = 1):
    """
    GPT-OSS' built-in apply_rotary_pos_emb assumes q and k have the same seq_len.
    DFlash uses q_len=block_size while k_len=ctx_len+block_size, so we must slice
    cos/sin for q but not for k (same as the Qwen3 DFlash implementation).
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = _apply_rotary_emb(q, cos[..., -q_len:, :], sin[..., -q_len:, :])
    k_embed = _apply_rotary_emb(k, cos, sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward_dflash(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    scaling: float,
    dropout: float = 0.0,
    sliding_window: Optional[int] = None,
    **kwargs,
):
    """
    DFlash non-causal attention over a concatenated KV sequence.

    We keep this as a simple eager attention implementation to remain compatible
    across environments while ensuring dtype stability.
    """
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # Stabilize softmax.
    attn_weights = attn_weights - attn_weights.max(dim=-1, keepdim=True).values
    attn_probs = F.softmax(attn_weights, dim=-1, dtype=attn_weights.dtype)
    attn_probs = F.dropout(attn_probs, p=dropout, training=module.training)

    # Ensure dtype match for matmul.
    if value_states.dtype != attn_probs.dtype:
        value_states = value_states.to(attn_probs.dtype)
    attn_output = torch.matmul(attn_probs, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_probs


class GptOssDFlashAttention(nn.Module):
    def __init__(self, config: GptOssDFlashConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.head_dim = int(config.head_dim)
        self.num_key_value_groups = int(config.num_attention_heads // config.num_key_value_heads)
        self.scaling = float(self.head_dim**-0.5)
        self.attention_dropout = float(config.attention_dropout)
        self.is_causal = False

        self.q_proj = nn.Linear(
            int(config.hidden_size),
            int(config.num_attention_heads) * int(self.head_dim),
            bias=bool(config.attention_bias),
        )
        self.k_proj = nn.Linear(
            int(config.hidden_size),
            int(config.num_key_value_heads) * int(self.head_dim),
            bias=bool(config.attention_bias),
        )
        self.v_proj = nn.Linear(
            int(config.hidden_size),
            int(config.num_key_value_heads) * int(self.head_dim),
            bias=bool(config.attention_bias),
        )
        self.o_proj = nn.Linear(
            int(config.num_attention_heads) * int(self.head_dim),
            int(config.hidden_size),
            bias=bool(config.attention_bias),
        )
        self.sliding_window = int(config.sliding_window) if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]

        # q: (B, n_heads, q_len, head_dim)
        q_shape = (bsz, q_len, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(q_shape).transpose(1, 2)

        # k/v over [ctx || noise] : (B, n_kv, ctx_len+q_len, head_dim)
        k_ctx = self.k_proj(target_hidden)
        v_ctx = self.v_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_noise = self.v_proj(hidden_states)

        kv_shape = (bsz, ctx_len + q_len, -1, self.head_dim)
        key_states = torch.cat([k_ctx, k_noise], dim=1).view(kv_shape).transpose(1, 2)
        value_states = torch.cat([v_ctx, v_noise], dim=1).view(kv_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_dflash(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Always use our dtype-stable eager attention for DFlash.
        attention_interface: Callable = eager_attention_forward_dflash

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            is_causal=False,
            **kwargs,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class GptOssDFlashMLP(nn.Module):
    def __init__(self, config: GptOssDFlashConfig):
        super().__init__()
        hidden = int(config.hidden_size)
        inter = int(round(hidden * float(config.mlp_ratio)))
        self.gate_up = nn.Linear(hidden, 2 * inter, bias=False)
        self.down = nn.Linear(inter, hidden, bias=False)
        self.act = str(config.hidden_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        if self.act == "silu":
            h = F.silu(gate) * up
        else:
            h = F.gelu(gate) * up
        return self.down(h)


class GptOssDFlashDecoderLayer(nn.Module):
    def __init__(self, config: GptOssDFlashConfig, layer_idx: int):
        super().__init__()
        self.self_attn = GptOssDFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = GptOssDFlashMLP(config)
        self.input_layernorm = GptOssRMSNorm(int(config.hidden_size), eps=float(config.rms_norm_eps))
        self.post_attention_layernorm = GptOssRMSNorm(int(config.hidden_size), eps=float(config.rms_norm_eps))

    def forward(
        self,
        *,
        target_hidden: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings if position_embeddings is not None else (None, None),  # type: ignore[arg-type]
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class GptOssDFlashDraftModel(PreTrainedModel):  # type: ignore[misc]
    config_class = GptOssDFlashConfig
    base_model_prefix = "model"
    _no_split_modules = ["GptOssDFlashDecoderLayer"]

    def __init__(self, config: GptOssDFlashConfig):
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required for GptOssDFlashDraftModel") from _IMPORT_ERR
        super().__init__(config)
        self.layers = nn.ModuleList(
            [GptOssDFlashDecoderLayer(config, layer_idx=i) for i in range(int(config.num_hidden_layers))]
        )
        self.target_layer_ids = build_target_layer_ids(int(config.num_target_layers), int(config.num_hidden_layers))
        self.norm = GptOssRMSNorm(int(config.hidden_size), eps=float(config.rms_norm_eps))
        self.rotary_emb = GptOssRotaryEmbedding(config)
        self.fc = nn.Linear(len(self.target_layer_ids) * int(config.hidden_size), int(config.hidden_size), bias=False)
        self.hidden_norm = GptOssRMSNorm(int(config.hidden_size), eps=float(config.rms_norm_eps))
        self.block_size = int(config.block_size)
        # Learned embedding used for masked/noised positions when the target
        # tokenizer has no dedicated mask token (e.g., GPT-OSS). This avoids
        # overloading pad_token embedding as "mask", which tends to hurt learning.
        self.mask_embedding = nn.Parameter(torch.zeros(int(config.hidden_size)))
        self.post_init()

    def forward(
        self,
        *,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: torch.Tensor,
        target_hidden: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Keep dtypes aligned (bf16 target hidden states vs fp32-initialized draft weights).
        param_dtype = self.fc.weight.dtype
        if target_hidden.dtype != param_dtype:
            target_hidden = target_hidden.to(param_dtype)
        if noise_embedding.dtype != param_dtype:
            noise_embedding = noise_embedding.to(param_dtype)

        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    def extract_context_feature(self, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        return extract_context_feature(hidden_states, self.target_layer_ids)

    @classmethod
    def from_target_config(
        cls,
        *,
        target_model_id: str,
        target_config: Any,
        num_hidden_layers: int = 4,
        block_size: int = 16,
        mlp_ratio: float = 4.0,
    ) -> "GptOssDFlashDraftModel":
        cfg = GptOssDFlashConfig(
            target_model_id=target_model_id,
            vocab_size=int(target_config.vocab_size),
            hidden_size=int(target_config.hidden_size),
            num_attention_heads=int(target_config.num_attention_heads),
            num_key_value_heads=int(target_config.num_key_value_heads),
            head_dim=int(getattr(target_config, "head_dim", target_config.hidden_size // target_config.num_attention_heads)),
            max_position_embeddings=int(target_config.max_position_embeddings),
            rope_theta=float(getattr(target_config, "rope_theta", 150000.0)),
            rope_scaling=getattr(target_config, "rope_scaling", None),
            rms_norm_eps=float(getattr(target_config, "rms_norm_eps", 1e-5)),
            attention_dropout=float(getattr(target_config, "attention_dropout", 0.0)),
            attention_bias=bool(getattr(target_config, "attention_bias", True)),
            sliding_window=int(getattr(target_config, "sliding_window", 128)),
            layer_types=list(getattr(target_config, "layer_types", ["full_attention"] * int(target_config.num_hidden_layers))),
            num_hidden_layers=int(num_hidden_layers),
            # Match upstream DFlash layer selection: `num_target_layers` is the
            # number of transformer blocks in the target (embeddings are offset
            # separately when indexing HF hidden_states).
            num_target_layers=int(getattr(target_config, "num_hidden_layers", 36)),
            block_size=int(block_size),
            mlp_ratio=float(mlp_ratio),
            hidden_act=str(getattr(target_config, "hidden_act", "silu")),
        )
        return cls(cfg)
