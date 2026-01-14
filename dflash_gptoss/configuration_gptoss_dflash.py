from __future__ import annotations

from transformers import PretrainedConfig


class GptOssDFlashConfig(PretrainedConfig):
    model_type = "gpt_oss_dflash"

    def __init__(
        self,
        *,
        target_model_id: str = "",
        vocab_size: int = 0,
        hidden_size: int = 0,
        num_attention_heads: int = 0,
        num_key_value_heads: int = 0,
        head_dim: int = 0,
        max_position_embeddings: int = 0,
        rope_theta: float = 150000.0,
        rope_scaling: dict | None = None,
        rms_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        attention_bias: bool = True,
        sliding_window: int = 128,
        layer_types: list[str] | None = None,
        initializer_range: float = 0.02,
        # Draft knobs
        num_hidden_layers: int = 4,
        num_target_layers: int = 36,
        block_size: int = 16,
        mlp_ratio: float = 4.0,
        hidden_act: str = "silu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_model_id = str(target_model_id)

        # Target-shape compatible params
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.max_position_embeddings = int(max_position_embeddings)
        self.rope_theta = float(rope_theta)
        self.rope_scaling = rope_scaling
        self.rms_norm_eps = float(rms_norm_eps)
        self.attention_dropout = float(attention_dropout)
        self.attention_bias = bool(attention_bias)
        self.sliding_window = int(sliding_window)
        self.layer_types = list(layer_types) if layer_types is not None else ["full_attention"] * int(num_target_layers)
        self.initializer_range = float(initializer_range)

        # Draft-only knobs
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_target_layers = int(num_target_layers)
        self.block_size = int(block_size)
        self.mlp_ratio = float(mlp_ratio)
        self.hidden_act = str(hidden_act)
