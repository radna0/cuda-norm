from __future__ import annotations

from dataclasses import dataclass
import typing as tp

from flax import nnx


@dataclass(frozen=True)
class DFlashDraftModelConfig:
    hidden_size: int
    num_layers: int
    mlp_ratio: float
    hidden_act: str
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    block_size: int
    num_context_features: int
    # Required for parity between training cache and inference-time verification.
    # Stored in config.json inside EasyDeL run-* checkpoints.
    target_layer_ids: list[int] | None = None
    add_one_for_pre_layer_capture: bool = True
    qk_norm: bool = True
    remat: bool = True

    def get_partition_rules(self):
        # DFlash draft is small; we replicate parameters by default.
        return []


def _repeat_kv(x, n_rep: int):
    # x: [B, S, kvH, D] -> [B, S, kvH*n_rep, D]
    import jax.numpy as jnp

    if int(n_rep) == 1:
        return x
    b, s, kvh, d = x.shape
    x = x[:, :, None, :, :]
    x = jnp.broadcast_to(x, (b, s, int(n_rep), kvh, d))
    return x.reshape((b, s, kvh * int(n_rep), d))


def _split_heads(x, n_heads: int, head_dim: int):
    return x.reshape((x.shape[0], x.shape[1], int(n_heads), int(head_dim)))


def _merge_heads(x):
    return x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))


def _apply_rope_separate(*, rope, pos_q, q, pos_k, k):
    q_rot, _ = rope(pos_q, q, q)
    _, k_rot = rope(pos_k, k, k)
    return q_rot, k_rot


def _per_head_rms_norm(x, eps: float):
    import jax
    import jax.numpy as jnp

    x_f = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x_f), axis=-1, keepdims=True)
    return (x_f * jax.lax.rsqrt(var + float(eps))).astype(x.dtype)


class DFlashDraftModel(nnx.Module):
    """NNX draft model: feature-conditioned non-causal decoder-only block.

    Inputs:
      - context_features: bf16 [B, ctx_len, K*hidden]
      - anchor_embedding: bf16 [B, hidden]
      - rope: EasyDeL RoPE object
    Output:
      - hidden: bf16 [B, block_size, hidden]
    """

    def __init__(self, cfg: DFlashDraftModelConfig, *, rngs):
        import jax.numpy as jnp

        self.cfg = cfg
        # EasyDeL expects models to expose `.param_dtype` and `.mesh`.
        self.param_dtype = jnp.bfloat16
        self.mesh: tp.Any | None = None
        # Not an EasyDeLBaseConfig, but useful for logging/debug.
        self.config = cfg

        self.fc = nnx.Linear(
            int(cfg.num_context_features) * int(cfg.hidden_size),
            int(cfg.hidden_size),
            use_bias=True,
            rngs=rngs,
        )
        self.hidden_norm = nnx.RMSNorm(int(cfg.hidden_size), epsilon=float(cfg.rms_norm_eps), rngs=rngs)
        self.mask_embedding = nnx.Param(
            nnx.initializers.normal(stddev=0.02)(rngs.params(), (int(cfg.hidden_size),), dtype=jnp.float32)
        )

        self.layers = []
        for _ in range(int(cfg.num_layers)):
            self.layers.append(_DFlashBlock(cfg, rngs=rngs))
        self.final_norm = nnx.RMSNorm(int(cfg.hidden_size), epsilon=float(cfg.rms_norm_eps), rngs=rngs)

    def project_context_features(self, context_features):
        """Project raw context features into draft hidden space.

        context_features: [B, ctx, K*hidden] -> ctx_hidden: [B, ctx, hidden]
        """
        return self.hidden_norm(self.fc(context_features))

    def __call__(self, *, context_features, anchor_embedding, rope):
        import jax.numpy as jnp

        c = self.cfg
        b = int(anchor_embedding.shape[0])

        ctx = self.project_context_features(context_features)

        # token0 = anchor, token1..B-1 = mask embedding
        mask = jnp.broadcast_to(
            self.mask_embedding.value.astype(ctx.dtype)[None, None, :],
            (b, int(c.block_size - 1), int(c.hidden_size)),
        )
        noise_hidden = jnp.concatenate([anchor_embedding[:, None, :].astype(ctx.dtype), mask], axis=1)

        x = noise_hidden
        for layer in self.layers:
            if bool(c.remat):
                def _call(rope_, ctx_hidden_, noise_hidden_):
                    return layer(rope=rope_, ctx_hidden=ctx_hidden_, noise_hidden=noise_hidden_)

                x = nnx.remat(_call)(rope, ctx, x)
            else:
                x = layer(rope=rope, ctx_hidden=ctx, noise_hidden=x)
        x = self.final_norm(x)
        return x

    def flops_per_token(self, *, include_loss: bool = True, include_backward: bool = False) -> float:
        # This model is trained with a custom CE loss over (block_size-1) tokens
        # via an external LM head, so the true FLOP accounting depends on vocab
        # size and tp sharding. We provide a conservative placeholder so EasyDeL
        # Trainer infrastructure can initialize.
        #
        # The training loop reports real wall-time throughput; FLOPs are only for
        # optional telemetry.
        return 0.0

    @property
    def dtype(self):
        return self.param_dtype


class _DFlashAttention(nnx.Module):
    def __init__(self, cfg: DFlashDraftModelConfig, *, rngs):
        self.cfg = cfg
        c = cfg
        self.q_proj = nnx.Linear(int(c.hidden_size), int(c.num_attention_heads) * int(c.head_dim), use_bias=True, rngs=rngs)
        self.k_proj = nnx.Linear(int(c.hidden_size), int(c.num_key_value_heads) * int(c.head_dim), use_bias=True, rngs=rngs)
        self.v_proj = nnx.Linear(int(c.hidden_size), int(c.num_key_value_heads) * int(c.head_dim), use_bias=True, rngs=rngs)
        self.o_proj = nnx.Linear(int(c.num_attention_heads) * int(c.head_dim), int(c.hidden_size), use_bias=True, rngs=rngs)

    def materialize_ctx_kv(self, *, rope, ctx_hidden):
        """Precompute K/V for ctx tokens for this layer.

        Returns:
          k_full_ctx: [B, ctx, H, D] (repeated to full heads, RoPE + optional qk_norm applied)
          v_full_ctx: [B, ctx, H, D] (repeated to full heads)
        """
        import jax.numpy as jnp

        c = self.cfg
        ctx_len = int(ctx_hidden.shape[1])
        k_ctx = _split_heads(self.k_proj(ctx_hidden), int(c.num_key_value_heads), int(c.head_dim))
        v_ctx = _split_heads(self.v_proj(ctx_hidden), int(c.num_key_value_heads), int(c.head_dim))

        pos_ctx = jnp.arange(ctx_len, dtype=jnp.int32)[None, :]
        # Apply RoPE to keys only by passing k as both q and k.
        _, k_ctx_rope = rope(pos_ctx, k_ctx, k_ctx)
        if bool(c.qk_norm):
            k_ctx_rope = _per_head_rms_norm(k_ctx_rope, eps=float(c.rms_norm_eps))

        rep = int(c.num_attention_heads) // int(c.num_key_value_heads)
        k_full_ctx = _repeat_kv(k_ctx_rope, rep)
        v_full_ctx = _repeat_kv(v_ctx, rep)
        return k_full_ctx, v_full_ctx

    def append_ctx_kv(self, *, rope, ctx_k_full, ctx_v_full, new_ctx_hidden, start_pos: int):
        """Append new ctx tokens to an existing ctx KV cache for this layer."""
        import jax.numpy as jnp

        c = self.cfg
        n_new = int(new_ctx_hidden.shape[1])
        if n_new <= 0:
            return ctx_k_full, ctx_v_full

        k_new = _split_heads(self.k_proj(new_ctx_hidden), int(c.num_key_value_heads), int(c.head_dim))
        v_new = _split_heads(self.v_proj(new_ctx_hidden), int(c.num_key_value_heads), int(c.head_dim))

        pos_new = (jnp.arange(n_new, dtype=jnp.int32) + int(start_pos))[None, :]
        _, k_new_rope = rope(pos_new, k_new, k_new)
        if bool(c.qk_norm):
            k_new_rope = _per_head_rms_norm(k_new_rope, eps=float(c.rms_norm_eps))

        rep = int(c.num_attention_heads) // int(c.num_key_value_heads)
        k_new_full = _repeat_kv(k_new_rope, rep)
        v_new_full = _repeat_kv(v_new, rep)
        return jnp.concatenate([ctx_k_full, k_new_full], axis=1), jnp.concatenate([ctx_v_full, v_new_full], axis=1)

    def forward_with_ctx_kv(self, *, rope, ctx_k_full, ctx_v_full, noise_hidden):
        """Compute attention for the draft block using pre-materialized ctx KV."""
        import jax
        import jax.numpy as jnp

        c = self.cfg
        ctx_len = int(ctx_k_full.shape[1])
        q_len = int(noise_hidden.shape[1])

        q = _split_heads(self.q_proj(noise_hidden), int(c.num_attention_heads), int(c.head_dim))
        k_noise = _split_heads(self.k_proj(noise_hidden), int(c.num_key_value_heads), int(c.head_dim))
        v_noise = _split_heads(self.v_proj(noise_hidden), int(c.num_key_value_heads), int(c.head_dim))

        pos = (jnp.arange(q_len, dtype=jnp.int32) + ctx_len)[None, :]
        q_rope, _ = rope(pos, q, q)
        _, k_noise_rope = rope(pos, k_noise, k_noise)

        if bool(c.qk_norm):
            q_rope = _per_head_rms_norm(q_rope, eps=float(c.rms_norm_eps))
            k_noise_rope = _per_head_rms_norm(k_noise_rope, eps=float(c.rms_norm_eps))

        rep = int(c.num_attention_heads) // int(c.num_key_value_heads)
        k_noise_full = _repeat_kv(k_noise_rope, rep)
        v_noise_full = _repeat_kv(v_noise, rep)

        k_all = jnp.concatenate([ctx_k_full, k_noise_full], axis=1)
        v_all = jnp.concatenate([ctx_v_full, v_noise_full], axis=1)

        scale = float(int(c.head_dim) ** -0.5)
        attn = jnp.einsum("bqhd,bkhd->bhqk", q_rope, k_all, precision=jax.lax.Precision.HIGHEST) * scale
        attn = attn - jnp.max(attn, axis=-1, keepdims=True)
        probs = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum("bhqk,bkhd->bqhd", probs, v_all, precision=jax.lax.Precision.HIGHEST)
        return self.o_proj(_merge_heads(out))

    def __call__(self, *, rope, ctx_hidden, noise_hidden):
        import jax
        import jax.numpy as jnp

        c = self.cfg
        ctx_len = int(ctx_hidden.shape[1])
        q_len = int(noise_hidden.shape[1])

        q = _split_heads(self.q_proj(noise_hidden), int(c.num_attention_heads), int(c.head_dim))
        k_ctx = _split_heads(self.k_proj(ctx_hidden), int(c.num_key_value_heads), int(c.head_dim))
        v_ctx = _split_heads(self.v_proj(ctx_hidden), int(c.num_key_value_heads), int(c.head_dim))
        k_noise = _split_heads(self.k_proj(noise_hidden), int(c.num_key_value_heads), int(c.head_dim))
        v_noise = _split_heads(self.v_proj(noise_hidden), int(c.num_key_value_heads), int(c.head_dim))

        k = jnp.concatenate([k_ctx, k_noise], axis=1)
        v = jnp.concatenate([v_ctx, v_noise], axis=1)

        pos_k = jnp.arange(ctx_len + q_len, dtype=jnp.int32)[None, :]
        pos_q = (jnp.arange(q_len, dtype=jnp.int32) + ctx_len)[None, :]
        q_rope, k_rope = _apply_rope_separate(rope=rope, pos_q=pos_q, q=q, pos_k=pos_k, k=k)

        if bool(c.qk_norm):
            q_rope = _per_head_rms_norm(q_rope, eps=float(c.rms_norm_eps))
            k_rope = _per_head_rms_norm(k_rope, eps=float(c.rms_norm_eps))

        rep = int(c.num_attention_heads) // int(c.num_key_value_heads)
        k_full = _repeat_kv(k_rope, rep)
        v_full = _repeat_kv(v, rep)

        scale = float(int(c.head_dim) ** -0.5)
        attn = jnp.einsum("bqhd,bkhd->bhqk", q_rope, k_full, precision=jax.lax.Precision.HIGHEST) * scale
        attn = attn - jnp.max(attn, axis=-1, keepdims=True)
        probs = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum("bhqk,bkhd->bqhd", probs, v_full, precision=jax.lax.Precision.HIGHEST)
        return self.o_proj(_merge_heads(out))


class _DFlashMLP(nnx.Module):
    def __init__(self, cfg: DFlashDraftModelConfig, *, rngs):
        self.cfg = cfg
        c = cfg
        inter = int(round(float(c.hidden_size) * float(c.mlp_ratio)))
        self.gate_proj = nnx.Linear(int(c.hidden_size), inter, use_bias=True, rngs=rngs)
        self.up_proj = nnx.Linear(int(c.hidden_size), inter, use_bias=True, rngs=rngs)
        self.down_proj = nnx.Linear(inter, int(c.hidden_size), use_bias=True, rngs=rngs)

    def __call__(self, x):
        import jax

        act = str(self.cfg.hidden_act).lower()
        if act not in ("silu", "swish"):
            raise ValueError(f"Unsupported hidden_act: {self.cfg.hidden_act!r}")
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _DFlashBlock(nnx.Module):
    def __init__(self, cfg: DFlashDraftModelConfig, *, rngs):
        self.cfg = cfg
        self.in_norm = nnx.RMSNorm(int(cfg.hidden_size), epsilon=float(cfg.rms_norm_eps), rngs=rngs)
        self.attn = _DFlashAttention(cfg, rngs=rngs)
        self.post_norm = nnx.RMSNorm(int(cfg.hidden_size), epsilon=float(cfg.rms_norm_eps), rngs=rngs)
        self.mlp = _DFlashMLP(cfg, rngs=rngs)

    def __call__(self, *, rope, ctx_hidden, noise_hidden):
        x = noise_hidden
        x = x + self.attn(rope=rope, ctx_hidden=ctx_hidden, noise_hidden=self.in_norm(x))
        x = x + self.mlp(self.post_norm(x))
        return x

    def materialize_ctx_kv(self, *, rope, ctx_hidden):
        return self.attn.materialize_ctx_kv(rope=rope, ctx_hidden=ctx_hidden)

    def append_ctx_kv(self, *, rope, ctx_k_full, ctx_v_full, new_ctx_hidden, start_pos: int):
        return self.attn.append_ctx_kv(
            rope=rope,
            ctx_k_full=ctx_k_full,
            ctx_v_full=ctx_v_full,
            new_ctx_hidden=new_ctx_hidden,
            start_pos=start_pos,
        )

    def forward_with_ctx_kv(self, *, rope, ctx_k_full, ctx_v_full, noise_hidden):
        x = noise_hidden
        x = x + self.attn.forward_with_ctx_kv(
            rope=rope,
            ctx_k_full=ctx_k_full,
            ctx_v_full=ctx_v_full,
            noise_hidden=self.in_norm(x),
        )
        x = x + self.mlp(self.post_norm(x))
        return x
