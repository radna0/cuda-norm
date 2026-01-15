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

    def __call__(self, *, context_features, anchor_embedding, rope):
        import jax.numpy as jnp

        c = self.cfg
        b = int(anchor_embedding.shape[0])

        ctx = self.hidden_norm(self.fc(context_features))

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
