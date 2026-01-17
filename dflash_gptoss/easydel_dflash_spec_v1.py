from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple


def dflash_accept_len_and_bonus(
    *,
    candidates,  # int32/int64 [bs, B]
    target_predict,  # int32/int64 [bs, B]
) -> Tuple["jax.Array", "jax.Array"]:
    """JAX port of SGLang `compute_dflash_accept_len_and_bonus`.

    Rule:
      accept while candidates[:, 1:] == target_predict[:, :-1] consecutively
      accept_len excludes the current token (index 0).
      bonus token is target_predict[:, accept_len].

    Returns:
      accept_len: int32 [bs]
      bonus: int32 [bs]
    """
    import jax
    import jax.numpy as jnp

    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={candidates.shape}")
    if target_predict.shape != candidates.shape:
        raise ValueError(
            f"target_predict must match candidates shape. candidates={candidates.shape} target_predict={target_predict.shape}"
        )
    bs, block_size = candidates.shape
    if int(block_size) <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if int(bs) <= 0:
        raise ValueError(f"batch size must be positive, got {bs}")

    matches = candidates[:, 1:] == target_predict[:, :-1]
    # Cumprod of 0/1 match flags stops at first mismatch; sum counts prefix length.
    accept_len = jnp.sum(jnp.cumprod(matches.astype(jnp.int32), axis=1), axis=1).astype(jnp.int32)
    bonus = jnp.take_along_axis(target_predict, accept_len[:, None], axis=1)[:, 0].astype(jnp.int32)
    return accept_len, bonus


def extract_dflash_context_features_from_hidden_states(
    *,
    hidden_states: Iterable["jax.Array"],
    target_layer_ids: list[int],
    add_one_for_pre_layer_capture: bool = True,
) -> "jax.Array":
    """Build per-token context features by concatenating selected layer hidden states.

    This mirrors SGLang's DFlash behavior where `target_layer_ids` are specified in
    HF-style "after layer i" space, but SGLang captures *pre-layer* activations, so
    it indexes `i+1`.

    EasyDeL's GPT-OSS currently returns `hidden_states` as a tuple where entry `i`
    is the hidden state *before* executing layer i (when output_hidden_states=True).

    Returns:
      context_features: [batch, seq_len, K * hidden_size]
    """
    import jax.numpy as jnp

    hs_list = list(hidden_states)
    if not hs_list:
        raise ValueError("hidden_states is empty")
    if not isinstance(target_layer_ids, list) or not target_layer_ids:
        raise ValueError("target_layer_ids must be a non-empty list[int]")

    idxs = []
    for val in target_layer_ids:
        i = int(val)
        if add_one_for_pre_layer_capture:
            i = i + 1
        idxs.append(i)

    max_i = max(idxs)
    if max_i >= len(hs_list):
        raise ValueError(
            f"Requested hidden state index {max_i} but only have {len(hs_list)} entries. "
            f"target_layer_ids={target_layer_ids} add_one={add_one_for_pre_layer_capture}"
        )

    picked = [hs_list[i] for i in idxs]
    # [K, batch, seq, hidden] -> [batch, seq, K*hidden]
    picked = [jnp.asarray(x) for x in picked]
    return jnp.concatenate(picked, axis=-1)


@dataclass(frozen=True)
class DFlashStepMetrics:
    accept_len_mean: float
    accept_rate: float


def summarize_accept(
    accept_len, *, block_size: int
) -> DFlashStepMetrics:
    import jax.numpy as jnp

    m = jnp.mean(accept_len.astype(jnp.float32))
    return DFlashStepMetrics(
        accept_len_mean=float(m),
        accept_rate=float(m / max(int(block_size), 1)),
    )
