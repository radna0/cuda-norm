#!/usr/bin/env python3
"""CPU-only parity test for chunked CE (DFlash trainer).

This guards against a subtle but catastrophic bug: streaming logsumexp must
rescale the running sum when the running max increases. Without rescaling,
loss is severely underestimated and training appears to "converge" while the
draft is actually wrong.
"""

from __future__ import annotations


def main() -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax

    from easydel.trainers.dflash_trainer import _chunked_ce_nll_and_acc

    key = jax.random.PRNGKey(0)
    b, s, h, v = 4, 7, 16, 256

    key, k1, k2, k3 = jax.random.split(key, 4)
    hs = jax.random.normal(k1, (b, s, h), dtype=jnp.float32).astype(jnp.bfloat16)
    lm_w = jax.random.normal(k2, (v, h), dtype=jnp.float32).astype(jnp.bfloat16)
    labels = jax.random.randint(k3, (b, s), minval=0, maxval=v, dtype=jnp.int32)

    # Full reference (float32 logits) to match the chunked path's math.
    logits32 = jnp.einsum(
        "bsh,vh->bsv",
        hs.astype(jnp.bfloat16),
        lm_w,
        precision=jax.lax.Precision.HIGHEST,
    ).astype(jnp.float32)
    ref_loss32 = optax.softmax_cross_entropy_with_integer_labels(logits32, labels).mean()
    ref_acc32 = jnp.mean((jnp.argmax(logits32, axis=-1).astype(jnp.int32) == labels).astype(jnp.float32))

    # Full reference (bf16 logits) to match the unchunked fast path.
    logits16 = logits32.astype(jnp.bfloat16)
    ref_loss16 = optax.softmax_cross_entropy_with_integer_labels(logits16, labels).mean()
    ref_acc16 = jnp.mean((jnp.argmax(logits16, axis=-1).astype(jnp.int32) == labels).astype(jnp.float32))

    # Chunked should match.
    for chunk in (1, 8, 32, 64, 128):
        loss, acc = _chunked_ce_nll_and_acc(hs=hs, labels=labels, lm_w=lm_w, vocab_chunk=int(chunk))
        loss_f = float(loss)
        acc_f = float(acc)
        if not np.isfinite(loss_f):
            raise AssertionError(f"chunk={chunk}: non-finite loss: {loss_f}")
        if abs(loss_f - float(ref_loss32)) > 1e-4:
            raise AssertionError(
                f"chunk={chunk}: loss mismatch {loss_f} vs ref {float(ref_loss32)}"
            )
        if abs(acc_f - float(ref_acc32)) > 1e-6:
            raise AssertionError(
                f"chunk={chunk}: acc mismatch {acc_f} vs ref {float(ref_acc32)}"
            )

    # Non-chunked path should also match.
    loss0, acc0 = _chunked_ce_nll_and_acc(hs=hs, labels=labels, lm_w=lm_w, vocab_chunk=0)
    if abs(float(loss0) - float(ref_loss16)) > 1e-6:
        raise AssertionError(f"chunk=0: loss mismatch {float(loss0)} vs ref {float(ref_loss16)}")
    if abs(float(acc0) - float(ref_acc16)) > 1e-6:
        raise AssertionError(f"chunk=0: acc mismatch {float(acc0)} vs ref {float(ref_acc16)}")

    print("[+] chunked CE parity ok", flush=True)


if __name__ == "__main__":
    main()
