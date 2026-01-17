#!/usr/bin/env python3
"""CPU/JAX end-to-end correctness test for the DFlash block-verify state machine.

This test validates:
- accept_len + bonus rule (SGLang semantics)
- cache crop/commit length bookkeeping
- generated token stream matches baseline greedy when verification is greedy

It uses a ToyTeacher with deterministic next-token rule.
"""

from __future__ import annotations


def main() -> None:
    import jax.numpy as jnp

    from dflash_gptoss.easydel_dflash_spec_v1 import dflash_accept_len_and_bonus
    from dflash_gptoss.toy_teacher_jax import ToyTeacher

    teacher = ToyTeacher(vocab_size=64, hidden_size=64, num_hidden_layers=8)

    block_size = 8
    max_new = 64

    # Prompt (prefix + current token).
    prompt = jnp.asarray([[1, 2, 3, 4, 5]], dtype=jnp.int32)
    prefix = prompt[:, :-1]
    cur = prompt[:, -1:]

    out_prefill = teacher(input_ids=prefix, use_cache=True, output_hidden_states=True, apply_lm_head=False)
    cache = out_prefill.past_key_values
    assert cache is not None
    base_len0 = int(cache.get_seq_length())
    assert base_len0 == int(prefix.shape[1])

    # Baseline greedy decode from the same starting point:
    # first step consumes `cur`, then emits next tokens.
    def baseline_tokens(cur_tok, n):
        toks = []
        t = int(cur_tok)
        for _ in range(int(n)):
            t = (t + 1) % 64
            toks.append(t)
        return toks

    baseline = baseline_tokens(int(cur[0, 0]), max_new)

    produced = []

    # Draft proposer: propose the correct sequence, but inject a mismatch every few blocks
    # to force early accept and test crop.
    mismatch_every = 3

    for step in range(max_new):
        # Build candidate block: [cur] + draft tokens
        # draft token j should match teacher's pred after token (cur + j accepted).
        # teacher pred after token t is (t+1) mod V.
        t0 = int(cur[0, 0])
        draft = []
        t = t0
        for j in range(block_size - 1):
            t = (t + 1) % 64
            draft.append(t)
        # Inject mismatch at position 2 (third drafted token) periodically.
        if (step // (block_size - 1)) % mismatch_every == 1:
            if len(draft) >= 3:
                draft[2] = (draft[2] + 7) % 64

        cand = jnp.asarray([[t0] + draft], dtype=jnp.int32)  # [1,B]

        base_len = int(cache.get_seq_length())
        out_v = teacher(
            input_ids=cand,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=False,
            apply_lm_head=True,
        )
        cache_full = out_v.past_key_values
        assert cache_full is not None

        target_predict = jnp.argmax(out_v.logits, axis=-1).astype(jnp.int32)
        accept_len, bonus = dflash_accept_len_and_bonus(candidates=cand, target_predict=target_predict)
        n_acc = int(accept_len[0])
        keep_in_block = 1 + n_acc

        # Crop cache to committed length (prefix + keep_in_block)
        cache_full.crop(base_len + keep_in_block)
        cache = cache_full
        assert int(cache.get_seq_length()) == base_len + keep_in_block

        # Commit accepted draft tokens (n_acc tokens), then emit bonus as next current token.
        for j in range(n_acc):
            produced.append(int(cand[0, 1 + j]))
            if len(produced) >= max_new:
                break
        cur = bonus[:, None].astype(jnp.int32)
        produced.append(int(cur[0, 0]))
        if len(produced) >= max_new:
            break

    produced = produced[:max_new]
    assert produced == baseline, (produced[:32], baseline[:32])
    print("[OK] DFlash toy block-verify parity matches baseline greedy", flush=True)


if __name__ == "__main__":
    main()

