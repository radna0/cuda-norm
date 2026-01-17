from __future__ import annotations

import numpy as np


def _ref_accept_len_and_bonus(candidates: np.ndarray, target_predict: np.ndarray):
    # Mirror SGLang torch logic:
    # matches = candidates[:, 1:] == target_predict[:, :-1]
    # accept_len = cumprod(matches).sum(axis=1)
    bs, b = candidates.shape
    matches = (candidates[:, 1:] == target_predict[:, :-1]).astype(np.int32)
    acc = np.cumprod(matches, axis=1).sum(axis=1).astype(np.int32)
    bonus = target_predict[np.arange(bs), acc].astype(np.int32)
    return acc, bonus


def main() -> None:
    try:
        import jax.numpy as jnp  # type: ignore
    except Exception:
        print("skipped (jax not installed in this environment)", flush=True)
        return

    from dflash_gptoss.easydel_dflash_spec_v1 import dflash_accept_len_and_bonus

    rng = np.random.default_rng(0)
    for bs in [1, 2, 8]:
        for b in [2, 8, 16]:
            cand = rng.integers(0, 100, size=(bs, b), dtype=np.int32)
            targ = rng.integers(0, 100, size=(bs, b), dtype=np.int32)

            # Force a few controllable match prefixes.
            if b >= 4:
                cand[:, 1] = targ[:, 0]  # accept >=1
                cand[:, 2] = targ[:, 1]  # accept >=2
                cand[:, 3] = targ[:, 2]  # accept >=3
                cand[0, 4:] = rng.integers(0, 100, size=(b - 4,), dtype=np.int32)

            ref_a, ref_bonus = _ref_accept_len_and_bonus(cand, targ)
            a, bonus = dflash_accept_len_and_bonus(
                candidates=jnp.asarray(cand),
                target_predict=jnp.asarray(targ),
            )
            a_np = np.asarray(a)
            b_np = np.asarray(bonus)
            assert np.array_equal(a_np, ref_a), (bs, b, a_np, ref_a)
            assert np.array_equal(b_np, ref_bonus), (bs, b, b_np, ref_bonus)

    print("ok", flush=True)


if __name__ == "__main__":
    main()
