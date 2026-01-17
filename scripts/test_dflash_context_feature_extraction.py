from __future__ import annotations

import numpy as np


def main() -> None:
    try:
        import jax.numpy as jnp  # type: ignore
    except Exception:
        print("skipped (jax not installed in this environment)", flush=True)
        return

    from dflash_gptoss.easydel_dflash_spec_v1 import (
        extract_dflash_context_features_from_hidden_states,
    )

    # Build fake hidden states list: entry i contains constant i.
    batch, seq, hidden = 2, 3, 4
    hs = [jnp.full((batch, seq, hidden), i, dtype=jnp.float32) for i in range(10)]

    # target_layer_ids refer to HF-style "after layer i"; we add_one to index pre-layer capture.
    target_layer_ids = [1, 3, 6]  # will map to indices 2,4,7
    out = extract_dflash_context_features_from_hidden_states(
        hidden_states=hs, target_layer_ids=target_layer_ids, add_one_for_pre_layer_capture=True
    )
    out_np = np.asarray(out)
    assert out_np.shape == (batch, seq, len(target_layer_ids) * hidden)
    # First hidden block should be 2s, next 4s, next 7s.
    assert np.all(out_np[..., 0:hidden] == 2)
    assert np.all(out_np[..., hidden : 2 * hidden] == 4)
    assert np.all(out_np[..., 2 * hidden : 3 * hidden] == 7)
    print("ok", flush=True)


if __name__ == "__main__":
    main()

