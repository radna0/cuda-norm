from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple


@dataclass
class ToyCache:
    """Minimal KV-cache stand-in with crop support.

    We only track seq_len, which is what the DFlash state machine needs to
    validate commit/crop logic.
    """

    seq_len: int

    def get_seq_length(self) -> int:
        return int(self.seq_len)

    def crop(self, new_seq_len: int) -> None:
        self.seq_len = int(new_seq_len)


@dataclass
class ToyOutput:
    logits: Any  # [B, S, V]
    hidden_states: Tuple[Any, ...]  # tuple of [B, S, H]
    past_key_values: Optional[ToyCache]


class ToyTeacher:
    """Toy causal LM for CPU/JAX end-to-end correctness testing.

    Behavior:
      - next token prediction after token t is (t + 1) % vocab_size
      - hidden_states are simple embeddings (+ layer index) so feature extraction
        works in shape-space.
    """

    def __init__(self, *, vocab_size: int = 128, hidden_size: int = 64, num_hidden_layers: int = 8):
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)

    def get_embedding(self):
        import jax.numpy as jnp

        def _emb(input_ids):
            # [B,S] -> [B,S,H] via first hidden_size dims of one-hot vocab.
            oh = jnp.eye(self.vocab_size, dtype=jnp.float32)[input_ids.astype(jnp.int32)]
            if self.hidden_size == self.vocab_size:
                return oh
            return oh[..., : self.hidden_size]

        return _emb

    def __call__(
        self,
        *,
        input_ids,
        past_key_values: Optional[ToyCache] = None,
        use_cache: bool = True,
        output_hidden_states: bool = True,
        apply_lm_head: bool = True,
        **_: Any,
    ) -> ToyOutput:
        import jax.numpy as jnp

        b, s = int(input_ids.shape[0]), int(input_ids.shape[1])
        base_len = int(past_key_values.seq_len) if past_key_values is not None else 0
        cache = ToyCache(seq_len=base_len + s) if bool(use_cache) else None

        # logits[t] predicts token after input_ids[t].
        if bool(apply_lm_head):
            nxt = (input_ids.astype(jnp.int32) + 1) % int(self.vocab_size)  # [B,S]
            logits = jnp.zeros((b, s, self.vocab_size), dtype=jnp.float32)
            logits = logits.at[jnp.arange(b)[:, None], jnp.arange(s)[None, :], nxt].set(1.0)
        else:
            logits = jnp.zeros((b, s, self.vocab_size), dtype=jnp.float32)

        if bool(output_hidden_states):
            emb = self.get_embedding()(input_ids)  # [B,S,H]
            hss: List[Any] = []
            # pre-layer hidden state + per-layer variants
            for i in range(int(self.num_hidden_layers) + 1):
                hss.append((emb + float(i)).astype(jnp.float32))
            hidden_states = tuple(hss)
        else:
            hidden_states = tuple()

        return ToyOutput(logits=logits, hidden_states=hidden_states, past_key_values=cache)

