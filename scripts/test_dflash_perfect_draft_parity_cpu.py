from __future__ import annotations

import numpy as np


def _ref_accept_len_and_bonus(candidates: np.ndarray, target_predict: np.ndarray):
    matches = (candidates[:, 1:] == target_predict[:, :-1]).astype(np.int32)
    accept_len = np.cumprod(matches, axis=1).sum(axis=1).astype(np.int32)
    bonus = target_predict[np.arange(candidates.shape[0]), accept_len].astype(np.int32)
    return accept_len, bonus


def main() -> None:
    rng = np.random.default_rng(0)

    bs = 4
    block_size = 8
    steps = 10
    vocab = 1000

    # Baseline greedy sequence: append one token each step.
    baseline = [[] for _ in range(bs)]
    # DFlash sequence: append up to B tokens per step.
    dflash = [[] for _ in range(bs)]

    for _ in range(steps):
        # Target predicts B tokens for the verify window (current+block-1) positions.
        target_predict = rng.integers(0, vocab, size=(bs, block_size), dtype=np.int32)

        # Perfect draft: candidates[1:] exactly match target_predict[:-1]
        current = rng.integers(0, vocab, size=(bs, 1), dtype=np.int32)
        candidates = np.concatenate([current, target_predict[:, :-1]], axis=1)

        accept_len, bonus = _ref_accept_len_and_bonus(candidates, target_predict)
        assert np.all(accept_len == (block_size - 1))

        # DFlash commits: accept all draft tokens then bonus => target_predict[0..B-1]
        for i in range(bs):
            dflash[i].extend(target_predict[i].tolist())

        # Baseline commits: one token at a time, B tokens total per DFlash step.
        for t in range(block_size):
            for i in range(bs):
                baseline[i].append(int(target_predict[i, t]))

    for i in range(bs):
        assert baseline[i] == dflash[i]

    print("ok", flush=True)


if __name__ == "__main__":
    main()

