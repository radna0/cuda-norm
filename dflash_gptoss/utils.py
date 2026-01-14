from __future__ import annotations

from typing import Any


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    # Match https://github.com/z-lab/dflash/blob/main/model/utils.py behavior.
    if num_draft_layers == 1:
        return [int(num_target_layers // 2)]
    start = 1
    end = int(num_target_layers) - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(int(num_draft_layers))
    ]


def extract_context_feature(hidden_states: list[Any], target_layer_ids: list[int]):
    import torch

    feats = []
    for i in target_layer_ids:
        feats.append(hidden_states[int(i) + 1])  # +1 offset (embeddings are hidden_states[0])
    return torch.cat(feats, dim=-1)
