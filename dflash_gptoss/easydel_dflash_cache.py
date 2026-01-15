from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DFlashTeacherCacheMeta:
    model_snapshot_dir: str
    platform: str
    ctx_len: int
    block_size: int
    num_blocks: int
    batch_size: int
    target_layer_ids: list[int]
    hidden_size: int
    num_context_features: int
    dtype: str
    seed: int
    calib_repo_id: str
    calib_data_files: list[str]
    max_rows_per_pack: int
    wall_s: float

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "DFlashTeacherCacheMeta":
        return cls(
            model_snapshot_dir=str(d["model_snapshot_dir"]),
            platform=str(d.get("platform", "")),
            ctx_len=int(d["ctx_len"]),
            block_size=int(d["block_size"]),
            num_blocks=int(d["num_blocks"]),
            batch_size=int(d.get("batch_size", 0)),
            target_layer_ids=[int(x) for x in d["target_layer_ids"]],
            hidden_size=int(d["hidden_size"]),
            num_context_features=int(d.get("num_context_features", len(d["target_layer_ids"]))),
            dtype=str(d.get("dtype", "")),
            seed=int(d.get("seed", 0)),
            calib_repo_id=str(d.get("calib_repo_id", "")),
            calib_data_files=[str(x) for x in d.get("calib_data_files", [])],
            max_rows_per_pack=int(d.get("max_rows_per_pack", 0)),
            wall_s=float(d.get("wall_s", 0.0)),
        )


class DFlashTeacherCacheDataset:
    """Memory-mapped dataset for DFlash cached teacher features.

    Exposes rows as dicts with:
      - context_features_u16: uint16 [ctx_len, K*hidden]
      - anchor_embedding_u16: uint16 [hidden]
      - target_ids: int32 [block_size-1]
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir).resolve()
        meta_path = self.cache_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing {meta_path}")
        self.meta = DFlashTeacherCacheMeta.from_json(json.loads(meta_path.read_text(encoding="utf-8")))

        self._context = np.load(self.cache_dir / "context_features_u16.npy", mmap_mode="r")
        self._anchor = np.load(self.cache_dir / "anchor_embedding_u16.npy", mmap_mode="r")
        self._targets = np.load(self.cache_dir / "target_ids.npy", mmap_mode="r")

        if self._context.dtype != np.uint16:
            raise TypeError(f"context_features_u16.npy must be uint16, got {self._context.dtype}")
        if self._anchor.dtype != np.uint16:
            raise TypeError(f"anchor_embedding_u16.npy must be uint16, got {self._anchor.dtype}")
        if self._targets.dtype != np.int32:
            self._targets = self._targets.astype(np.int32, copy=False)

        n = int(self._context.shape[0])
        if int(self._anchor.shape[0]) != n or int(self._targets.shape[0]) != n:
            raise ValueError("Cache arrays have mismatched first dimension.")

    def __len__(self) -> int:
        return int(self._context.shape[0])

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return {
            "context_features_u16": self._context[int(idx)],
            "anchor_embedding_u16": self._anchor[int(idx)],
            "target_ids": self._targets[int(idx)],
        }

    def get_batch(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        """Vectorized batch fetch (fast path).

        `indices` must be a 1D int array. Returns contiguous numpy arrays with
        shape [B, ...] suitable for the trainer data pipeline.
        """
        idx = np.asarray(indices, dtype=np.int64).reshape((-1,))
        return {
            "context_features_u16": np.asarray(self._context[idx], dtype=np.uint16),
            "anchor_embedding_u16": np.asarray(self._anchor[idx], dtype=np.uint16),
            "target_ids": np.asarray(self._targets[idx], dtype=np.int32),
        }
