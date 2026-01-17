from __future__ import annotations

# Back-compat shim: the canonical cache dataset now lives in EasyDeL.

from easydel.trainers.dflash_cache import DFlashTeacherCacheDataset, DFlashTeacherCacheMeta

__all__ = ["DFlashTeacherCacheDataset", "DFlashTeacherCacheMeta"]

