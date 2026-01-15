"""
`dflash_gptoss` package init.

Important: keep this import-light.

We run on Kaggle images where optional binary deps (TensorFlow / matplotlib /
sklearn) can be ABI-incompatible with the system NumPy. Importing Transformers
at module import time can indirectly import those stacks and crash.

If you need the HF config/model classes, import them explicitly:
  - `from dflash_gptoss.configuration_gptoss_dflash import GptOssDFlashConfig`
  - `from dflash_gptoss.modeling_gptoss_dflash import GptOssDFlashDraftModel`
"""

from __future__ import annotations

from typing import Any

__all__ = ["GptOssDFlashConfig", "GptOssDFlashDraftModel"]


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name == "GptOssDFlashConfig":
        from .configuration_gptoss_dflash import GptOssDFlashConfig

        return GptOssDFlashConfig
    if name == "GptOssDFlashDraftModel":
        from .modeling_gptoss_dflash import GptOssDFlashDraftModel

        return GptOssDFlashDraftModel
    raise AttributeError(name)
