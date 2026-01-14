from .configuration_gptoss_dflash import GptOssDFlashConfig

try:  # allow importing this package on CPU-only boxes without torch installed
    import torch as _torch  # noqa: F401
except Exception:  # pragma: no cover
    GptOssDFlashDraftModel = None  # type: ignore[assignment]
else:
    from .modeling_gptoss_dflash import GptOssDFlashDraftModel  # noqa: F401

__all__ = [
    "GptOssDFlashConfig",
    "GptOssDFlashDraftModel",
]
