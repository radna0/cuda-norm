from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any


@dataclass(frozen=True)
class MoETensorShape:
    name: str
    shape: tuple[int, ...]
    params: int


@dataclass(frozen=True)
class GptOssMoECost:
    model_id: str
    num_layers: int
    num_experts: int
    experts_per_token: int
    hidden_size: int
    intermediate_size: int
    assumed_param_bytes: int
    per_layer: dict[str, MoETensorShape]
    total_expert_params: int
    total_expert_bytes: int

    def human_total_expert_bytes_gib(self) -> float:
        return float(self.total_expert_bytes) / (1024.0**3)


def _require_int(cfg: dict[str, Any], key: str) -> int:
    if key not in cfg:
        raise KeyError(f"Missing required config key: {key}")
    return int(cfg[key])


def estimate_gpt_oss_expert_bytes_from_config(
    cfg: dict[str, Any],
    *,
    model_id: str,
    assumed_param_bytes: int = 2,
) -> GptOssMoECost:
    num_layers = _require_int(cfg, "num_hidden_layers")
    num_experts = _require_int(cfg, "num_local_experts")
    experts_per_token = int(cfg.get("num_experts_per_tok", cfg.get("experts_per_token", 4)))
    hidden_size = _require_int(cfg, "hidden_size")
    intermediate_size = _require_int(cfg, "intermediate_size")

    if hidden_size <= 0 or intermediate_size <= 0 or num_layers <= 0 or num_experts <= 0:
        raise ValueError("Invalid GPT-OSS config values for MoE cost estimate.")

    # GPT-OSS MLP uses a fused SwiGLU-like "gate_up" projection producing 2*intermediate.
    gate_up_proj = num_experts * hidden_size * (2 * intermediate_size)
    gate_up_proj_bias = num_experts * (2 * intermediate_size)
    down_proj = num_experts * intermediate_size * hidden_size
    down_proj_bias = num_experts * hidden_size
    router_weight = num_experts * hidden_size
    router_bias = num_experts

    per_layer = {
        "router.weight": MoETensorShape(
            name="router.weight",
            shape=(num_experts, hidden_size),
            params=router_weight,
        ),
        "router.bias": MoETensorShape(
            name="router.bias",
            shape=(num_experts,),
            params=router_bias,
        ),
        "experts.gate_up_proj": MoETensorShape(
            name="experts.gate_up_proj",
            shape=(num_experts, hidden_size, 2 * intermediate_size),
            params=gate_up_proj,
        ),
        "experts.gate_up_proj_bias": MoETensorShape(
            name="experts.gate_up_proj_bias",
            shape=(num_experts, 2 * intermediate_size),
            params=gate_up_proj_bias,
        ),
        "experts.down_proj": MoETensorShape(
            name="experts.down_proj",
            shape=(num_experts, intermediate_size, hidden_size),
            params=down_proj,
        ),
        "experts.down_proj_bias": MoETensorShape(
            name="experts.down_proj_bias",
            shape=(num_experts, hidden_size),
            params=down_proj_bias,
        ),
    }

    per_layer_total = sum(x.params for x in per_layer.values())
    total_expert_params = per_layer_total * num_layers
    total_expert_bytes = total_expert_params * int(assumed_param_bytes)

    return GptOssMoECost(
        model_id=str(model_id),
        num_layers=int(num_layers),
        num_experts=int(num_experts),
        experts_per_token=int(experts_per_token),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        assumed_param_bytes=int(assumed_param_bytes),
        per_layer=per_layer,
        total_expert_params=int(total_expert_params),
        total_expert_bytes=int(total_expert_bytes),
    )


def estimate_layer_working_set_gib(
    cost: GptOssMoECost,
    *,
    keep_expert_fraction: float = 1.0,
    include_router: bool = True,
) -> float:
    if keep_expert_fraction <= 0 or keep_expert_fraction > 1:
        raise ValueError("keep_expert_fraction must be in (0, 1].")

    kept = int(ceil(cost.num_experts * keep_expert_fraction))
    kept = max(1, min(cost.num_experts, kept))

    # Working set for pruning a single layer: gate_up + down + biases (+ router).
    per_layer_params = (
        kept * cost.hidden_size * (2 * cost.intermediate_size)
        + kept * (2 * cost.intermediate_size)
        + kept * cost.intermediate_size * cost.hidden_size
        + kept * cost.hidden_size
    )
    if include_router:
        per_layer_params += kept * cost.hidden_size + kept
    return (per_layer_params * cost.assumed_param_bytes) / (1024.0**3)
