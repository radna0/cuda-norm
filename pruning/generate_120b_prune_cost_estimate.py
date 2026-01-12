from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from huggingface_hub import hf_hub_download

from pruning.gpt_oss_moe_cost import estimate_gpt_oss_expert_bytes_from_config
from pruning.gpt_oss_moe_cost import estimate_layer_working_set_gib


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _fmt_gib(x: float) -> str:
    return f"{x:.2f} GiB"


def main() -> None:
    model_id = "openai/gpt-oss-120b"
    out_path = Path("reports/120b_prune_cost_estimate.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg_path = hf_hub_download(model_id, filename="config.json")
    cfg = _read_json(cfg_path)

    cost = estimate_gpt_oss_expert_bytes_from_config(cfg, model_id=model_id, assumed_param_bytes=2)

    layer_full = estimate_layer_working_set_gib(cost, keep_expert_fraction=1.0)
    layer_50 = estimate_layer_working_set_gib(cost, keep_expert_fraction=0.5)
    layer_25 = estimate_layer_working_set_gib(cost, keep_expert_fraction=0.25)

    md = f"""# 120B MoE prune cost estimate (GPT-OSS)

Model: `{model_id}`

This is a **CPU-only estimate** based on `config.json` (architecture shapes) and an **assumed BF16 parameter size (2 bytes/param)**. Note: the released GPT‑OSS checkpoints are stored in MXFP4 format, so **disk bytes can differ** from BF16 param bytes; this estimate is meant to size **compute and memory movement** for remapping.

## Config summary

- Layers: {cost.num_layers}
- Local experts per layer: {cost.num_experts}
- Experts per token (top-k): {cost.experts_per_token}
- Hidden size: {cost.hidden_size}
- Intermediate size: {cost.intermediate_size}

## Expert tensor shapes (per layer)

| Tensor | Shape | Params |
|---|---:|---:|
| router.weight | ({cost.num_experts}, {cost.hidden_size}) | {_fmt_int(cost.per_layer['router.weight'].params)} |
| router.bias | ({cost.num_experts},) | {_fmt_int(cost.per_layer['router.bias'].params)} |
| experts.gate_up_proj | ({cost.num_experts}, {cost.hidden_size}, {2*cost.intermediate_size}) | {_fmt_int(cost.per_layer['experts.gate_up_proj'].params)} |
| experts.gate_up_proj_bias | ({cost.num_experts}, {2*cost.intermediate_size}) | {_fmt_int(cost.per_layer['experts.gate_up_proj_bias'].params)} |
| experts.down_proj | ({cost.num_experts}, {cost.intermediate_size}, {cost.hidden_size}) | {_fmt_int(cost.per_layer['experts.down_proj'].params)} |
| experts.down_proj_bias | ({cost.num_experts}, {cost.hidden_size}) | {_fmt_int(cost.per_layer['experts.down_proj_bias'].params)} |

## Total expert params/bytes (all layers)

- Expert-related params: {_fmt_int(cost.total_expert_params)}
- Expert-related bytes (BF16): {_fmt_gib(cost.human_total_expert_bytes_gib())}

## Practical prune rewrite sizing

If we process **one layer at a time** (streaming rewrite), the rough peak working-set for expert tensors per layer is:

- Keep 100% experts: ~{_fmt_gib(layer_full)}
- Keep 50% experts: ~{_fmt_gib(layer_50)}
- Keep 25% experts: ~{_fmt_gib(layer_25)}

This excludes framework overhead and any temporary buffers during safetensors read/write.

## IO/wall-time back-of-the-envelope

If a full prune rewrite reads ~{_fmt_gib(cost.human_total_expert_bytes_gib())} of expert tensors and writes a smaller pruned set:

- At 0.5 GiB/s sustained IO, 214 GiB read is ~7.1 min (plus write time).
- At 1.0 GiB/s sustained IO, 214 GiB read is ~3.6 min (plus write time).

Actual time will depend on:
- whether the MXFP4 expert tensors are co-located in a few shards or spread across many files
- CPU decompression / safetensors parsing overhead
- filesystem / volume performance

## Reproduce

Generate this report:

```bash
python3 -m pruning.generate_120b_prune_cost_estimate
```
"""

    out_path.write_text(md, encoding="utf-8")

    meta_path = Path("reports/120b_prune_cost_estimate.json")
    meta_path.write_text(json.dumps(asdict(cost), indent=2, sort_keys=True), encoding="utf-8")

    print(f"[+] Wrote {out_path}")
    print(f"[+] Wrote {meta_path}")


if __name__ == "__main__":
    main()
