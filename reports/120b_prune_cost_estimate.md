# 120B MoE prune cost estimate (GPT-OSS)

Model: `openai/gpt-oss-120b`

This is a **CPU-only estimate** based on `config.json` (architecture shapes) and an **assumed BF16 parameter size (2 bytes/param)**. Note: the released GPT‑OSS checkpoints are stored in MXFP4 format, so **disk bytes can differ** from BF16 param bytes; this estimate is meant to size **compute and memory movement** for remapping.

## Config summary

- Layers: 36
- Local experts per layer: 128
- Experts per token (top-k): 4
- Hidden size: 2880
- Intermediate size: 2880

## Expert tensor shapes (per layer)

| Tensor | Shape | Params |
|---|---:|---:|
| router.weight | (128, 2880) | 368,640 |
| router.bias | (128,) | 128 |
| experts.gate_up_proj | (128, 2880, 5760) | 2,123,366,400 |
| experts.gate_up_proj_bias | (128, 5760) | 737,280 |
| experts.down_proj | (128, 2880, 2880) | 1,061,683,200 |
| experts.down_proj_bias | (128, 2880) | 368,640 |

## Total expert params/bytes (all layers)

- Expert-related params: 114,714,874,368
- Expert-related bytes (BF16): 213.67 GiB

## Practical prune rewrite sizing

If we process **one layer at a time** (streaming rewrite), the rough peak working-set for expert tensors per layer is:

- Keep 100% experts: ~5.94 GiB
- Keep 50% experts: ~2.97 GiB
- Keep 25% experts: ~1.48 GiB

This excludes framework overhead and any temporary buffers during safetensors read/write.

## IO/wall-time back-of-the-envelope

If a full prune rewrite reads ~213.67 GiB of expert tensors and writes a smaller pruned set:

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
