# 20B REAP-lite saliency profile (GPT-OSS MoE)

- Model: `openai/gpt-oss-20b`
- Dataset: `radna0/nemotron-math-v2-harmony-tools` split `high_part00` col `text`
- Domain filter: `` (empty = no filter)
- Rows: 20 | Batch size: 1
- Max seq length: 512
- Layers: 24 | Experts: 32 | Top-k: 4
- Total tokens processed: 10,240 | Kept (assistant-span) tokens: 10,220
- Forward throughput: 1551 tokens/s
- Prompts hash: `0eb2ae71419d5bc43e2c554cf1c7a127e355daa1fa4296b90f65437adf7910d9`

## Top experts by layer (top 10, ranked by saliency_mean)

- layer_0: [23, 13, 25, 14, 31, 30, 16, 9, 19, 1]
- layer_1: [18, 10, 14, 12, 16, 13, 0, 27, 23, 6]
- layer_2: [22, 7, 20, 29, 15, 10, 8, 28, 3, 21]
- layer_3: [10, 22, 12, 7, 21, 2, 13, 23, 19, 31]
- layer_4: [27, 30, 10, 20, 19, 6, 23, 15, 12, 24]
- layer_5: [17, 25, 0, 30, 14, 2, 10, 22, 3, 26]
- layer_6: [5, 26, 4, 16, 20, 11, 31, 2, 3, 21]
- layer_7: [0, 4, 16, 19, 30, 14, 17, 24, 13, 3]
- layer_8: [31, 2, 21, 7, 24, 11, 3, 20, 4, 10]
- layer_9: [31, 17, 15, 19, 0, 28, 4, 30, 22, 5]
- layer_10: [28, 6, 16, 13, 10, 22, 7, 29, 25, 5]
- layer_11: [6, 27, 3, 17, 2, 12, 9, 30, 13, 18]
- layer_12: [23, 19, 11, 25, 27, 6, 14, 13, 28, 7]
- layer_13: [28, 21, 2, 9, 10, 19, 11, 24, 7, 5]
- layer_14: [22, 25, 20, 18, 4, 9, 3, 29, 27, 13]
- layer_15: [14, 10, 5, 31, 7, 13, 0, 29, 9, 16]
- layer_16: [22, 24, 28, 8, 4, 29, 11, 17, 18, 21]
- layer_17: [5, 4, 0, 19, 26, 12, 25, 22, 31, 20]
- layer_18: [30, 7, 6, 15, 11, 10, 23, 4, 0, 26]
- layer_19: [8, 21, 29, 11, 3, 17, 31, 22, 15, 25]
- layer_20: [24, 8, 7, 6, 11, 3, 31, 17, 20, 18]
- layer_21: [6, 11, 28, 14, 8, 1, 18, 27, 2, 30]
- layer_22: [16, 6, 19, 5, 14, 15, 28, 25, 22, 4]
- layer_23: [8, 29, 1, 26, 16, 11, 28, 6, 15, 2]

## Saliency concentration (by gate_norm_sum mass)

| layer | top_4 | top_8 | top_16 |
|---:|---:|---:|---:|
| 0 | 0.539 | 0.690 | 0.880 |
| 1 | 0.433 | 0.678 | 0.887 |
| 2 | 0.605 | 0.759 | 0.916 |
| 3 | 0.608 | 0.778 | 0.931 |
| 4 | 0.619 | 0.820 | 0.963 |
| 5 | 0.480 | 0.694 | 0.943 |
| 6 | 0.807 | 0.898 | 0.966 |
| 7 | 0.417 | 0.642 | 0.901 |
| 8 | 0.546 | 0.776 | 0.952 |
| 9 | 0.569 | 0.792 | 0.970 |
| 10 | 0.601 | 0.822 | 0.981 |
| 11 | 0.644 | 0.892 | 0.986 |
| 12 | 0.670 | 0.878 | 0.980 |
| 13 | 0.741 | 0.894 | 0.986 |
| 14 | 0.520 | 0.739 | 0.939 |
| 15 | 0.456 | 0.727 | 0.908 |
| 16 | 0.537 | 0.758 | 0.944 |
| 17 | 0.538 | 0.766 | 0.967 |
| 18 | 0.586 | 0.760 | 0.918 |
| 19 | 0.536 | 0.715 | 0.907 |
| 20 | 0.580 | 0.780 | 0.939 |
| 21 | 0.613 | 0.795 | 0.955 |
| 22 | 0.584 | 0.799 | 0.951 |
| 23 | 0.677 | 0.839 | 0.944 |

## Artifacts

- `data/20b_reap_saliency.parquet` (per-layer, per-expert count + gate/norm/saliency stats)
- `data/20b_reap_saliency_ranking_by_layer.json` (sorted experts per layer)

## Reproduce

```bash
modal run modal/gpt_oss_pruning_track.py --task reap_saliency_20b --dataset-id radna0/nemotron-math-v2-harmony-tools --dataset-split high_part00 --text-column text --domain  --num-rows 20 --max-seq-length 512 --batch-size 1
```
