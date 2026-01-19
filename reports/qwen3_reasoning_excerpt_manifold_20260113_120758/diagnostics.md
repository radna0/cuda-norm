# qwen3 reasoning_excerpt (n=154868, max_tokens=2048)

- generated_at: 2026-01-13 12:08:02 +0000
- embedding_dir: /dev/shm/qwen3_reasoning_excerpt_v1_20260113_112510_hf/qwen3_reasoning_excerpt_v1_20260113_112510

## PCA (full)

- rows: 154868
- explained_var_pc1: 0.12027633295713332
- explained_var_pc1_3: 0.19720839680405677

## Density (PCA space, xyz voxels)

- grid3: 96 (max_voxels=884736)
- mean_occupancy: 0.01798728660300926

| dataset | meta_domain | meta_difficulty_bin | len_bucket | points | voxels | avg_pts/voxel | occupancy |
|---|---|---|---|---:|---:|---:|---:|
| nvidia/Nemotron-Math-v2 | math | high | 1792_2048 | 150320 | 122995 | 1.22 | 0.13901887116608797 |
| nvidia/Nemotron-Math-v2 | math | high | 1536_1792 | 1846 | 1721 | 1.07 | 0.001945213035300926 |
| nvidia/Nemotron-Math-v2 | math | high | 1280_1536 | 1459 | 1400 | 1.04 | 0.001582392939814815 |
| nvidia/Nemotron-Math-v2 | math | high | 1024_1280 | 1171 | 1124 | 1.04 | 0.001270435474537037 |
| nvidia/Nemotron-Math-Proofs-v1 | proof | unknown | 1024_1280 | 54 | 54 | 1.00 | 6.103515625e-05 |
| nvidia/Nemotron-Math-Proofs-v1 | proof | unknown | 1280_1536 | 12 | 12 | 1.00 | 1.3563368055555555e-05 |
| nvidia/Nemotron-Math-Proofs-v1 | proof | unknown | 1536_1792 | 3 | 3 | 1.00 | 3.3908420138888887e-06 |
| nvidia/Nemotron-Math-Proofs-v1 | proof | unknown | 1792_2048 | 3 | 3 | 1.00 | 3.3908420138888887e-06 |

## UMAP local-structure report (200k fit, 50k eval)

| k | overlap | purity(dataset) | purity(domain) | purity(mix_group) | purity(difficulty) | purity(len_bucket) |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.0856 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 30 | 0.1256 | 0.9998 | 0.9998 | 1.0000 | 0.9998 | 1.0000 |
| 50 | 0.1463 | 0.9997 | 0.9997 | 1.0000 | 0.9997 | 1.0000 |

## kNN redundancy + purity (embedding space, sampled)

- n(sample): 154868 k=10
- nn1 cosine quantiles: {'0.0': 0.2841179370880127, '0.01': 0.5031711351871491, '0.1': 0.6254967093467713, '0.5': 0.7478551268577576, '0.9': 0.8516473591327667, '0.99': 0.9109769231081009, '1.0': 0.9714435338973999}
- nn1 near-dup rates: {'0.99': 0.0, '0.995': 0.0, '0.999': 0.0, '0.9995': 0.0}

Mean neighbor-label purity (kNN@k, exclude self):
- dataset: 1.0000
- meta_difficulty_bin: 1.0000
- meta_domain: 1.0000
- mix_group: 1.0000

## Full 2M label counts (top10)

- dataset: nvidia/Nemotron-Math-v2=154796, nvidia/Nemotron-Math-Proofs-v1=72
- mix_group: reasoning=154868
- meta_domain: math=154796, proof=72
- difficulty_bin: high=154796, unknown=72
- len_bucket: 1792_2048=150323, 1536_1792=1849, 1280_1536=1471, 1024_1280=1225
