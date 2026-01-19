# Qwen3 2M embedding manifold analysis

- generated_at: 2026-01-13 15:51:53 +0000

## What to open (full 2M, interactive 3D)

- Prompt view pointcloud: `harmony/cuda-norm/artifacts/map_view/qwen3_prompt_full_pca_2m_20260112/pointcloud_2m_3d.html`
- Behavior view pointcloud: `harmony/cuda-norm/artifacts/map_view/qwen3_behavior_full_pca_2m_20260112/pointcloud_2m_3d.html`
- Prompt density: `harmony/cuda-norm/artifacts/map_view/qwen3_prompt_full_pca_2m_20260112/density_view.html`
- Behavior density: `harmony/cuda-norm/artifacts/map_view/qwen3_behavior_full_pca_2m_20260112/density_view.html`

## What to open (deep reasoning excerpt v2, 300k)

- Excerpt PCA pointcloud: `harmony/cuda-norm/artifacts/map_view/qwen3_reasoning_excerpt_v2_full_pca_300k_20260113_212547/pointcloud_300k_pca_3d_dark.html`
- Excerpt PCA density: `harmony/cuda-norm/artifacts/map_view/qwen3_reasoning_excerpt_v2_full_pca_300k_20260113_212547/density_view.html`
- Excerpt UMAP pointcloud: `harmony/cuda-norm/artifacts/map_view/qwen3_reasoning_excerpt_v2_umap_300k_20260113_212547/pointcloud_300k_umap_3d_dark.html`
- Excerpt UMAP density: `harmony/cuda-norm/artifacts/map_view/qwen3_reasoning_excerpt_v2_umap_300k_20260113_212547/density_view.html`
- Excerpt UMAP report: `harmony/cuda-norm/artifacts/map_view/qwen3_reasoning_excerpt_v2_umap_300k_20260113_212547/umap_report_300k.md`

## Key findings (from quantitative diagnostics)

- Prompt view: pc1=7.69% occupancy=6.11% nn1>=0.99=33.24% nn1>=0.999=32.56%
  - kmeans(k=1000): effective_clusters≈936.73 gini≈0.2004
- Behavior view: pc1=14.85% occupancy=7.31% nn1>=0.99=1.00% nn1>=0.999=0.31%
  - kmeans(k=1000): effective_clusters≈889.95 gini≈0.2631

Deep reasoning excerpt v2 (300k):
- Excerpt view: pc1=15.34% occupancy=8.17% nn1>=0.99=0.00% nn1>=0.999=0.00% (kNN sample n=200k)
  - kmeans(k=1000): effective_clusters≈891.45 gini≈0.2583

Interpretation:
- “Two blobs” in PCA/UMAP is not inherently bad; it often reflects major modes (dataset/style). The real question is redundancy *within* blobs and whether each mode has sufficient internal structure.
- If nn1>=0.99 is high, compress via medoids/LSH/k-means; this preserves coverage and saves compute.

## Prompt view

- embedding_dir: `/dev/shm/hf_qwen3_prompt_embed_20260112/runs/qwen3_bf16_prompt_dp8_trtllm_mha_kvbf16_m1024_bs1024_2m_20260112`
- PCA variance: pc1=7.69% pc1-3=15.68% (pc1 dominance = collapse indicator)
- PCA voxel occupancy (xyz, mean over dataset×mix_group): 6.11%
- kNN redundancy (sample n=200000): nn1>=0.99=33.24%, nn1>=0.999=32.56%
- Neighborhood purity (k=10, embedding space): dataset=90.98%, mix_group=72.48%, len_bucket=51.70%
- UMAP overlap@10 (local structure preserved): 0.1590

**Top density groups (xyz, by points)**

| dataset | mix_group | points | voxels | avg pts/voxel |
|---|---:|---:|---:|---:|
| nvidia/Nemotron-Math-v2 | tool | 589051 | 111510 | 5.28 |
| nvidia/Nemotron-Math-Proofs-v1 | reasoning | 379296 | 143737 | 2.64 |
| nvidia/Nemotron-Math-v2 | reasoning | 370003 | 87660 | 4.22 |
| nvidia/Nemotron-Math-v2 | general | 340094 | 120135 | 2.83 |
| nvidia/Nemotron-Agentic-v1 | tool | 210949 | 55552 | 3.80 |
| nvidia/Nemotron-Instruction-Following-Chat-v1 | reasoning | 49386 | 28283 | 1.75 |
| nvidia/Nemotron-Instruction-Following-Chat-v1 | general | 29140 | 21030 | 1.39 |
| nvidia/Nemotron-Science-v1 | general | 16700 | 13802 | 1.21 |
| nvidia/Nemotron-Math-Proofs-v1 | general | 9966 | 9471 | 1.05 |
| nvidia/Nemotron-Agentic-v1 | general | 4100 | 2674 | 1.53 |

## Behavior view

- embedding_dir: `/dev/shm/hf_home/hub/datasets--radna0--harmony-qwen3-embeddings-2m/snapshots/a6891198de396cf48275729cca45aa50d5ad86b1/behavior_v5_trace_sketch_full_20260113_044747`
- PCA variance: pc1=14.85% pc1-3=24.99% (pc1 dominance = collapse indicator)
- PCA voxel occupancy (xyz, mean over dataset×mix_group): 7.31%
- kNN redundancy (sample n=187478): nn1>=0.99=1.00%, nn1>=0.999=0.31%
- Neighborhood purity (k=10, embedding space): dataset=94.46%, mix_group=65.06%, len_bucket=72.58%
- UMAP overlap@10 (local structure preserved): 0.1013

**Top density groups (xyz, by points)**

| dataset | mix_group | points | voxels | avg pts/voxel |
|---|---:|---:|---:|---:|
| nvidia/Nemotron-Math-v2 | tool | 589051 | 169365 | 3.48 |
| nvidia/Nemotron-Math-Proofs-v1 | reasoning | 379296 | 85146 | 4.45 |
| nvidia/Nemotron-Math-v2 | reasoning | 370003 | 141011 | 2.62 |
| nvidia/Nemotron-Math-v2 | general | 340094 | 137357 | 2.48 |
| nvidia/Nemotron-Agentic-v1 | tool | 210949 | 91049 | 2.32 |
| nvidia/Nemotron-Instruction-Following-Chat-v1 | reasoning | 49386 | 36374 | 1.36 |
| nvidia/Nemotron-Instruction-Following-Chat-v1 | general | 29140 | 25062 | 1.16 |
| nvidia/Nemotron-Science-v1 | general | 16700 | 15426 | 1.08 |
| nvidia/Nemotron-Math-Proofs-v1 | general | 9966 | 5570 | 1.79 |
| nvidia/Nemotron-Agentic-v1 | general | 4100 | 3890 | 1.05 |

## Recommended next steps (quality-first)

1) Use the prompt view for coverage-driven packs: compress with LSH/k-means into a 200k cover, then derive 1k/10k/100k packs and SWA/full variants.
2) For agentic/tool behavior, select from the behavior view with explicit per-tool-sequence quotas (coverage over tool policies, not just dataset name).
3) For deep reasoning diversity, build a dedicated reasoning-excerpt view on a 50k–300k subset (1024–2048 tokens), embed, then select a reasoning_style_10k pack by clustering.
4) Track KPIs (dup rates, voxel occupancy, k-means gini/effective_clusters) for each new view so decisions are backed by numbers, not just plots.

## Latest pack outputs

- HF repo (private): `radna0/harmony-qwen3-calib-packs-v2-20260113`
- Tool/agentic: `packs/tool_agentic_10k_v6` (behavior-v5, strict tool-seq quotas)
- Prompt coverage: `packs/prompt_*_v2` (+ SWA/full/tool-only/agentic-only variants)
- Deep reasoning style: `packs/reasoning_style_10k_v2` (selected from excerpt embeddings; current build uses complete shards 0–7 and includes `NOTE_FULL_EXCERPT_EMBEDDINGS.txt`)
