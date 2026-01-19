# Pruning Track TODOs (GPT‑OSS MoE) — EAFT‑REAP (NO finetune)

**Goal (100%)**: structural-prune `openai/gpt-oss-20b` with **near-lossless quality** at:
- **keep_frac=0.75 (keep_n=24/32)**, `top_k=4` unchanged
- **keep_frac=0.60 target**, evaluated as **keep_n=20/32 (0.625)** due to the `keep_n % 4 == 0` kernel constraint, `top_k=4` unchanged

**Guardrails**
- No finetune / LoRA / recovery training / distillation.
- Do not change `top_k` for the “lossless pruning” goal (keep `top_k=4`).
- All layers must keep the **same** `keep_n` (GPT‑OSS constraint).
- Canonical quality metric: **EAFT parity** (completion-only PPL + CC-rate + distribution shift JS2D) on curated calib packs.
  - Practical note: SGLang’s FlashInfer MXFP4 fused MoE routing kernels may require `keep_n % 4 == 0`. If that constraint is enforced, “0.60” rounds to **keep_n=20/32 (0.625)** for kernel compatibility.

**Current overall progress**: **~82%**
- 0.75: **PASS** under the canonical high-token-budget regime (1024+2048), with **noise-floor** and **noop rewrite** validated.
- 0.75 long-seq validation (4096/8192/16384; token budget ~1M/pack): **DONE** (`harmony/cuda-norm/reports/eaftreap75_longseq_tokenbudget1m_keep24_uniform.md`).
- 0.75 long-seq validation (same matrix; token budget ~10M/pack): **planned** (`harmony/cuda-norm/scripts/kaggle_eaftreap75_longseq_tokenbudget10m_keep24_uniform.sh`).
- 0.60: **planned** (**keep_n=20/32 = 0.625 actual**), **gate eval pending** under the same regime.

**Compute note**: Modal profiles have been intermittently blocked by spend limits. The current work is running on Kaggle via Versa (remote Jupyter `/proxy`) for GPU steps, with CPU-side analysis on this workstation.

**Kaggle/VERSA stability note**: Versa re-sync can replace the remote `harmony/cuda-norm` code directory. All pruning + EAFT outputs must be written under `/kaggle/working/{artifacts,reports,logs}` (not under the synced code dir) to avoid losing pruned checkpoints between jobs. This is now enforced by the runners + scripts.

**MXFP4 note (do not ignore)**: GPT‑OSS MXFP4 checkpoints require `triton>=3.4.0` and `kernels` installed, otherwise Transformers may dequantize to BF16 and OOM. The Kaggle runners now install and verify `triton==3.4.0` + `kernels==0.11.7` up front.
**Profiling note (EAFT‑REAP specific)**: EAFT‑REAP requires *direct* access to expert projection weights as real PyTorch tensors (for per-expert matmul norm estimates). When MXFP4 loads with kernels enabled, Transformers may wrap weights as `triton_kernels.*.tensor.Tensor`. Our profilers now force `Mxfp4Config(dequantize=True)` (profiling-only) so weights are `torch.nn.Parameter` and we don’t waste hours before failing in pass2.

---

## Canonical “Near-Lossless” Gate (what we trust)

Use `harmony/cuda-norm/pruning/near_lossless_gates.json`:
- Primary packs: `UNION`
- Primary seq lens: `1024`, `2048`
- Thresholds: `ΔPPL` (abs+rel), `ΔCC`, `Δmean_p`, `JS2D`

**Critical lesson learned**: JS2D is **evaluation-budget sensitive**. Small evals can show false FAIL (JS2D ~0.023) that disappear at high token budget.

**Canonical eval regime (“bigblocks”)**
- `num_blocks=512`
- `sample_points=200000`
- `batch_size=1`
- `seq_lens=1024,2048`
- `top_k=4`

**Canonical PASS reference**
- `harmony/cuda-norm/reports/eaftreap75_bigblocks_1024_2048.md`
- Noise floor: `harmony/cuda-norm/reports/eaft_base_vs_base_noise_20b_bigblocks.md`
- Rewrite lossless: `harmony/cuda-norm/reports/eaft_noop_rewrite_lossless_20b_bigblocks.md`

---

## What’s Done (milestones)

### A) Harness + infra (done)
- [x] **EAFT parity harness + strict gates** (`near_lossless_gates.json`)
- [x] **SGLang collector** for EAFT metrics (no Transformers eval path)
- [x] **Packed-token structural rewrite** emits ~`<=5GB` safetensors shards
- [x] **Uniform keep_n enforcement** (fix for: “All layers must keep the same number of experts.”)

### B) EAFT‑REAP algorithm (done)
- [x] **EAFT‑REAP profiling** (token-level weighting by EAFT regions)
- [x] **Core experts forced first**, then fill remaining slots by ranking
- [x] **Ranking signal corrected**: use **total EAFT-weighted saliency mass** (not mean-per-hit) — major quality fix

### C) keep_frac=0.75 result (done)
- [x] **0.75 built** (keep_n=24/32) with packed shards
- [x] **0.75 PASS** at 1024 and 2048 in bigblocks regime
- [x] **Base-vs-base noise floor PASS** (defines “measurement wobble” ≈ 0 under bigblocks)
- [x] **No-op rewrite lossless PASS** (proves rewrite/index/mapping path introduces no drift)
- [x] **Manifest compatibility**: keepfrac sweep writes both `manifest_eaftreap_keepfrac.json` and `manifest.json`

---

## “Noisy” / Historical Runs (do not use for final decisions)

These were earlier iterations where JS2D was noisy or scoring was still being fixed:
- `harmony/cuda-norm/reports/eaftreap_budgeted_keepfrac75_corefix_parity_summary.md` (broken scoring; clearly bad)
- `harmony/cuda-norm/reports/eaftreap_budgeted_keepfrac75_parity_summary.md` (borderline JS2D fail)
- `harmony/cuda-norm/reports/eaftreap_budgeted_keepfrac75_profile2k_v2_parity_summary.md` (borderline JS2D fail)
- `harmony/cuda-norm/reports/eaftreap_budgeted_keepfrac75_toolweighted_20260117_084613.md` (same as above)
- `harmony/cuda-norm/reports/eaftreap_budgeted_keepfrac75_toolweighted_20260117_092425.md` (same as above)
- `harmony/cuda-norm/reports/eaftreap_budgeted_keepfrac75_wconfm1_20260117_082447.md` (borderline JS2D fail)

**Interpretation**: these are useful for throughput/relative trends, but not definitive quality. Bigblocks is definitive.

---

## Next TODOs (10–30 tasks; quality-first)

### Phase 1 — Lock 0.75 as “official” (target: 75% done → 80% done)
1. [x] **Base-vs-base noise baseline** (bigblocks 1024/2048; expect near-zero deltas)
2. [x] **No-op rewrite check**: rewrite with keep_n=32 and verify EAFT parity is lossless
3. [ ] **Multi-seed stability** for 0.75 (2–3 seeds) to confirm PASS is robust
4. [ ] **Tool-agentic deep dive**: verify the “hardest pack” isn’t hiding regressions (CC and JS2D per-pack)
5. [ ] **Freeze a “0.75 lockfile”**: one manifest + report that pins all parameters (seed, weights, pack strategy, keep_n)

### Phase 2 — Get 0.60 near-lossless (this is the core milestone) (target: 58% → 90%)
6. [ ] **Run 0.60 bigblocks gate** (1024+2048) against the same base bigblocks JSON
7. [ ] **Write a definitive 0.60 parity report** (PASS/FAIL + per-pack breakdown)
8. [ ] If FAIL: **identify which pack/seq fails first** (usually tool_agentic or reasoning @2048)
9. [ ] If FAIL: **increase profiling budget** (e.g., 20k rows) and rebuild 0.60 (keep_n fixed to 20 for kernel compatibility)
10. [ ] If FAIL: **pack-weighted profiling** (oversample tool_agentic/reasoning) — keep top_k fixed
11. [ ] If FAIL: **EAFT weight sweep** (w_conflict, w_uncertain) with a tight grid (minimize GPU jobs, maximize signal)
12. [ ] If FAIL: **constrainted selection** (ensure each pack’s top-M experts per layer are represented inside keep_n)
13. [ ] **Re-run 0.60 bigblocks gate** after each *single* change (no confounded experiments)
14. [ ] **Stop condition**: once 0.60 PASS is stable at 1024+2048, freeze a 0.60 lockfile

### Phase 3 — Make it reproducible + shippable (target: 90% → 100%)
15. [ ] **Artifact integrity**: manifest.json + mapping + shard sizes (≤~5GB) + smoke-load in SGLang
16. [ ] **Determinism audit**: record dataset snapshot IDs + prompt hashes used for profiling/eval
17. [ ] **Update** `harmony/cuda-norm/reports/PRUNING_DECISION_MEMO.md` with final 0.75/0.60 conclusions
18. [ ] **Write “How-to”**: one command per milestone (predownload → profile → prune → eval → summarize)

### Phase 4 — 120B (only after 20B 0.60 PASS) (optional / later)
19. [ ] Run **120B EAFT baseline** (no pruning) on the same packs to establish model-size ordering
20. [ ] Run **120B partial-prune IO/RAM probe** (single layer rewrite) to cost-estimate full prune

---

## Repro Commands (Modal profile dependent)

**0.75 canonical PASS report**
- `harmony/cuda-norm/reports/eaftreap75_bigblocks_1024_2048.md`

**0.60 prune build (already done)**
- Manifest: `harmony/cuda-norm/artifacts/20b_pruned_models_eaftreap_keepfrac/manifest_eaftreap_keepfrac.json`
- Target variant (keep_n=20): `calib_union_keep20of32_k62_eaftreap` (keep_n%4 compatible)

**Pipeline helper**
- `harmony/cuda-norm/scripts/modal_pipeline_eaftreap_keepfrac.sh`

## REAP Recipe YAMLs (single source of truth)

- Folder: `harmony/cuda-norm/pruning/recipes/README.md`
- 20B prune recipe (0.75): `harmony/cuda-norm/pruning/recipes/reap-recipe_20b_eaftreap_keepfrac075_budgeted.yaml`
- 20B scores snapshot (0.75): `harmony/cuda-norm/pruning/recipes/reap-scores_20b_eaftreap_keepfrac075_budgeted.yaml`
- 20B prune recipe (0.75, keep24 uniform): `harmony/cuda-norm/pruning/recipes/reap-recipe_20b_eaftreap_keepfrac075_keep24_uniform.yaml`
- 20B scores snapshot (0.75, keep24 uniform): `harmony/cuda-norm/pruning/recipes/reap-scores_20b_eaftreap_keepfrac075_keep24_uniform.yaml`
- 20B eval recipe (0.75 bigblocks): `harmony/cuda-norm/pruning/recipes/eval-20b_eaftreap075_bigblocks.yaml`
- 20B eval recipe (0.75 bigblocks, keep24 uniform): `harmony/cuda-norm/pruning/recipes/eval-20b_eaftreap075_bigblocks_keep24_uniform.yaml`
- 20B eval recipe (0.75 long-seq tokenbudget ~1M, keep24 uniform): `harmony/cuda-norm/pruning/recipes/eval-20b_eaftreap075_longseq_tokenbudget1m_keep24_uniform.yaml`
- 20B eval recipe (0.75 long-seq tokenbudget ~10M, keep24 uniform): `harmony/cuda-norm/pruning/recipes/eval-20b_eaftreap075_longseq_tokenbudget10m_keep24_uniform.yaml`
