# EAFT‑REAP keep_frac=0.75 (no finetune) — Playbook (Kaggle/VERSA)

Goal: make `keep_frac=0.75` (24/32 experts) **near‑lossless** on curated calib packs, with **top_k unchanged (4)**.

This playbook assumes:
- You run on Kaggle via Versa (remote Jupyter).
- Base 20B is mounted at `/kaggle/input/gpt-oss-20b/transformers/default/1`.

## 0) Expert math sanity (cheap, CPU)

```bash
bash harmony/cuda-norm/scripts/versa_run_pruning_track_kaggle.sh \
  --task validate_expert_math_toy
```

Output: `reports/validate_gpt_oss_expert_math_toy.md` (on the remote).

## 1) Build pruned checkpoint (budgeted + safety core)

Use **full 30k packs** for profiling (no finetune). This produces a single pruned checkpoint.

```bash
bash harmony/cuda-norm/scripts/versa_run_pruning_track_kaggle.sh \
  --task build_pruned_20b_eaftreap_budgeted \
  --calib-packs-repo radna0/harmony-qwen3-calib-packs-v2-20260113 \
  --calib-pack-files "packs/reasoning_style_10k_v2/reasoning_style_10k_v2.parquet,tool_agentic_10k_v6.parquet,packs/calib_prompt_10000_v2/calib_prompt_10000_v2.parquet" \
  --calib-pack-sample-strategy per_file \
  --num-rows 30000 \
  --max-seq-length 4096 --batch-size 1 \
  --keep-frac 0.75 \
  --min-keep-per-layer 16 --max-keep-per-layer 32 \
  --core-pos-top-m 4 --core-count-top-m 0
```

Remote artifacts:
- `artifacts/20b_pruned_models_eaftreap_budgeted/manifest.json` (contains `out_dir`)
- `artifacts/20b_pruned_models_eaftreap_budgeted/keep_counts_by_layer.json`
- `artifacts/20b_pruned_models_eaftreap_budgeted/core_experts_by_layer.json`

## 2) EAFT eval parity (base vs pruned) on packs (seq=1024,2048)

1) Get the pruned model path:

```bash
python - <<'PY'
import json
print(json.load(open("artifacts/20b_pruned_models_eaftreap_budgeted/manifest.json"))["out_dir"])
PY
```

2) Run EAFT single-model collector for base + pruned:

```bash
# Base (uses mounted Kaggle input)
bash harmony/cuda-norm/scripts/versa_run_eaft_single_kaggle.sh \
  --no-detach \
  --model-id openai/gpt-oss-20b \
  --model-path /kaggle/input/gpt-oss-20b/transformers/default/1 \
  --seq-lens-csv 1024,2048 \
  --num-blocks 256 --batch-size 1 --sample-points 200000 \
  --top-k 4

# Pruned (use out_dir printed above)
bash harmony/cuda-norm/scripts/versa_run_eaft_single_kaggle.sh \
  --no-detach \
  --model-id eaftreap_budgeted_keepfrac075 \
  --model-path "<PASTE_out_dir_HERE>" \
  --seq-lens-csv 1024,2048 \
  --num-blocks 256 --batch-size 1 --sample-points 200000 \
  --top-k 4
```

## 3) Summarize gates locally (host CPU)

After each EAFT run finishes, the local log prints a line like:
`[+] Wrote artifacts/eaft_models/<run_id>/<model>.json`

Run:

```bash
python harmony/cuda-norm/scripts/summarize_eaft_pair.py \
  --left-json  artifacts/eaft_models/<...>/openai__gpt-oss-20b.json \
  --right-json artifacts/eaft_models/<...>/eaftreap_budgeted_keepfrac075.json \
  --out-md harmony/cuda-norm/reports/eaftreap_budgeted_keepfrac75_parity_summary.md
```

Gate thresholds live in `harmony/cuda-norm/pruning/near_lossless_gates.json`.

## 4) Decode-only sanity (optional; top_k unchanged)

This is a throughput sanity check only (quality gates decide success).

```bash
# Base
bash harmony/cuda-norm/scripts/versa_run_decode_bench_sglang_kaggle.sh \
  --name base20b_k4 \
  --model-path /kaggle/input/gpt-oss-20b/transformers/default/1 \
  --prompt-len 256 --max-new-tokens 8192 --batch-sizes 1,2,4,8,16,32

# Pruned
bash harmony/cuda-norm/scripts/versa_run_decode_bench_sglang_kaggle.sh \
  --name pruned20b_k4_budgeted075 \
  --model-path \"<PASTE_out_dir_HERE>\" \
  --prompt-len 256 --max-new-tokens 8192 --batch-sizes 1,2,4,8,16,32
```

## 5) Gate decision (what “near‑lossless” means here)

We treat this as **PASS** only if the UNION hero rows (seq=1024 and 2048) pass `near-lossless-v1`:
- `|ΔPPL| <= min(0.25, 0.05 * ppl_base)`
- `|ΔCC_rate| <= 0.002`
- `|Δmean_prob| <= 0.02`
- `JS2D <= 0.02`

If it **FAILS** at keep_frac=0.75 (top_k=4 unchanged), next deterministic levers (still no finetune) are:
- Increase profiling rows (already maxed at 30k packs → next is add more packs / domains).
- Increase per-layer minimum keep for the most sensitive layers (`--min-keep-per-layer`) and reduce elsewhere (budget keeps total constant).
- Increase safety core size (`--core-pos-top-m`) conservatively (e.g. 6–8) and re-run.
