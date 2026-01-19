# PRUNING DECISION MEMO (GPT‑OSS MoE) — 20B pruning R&D, 120B cost probe

Modal profile: `locthaokien1201`  
Scope: pruning R&D lane (no distillation runs)

## Executive summary

1. **REAP-lite saliency ranking beats frequency ranking** at fixed keep_frac on 20B (better parity-PPL at the same structural prune level).
2. **EAFT‑REAP (correctness-aware) pruning is directionally better than “pure REAP” at the same keep_frac, but it is not near‑lossless at keep_frac=0.50** on curated union packs (fails strict gates).
3. **Decode throughput gains are primarily about routing compute (`top_k`) + batch headroom**. Structural pruning is mostly a *batch-headroom lever*; at fixed batch we see modest total tok/s changes.
4. **120B structural pruning looks feasible I/O-wise** (minutes-scale), but shard locality varies by layer; we should treat “full prune” as an I/O job and schedule it after we lock the architecture (TransMLA/DSA).

## 1) Does REAP ranking beat frequency ranking at same keep_frac?

Yes, on our parity harness for `radna0/nemotron-math-v2-harmony-tools` (`high_part00`).

Source: `harmony/cuda-norm/reports/20b_prune_quality_reap_vs_freq.md:1`

- keep_frac=0.50, top_k=4:
  - freq_50 ppl1024/2048 = 6.459 / 5.538
  - reap_50 ppl1024/2048 = 3.977 / 3.529
- keep_frac=0.25, top_k=4:
  - freq_25 ppl1024/2048 = 4.042 / 3.684
  - reap_25 ppl1024/2048 = 3.864 / 3.547

Interpretation: REAP-lite ranking is the correct default ranking method for structural expert pruning in our pipeline.

## 2) Does union pruning preserve multi-domain quality?

Not yet.

Source: `harmony/cuda-norm/reports/union_prune_quality.md:1`

- **Math** (good-ish):
  - base 2.921/2.575 → union50 3.088/2.704 (1024/2048)
- **Agentic** (noticeable hit):
  - base 5.525/4.828 → union50 5.973/5.125
- **General (chat_if)** (large hit):
  - base 7.208/5.017 → union50 9.674/6.498
- unionAgg is unacceptable across non-math domains.

Interpretation: current union set construction is too math-heavy (and/or under-profiled for general/chat_if). We should treat union50 as a throughput/memory prototype, not a final generalist prune.

## 3) Do we get decode throughput wins, and where do they come from?

Decode scoreboard is **long decode at batch**, not prefill.

Source: `harmony/cuda-norm/reports/union_decode_bench_2048_bs32.md:1`

At prompt_len=256, new_tokens=2048, batch up to 32:

- base_topk4 @bs32: total tok/s 588.16, mem_used 20.9 GiB
- base_topk2 @bs32: total tok/s 590.64, mem_used 29.8 GiB
- union50_topk2 @bs32: total tok/s 589.96, mem_used 20.6 GiB

Interpretation:

- At bs≤32, total tok/s is similar; **no large speedup yet**.
- The value of pruning here is **memory reduction / batch headroom**, not per-stream speed at fixed batch.
- `top_k` reduction is likely the real compute lever, but it must be validated in the “max stable batch” regime (push beyond 32 until OOM/paging).

Longer decode stress confirms the same story:

Source: `harmony/cuda-norm/reports/union_decode_bench_8192.md:1`

At prompt_len=256, new_tokens=8192, bs=32:

- base_topk4: 452.03 tok/s (mem_used 76.1 GiB)
- base_topk2: 472.03 tok/s (mem_used 70.0 GiB)
- union50_topk2: 460.45 tok/s (mem_used 75.8 GiB)

Interpretation:

- `top_k=2` is a small but consistent decode win vs `top_k=4` at long decode.
- Union pruning does **not** improve long-decode tok/s at fixed batch (decode is dominated by KV cache growth + attention compute; pruning mostly changes MoE MLP compute).

### Update (Kaggle / SGLang decode, base vs EAFT‑REAP‑50)

On a fresh Kaggle server (no Modal persistence), we re-ran decode-only benchmarks with **forced long decode**
(`min_new_tokens=max_new_tokens`, `ignore_eos=true`) and measured total tokens/sec.

Source: `harmony/cuda-norm/reports/eaftreap_decode_throughput_summary.md:1`

- `max_new_tokens=2048`, batch=32:
  - base: **6149.6 tok/s**
  - EAFT‑REAP‑50: **6529.8 tok/s** (**+6.2%**)
- `max_new_tokens=8192`, batch=32:
  - base: **5289.7 tok/s**
  - EAFT‑REAP‑50: **5472.2 tok/s** (**+3.5%**)

Interpretation:
- EAFT‑REAP‑50 gives a **small but consistent decode throughput improvement at higher batch**, consistent with reducing MoE compute per token.
- This is not a “massive speedup” lever by itself; the bigger lever remains **top_k reduction** + **max stable batch**.

## 3.5) Does EAFT‑REAP achieve near‑lossless quality at keep_frac=0.50?

No, not under the current strict gates on curated union packs.

Source: `harmony/cuda-norm/reports/eaftreap_quality_summary.md:1`

- UNION PPL @1024: base 5.237 → EAFT‑REAP‑50 7.290 (Δ +2.053)
- UNION PPL @2048: base 4.716 → EAFT‑REAP‑50 6.357 (Δ +1.641)

Interpretation:
- EAFT‑REAP is a better *ranking signal*, but **50% expert removal without recovery distill is not “near‑lossless”** on generalist packs.
- If “near‑lossless” is the requirement, we need either:
  - a much higher keep fraction (e.g. 0.8–0.9), or
  - a small recovery step (later).

## 4) Is 120B structural pruning feasible cost-wise right now?

Likely yes for the rewrite itself; it’s an I/O job measured in minutes, not hours, on Modal CPU workers with HF cache volumes.

Source: `harmony/cuda-norm/reports/120b_prune_cost_probe.md:1`

Key points:

- `openai/gpt-oss-120b` “main” checkpoint: **15 shards**, **60.77 GiB** total (`model-*.safetensors`).
- Partial prune probe (keep_frac=0.5):
  - layer0: 1 shard (3.88 GiB), total 33.34s, peak RSS 13.81 GiB
  - layer18: 2 shards (8.19 GiB), total 43.97s, peak RSS 14.73 GiB
- Shard locality varies per layer (1–2 shards per layer for MoE tensors in these probes).

Recommendation:

- Structural pruning on 120B is **feasible as a cost probe / tooling validation**.
- A “real” 120B prune should wait until the architecture (TransMLA/DSA) is stable, since any subsequent conversion could invalidate the pruned checkpoint format or require re-pruning.

## 5) Recommended next actions (20B first, then 120B)

1. **Finish decode benchmarking in the real regime**:
   - Extend batch sweep beyond 32 to max stable batch (base vs union50, top_k=4 vs 2).
2. **Fix union quality**:
   - Increase chat_if/general profiling budget and/or adjust union weighting/caps; re-run union50 with a higher expert cap for general.
3. **Then: 120B next step**:
   - Do a “one-layer structural rewrite” dry-run for a couple more layers (verify shard locality spread), then produce a bounded estimate for full rewrite cost.
