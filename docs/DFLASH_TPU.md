# DFlash on TPU (EasyDeL/JAX) — Implementation Notes

This repo now contains a TPU-first DFlash draft training stack using:

- **Teacher/target model**: EasyDeL GPT‑OSS (`AutoEasyDeLModelForCausalLM`, `from_torch=True`)
- **Draft model**: Flax (Linen) module conditioned on **K** target hidden-state streams
- **Storage/caches**: `/dev/shm` (HF cache + JAX compilation cache + DFlash caches)

## Key answer: “Can we use the current conversion pipeline on TPU?”

Yes — we **do not need a separate conversion step** to start:

- Download / keep the HF snapshot in `/dev/shm/hf/hub/...`
- Load the PyTorch checkpoint directly via EasyDeL:
  - `AutoEasyDeLModelForCausalLM.from_pretrained(snapshot_dir, from_torch=True, ...)`

That’s the same mechanism used by the existing TPU scripts in `harmony/cuda-norm/scripts/`.

## What’s implemented (no training run started by this change)

### Core library
- `harmony/cuda-norm/scripts/tpu_dflash_lib.py`
  - Draft architecture (non-causal attention over `ctx + block`, GPT‑OSS-style GQA)
  - DFlash block semantics: `token0 = anchor`, `token1..B-1 = masked`
  - Uses EasyDeL RoPE (`easydel.layers.rotary_embedding.get_rope`) for compatibility

### Teacher cache builder (expensive forward; run once)
- `harmony/cuda-norm/scripts/tpu_dflash_build_teacher_cache.py`
  - Builds a cache `.npz` (default: `/dev/shm/out/...`) containing:
    - `context_features` (bf16): `[N, ctx_len, K*hidden]`
    - `anchor_embedding` (bf16): `[N, hidden]`
    - `target_ids` (int32): `[N, block_size-1]`
  - Loads GPT‑OSS via EasyDeL with `from_torch=True`
  - Configurable `--sharding-axis-dims` (default: `1,8,1,1,1`)

### Draft trainer (cheap loop; uses cache)
- `harmony/cuda-norm/scripts/tpu_dflash_train_draft_from_cache.py`
  - Trains only the draft parameters (teacher frozen)
  - Uses teacher **LM head** weights for CE loss on drafted tokens
  - Saves draft params as `draft_params.msgpack` periodically

### Correctness harness (slow but trustworthy)
- `harmony/cuda-norm/scripts/tpu_dflash_spec_decode_naive.py`
  - Naive DFlash spec‑v1 greedy verify:
    - Propose a block with the draft
    - Verify by recomputing the target greedy next token(s)
  - Not meant for throughput; meant for correctness + acceptance-rate tracking

## Running (once TPU is free)

Use the EasyDeL venv:

- `/dev/shm/easydel-venv/bin/python ...`

These scripts will fail to init TPU if another process is holding the TPU backend
(JAX TPU backend is exclusive per host process).

