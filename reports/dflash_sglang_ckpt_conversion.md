# EasyDeL (TPU) → SGLang DFLASH checkpoint conversion

This repo’s TPU trainer saves draft checkpoints as zstd-compressed Zarr arrays under `run-*/model/...`. SGLang’s `DFlashDraftModel` expects a HF-style `config.json` + `model.safetensors.index.json` + safetensors shards.

## Convert a TPU run to an SGLang-loadable draft checkpoint

Example (GPT‑OSS‑20B run at step 2000):

```bash
python harmony/cuda-norm/scripts/convert_easydel_dflash_ckpt_to_sglang.py \
  --run-dir /dev/shm/dflash-checkpoints/gptoss20b_dflash_ctx1024_b8_k4_bs160_s2000/run-2000 \
  --dst harmony/cuda-norm/artifacts/dflash_sglang/gptoss20b_run2000 \
  --keep-fc-bias \
  --force
```

Notes:
- The converter is **CPU-only** and uses the system `zstd` binary (no `tensorstore`/`zarr` Python deps).
- `--keep-fc-bias` requires our patched SGLang overlay (we also enable MLP biases via `dflash_config.mlp_bias=true`).
- If you want the exported HF config to follow upstream SGLang PR #16818 layer-id semantics, pass:
  - `--target-layer-ids-mode afterlayer`
  This writes `dflash_config.target_layer_ids = (prelayer_ids - 1)`, so SGLang’s internal +1 pre-layer capture lands on the same activations.

## Smoke test (on a machine with `torch` + SGLang)

```bash
python harmony/cuda-norm/scripts/sglang_dflash_draft_load_smoke.py \
  --ckpt harmony/cuda-norm/artifacts/dflash_sglang/gptoss20b_run2000
```

Expected output includes:
- `has_fc_bias_param: true` (if using `--keep-fc-bias`)
- `has_any_mlp_bias_param: true`
- `block_size: 8`
- `num_context_features: 4`

## Packaging for transfer (optional)

```bash
tar -C harmony/cuda-norm/artifacts/dflash_sglang -czf \
  harmony/cuda-norm/artifacts/dflash_sglang/gptoss20b_run2000_sglang.tar.gz \
  gptoss20b_run2000
```
