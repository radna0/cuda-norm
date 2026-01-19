# TPU: run DFlash spec‑v1 decode benchmarks (TPU‑first)

There are 2 decode harnesses:

- `cached_sequential_verify` (works everywhere; slower): per-token teacher verify with KV-cache.
- `cached_block_verify` (DFlash core; faster): verify the whole block with a single teacher forward, then crop/commit KV.

## 1) CPU unit tests (local)

```bash
harmony/cuda-norm/scripts/run_dflash_unit_tests_cpu.sh
```

## 2) TPU cached decode (sequential verify; correctness-first)

Set:
- `TEACHER_SNAPSHOT_DIR`: local HF snapshot dir for GPT‑OSS target (as used in TPU training)
- `DRAFT_RUN_DIR`: EasyDeL run dir containing `config.json` + tensorstore weights (e.g., `/dev/shm/dflash-checkpoints/<run>/run-<step>`)

Run with logging:

```bash
export TEACHER_SNAPSHOT_DIR=...
export DRAFT_RUN_DIR=...
export ALSO_RUN_BASELINE=1
export RUN_NAME=gptoss20b_dflash_decode_ctx512_b8_sanity

harmony/cuda-norm/scripts/run_tpu_dflash_decode_cached_logged.sh \
  --platform tpu \
  --max-prompt-len 512 \
  --max-new-tokens 256 \
  --block-size 8
```

Outputs (JSON):
- `accept_rate`
- `tok_s_total` (DFLASH)
- `baseline_tok_s_total` (target-only)
- `speedup_x` (DFLASH / baseline)

## 3) TPU cached decode (block verify; TARGET_VERIFY-style)

```bash
export TEACHER_SNAPSHOT_DIR=...
export DRAFT_RUN_DIR=...

harmony/cuda-norm/scripts/run_tpu_smoke_blockverify.sh \
  --teacher-snapshot-dir "$TEACHER_SNAPSHOT_DIR" \
  --draft-run-dir "$DRAFT_RUN_DIR"
```

Or use the generic logger wrapper with `DFLASH_DECODE_SCRIPT`:

```bash
export TEACHER_SNAPSHOT_DIR=...
export DRAFT_RUN_DIR=...
export ALSO_RUN_BASELINE=1
export DFLASH_DECODE_SCRIPT=tpu_dflash_spec_decode_blockverify_v1.py
harmony/cuda-norm/scripts/run_tpu_dflash_decode_cached_logged.sh --platform tpu --max-prompt-len 512 --max-new-tokens 256 --block-size 8
```

## 4) TPU cached benchmark sweep

```bash
export TEACHER_SNAPSHOT_DIR=...
export DRAFT_RUN_DIR=...
export ALSO_RUN_BASELINE=1
export DFLASH_DECODE_SCRIPT=tpu_dflash_spec_decode_blockverify_v1.py
export BLOCK_SIZES="8,16"
export MAX_NEW_TOKENS_LIST="256,1024"

harmony/cuda-norm/scripts/run_tpu_bench_cached_matrix.sh
```
