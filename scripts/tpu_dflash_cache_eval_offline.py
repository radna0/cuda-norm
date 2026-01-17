#!/usr/bin/env python3
"""Offline (no teacher forward) evaluation of a DFlash draft on a teacher cache.

This is the fastest, most definitive diagnostic for draft quality:
- It uses the cached teacher labels (target_ids) directly (no verify forward).
- It runs the draft forward to propose (block_size-1) tokens.
- It computes:
  - token accuracy vs cached labels
  - accept_len distribution (prefix match length), which is the key driver of
    speculative speedups.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def _resolve_draft_run_dir(path: Path) -> Path:
    if (path / "config.json").is_file():
        return path
    runs = [p for p in path.glob("run-*") if (p / "config.json").is_file()]
    if not runs:
        raise FileNotFoundError(f"No run-* with config.json under {path}")

    def _step(p: Path) -> int:
        try:
            return int(p.name.split("-", 1)[1])
        except Exception:
            return -1

    runs.sort(key=_step, reverse=True)
    return runs[0]


def _load_lm_head_weight(snapshot_dir: Path):
    from safetensors import safe_open

    idx = snapshot_dir / "model.safetensors.index.json"
    if idx.exists():
        weight_map = json.loads(idx.read_text(encoding="utf-8"))["weight_map"]
        for name in ("lm_head.weight", "model.lm_head.weight"):
            shard = weight_map.get(name)
            if shard:
                with safe_open(str(snapshot_dir / shard), framework="flax") as f:
                    return f.get_tensor(name)
        raise KeyError("lm_head.weight not found in index.json")
    single = snapshot_dir / "model.safetensors"
    with safe_open(str(single), framework="flax") as f:
        for name in ("lm_head.weight", "model.lm_head.weight"):
            if name in f.keys():
                return f.get_tensor(name)
    raise KeyError("lm_head.weight not found in safetensors")


def _parse_csv_ints(s: str) -> list[int]:
    out: list[int] = []
    for part in (s or "").split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--teacher-snapshot-dir", required=True)
    ap.add_argument("--draft-run-dir", required=True, help="run-* dir (or a directory containing run-* subdirs)")
    ap.add_argument("--num-samples", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-idxs", default="", help="Optional explicit indices, comma-separated (overrides num-samples)")
    ap.add_argument("--batch-size", type=int, default=128, help="Global batch (will be split across TPU devices).")
    ap.add_argument("--vocab-chunk", type=int, default=8192, help="Argmax chunk size along vocab dimension.")
    ap.add_argument("--out-json", default="", help="Optional path to save raw per-sample stats.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    local_easydel = repo_root / "external" / "EasyDeL"
    if local_easydel.exists():
        sys.path.insert(0, str(local_easydel))
    os.environ.setdefault("EASYDEL_SKIP_VERSION_CHECK", "1")
    os.environ.setdefault("HF_HOME", "/dev/shm/hf")
    os.environ.setdefault("HF_HUB_CACHE", "/dev/shm/hf/hub")
    os.environ.setdefault("XDG_CACHE_HOME", "/dev/shm/xdg")
    os.environ.setdefault("TMPDIR", "/dev/shm/tmp")
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    import jax
    import jax.numpy as jnp

    from easydel.inference.speculative import DFlashDraftModelConfig, load_dflash_draft_from_run_dir
    from easydel.layers.rotary_embedding import get_rope

    cache_dir = Path(args.cache_dir).resolve()
    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    ctx_len = int(meta["ctx_len"])
    block_size = int(meta["block_size"])
    block = int(block_size - 1)
    hidden = int(meta["hidden_size"])
    k_hidden = int(meta["num_context_features"]) * hidden

    ctx_u16 = np.load(cache_dir / "context_features_u16.npy", mmap_mode="r")
    anchor_u16 = np.load(cache_dir / "anchor_embedding_u16.npy", mmap_mode="r")
    target_ids = np.load(cache_dir / "target_ids.npy", mmap_mode="r")
    pos_path = cache_dir / "ctx_pos_start_i32.npy"
    ctx_pos_start_i32 = np.load(pos_path, mmap_mode="r").astype(np.int32, copy=False) if pos_path.exists() else None

    n_total = int(ctx_u16.shape[0])
    if ctx_u16.shape != (n_total, ctx_len, k_hidden):
        raise ValueError(f"Unexpected context_features_u16 shape {ctx_u16.shape}, expected ({n_total},{ctx_len},{k_hidden})")
    if anchor_u16.shape != (n_total, hidden):
        raise ValueError(f"Unexpected anchor_embedding_u16 shape {anchor_u16.shape}, expected ({n_total},{hidden})")
    if target_ids.shape != (n_total, block):
        raise ValueError(f"Unexpected target_ids shape {target_ids.shape}, expected ({n_total},{block})")
    if ctx_pos_start_i32 is not None and int(ctx_pos_start_i32.shape[0]) != n_total:
        raise ValueError(f"Unexpected ctx_pos_start_i32 shape {ctx_pos_start_i32.shape}, expected ({n_total},)")

    explicit = _parse_csv_ints(args.sample_idxs)
    if explicit:
        idxs = np.asarray(explicit, dtype=np.int32)
    else:
        rng = np.random.default_rng(int(args.seed))
        k = int(min(max(1, int(args.num_samples)), n_total))
        idxs = rng.choice(n_total, size=k, replace=False).astype(np.int32)

    teacher_snapshot = Path(args.teacher_snapshot_dir).resolve()
    teacher_cfg = json.loads((teacher_snapshot / "config.json").read_text(encoding="utf-8"))
    rope = get_rope(
        head_size=int(teacher_cfg["head_dim"]),
        rotary_dim=int(teacher_cfg["head_dim"]),
        max_position=int(teacher_cfg["max_position_embeddings"]),
        base=int(teacher_cfg["rope_theta"]),
        is_neox_style=True,
        rope_scaling=teacher_cfg.get("rope_scaling"),
        dtype=jnp.bfloat16,
    )
    lm_w = jax.lax.stop_gradient(_load_lm_head_weight(teacher_snapshot))
    vocab_size = int(lm_w.shape[0])
    vocab_chunk = int(args.vocab_chunk)
    if vocab_chunk <= 0:
        vocab_chunk = 0
    elif vocab_size % vocab_chunk != 0:
        # Keep chunking deterministic; pick the largest divisor <= requested.
        for d in range(vocab_chunk, 0, -1):
            if vocab_size % d == 0:
                vocab_chunk = d
                break
        if vocab_chunk <= 0:
            vocab_chunk = 0

    # Load draft checkpoint.
    run_dir = _resolve_draft_run_dir(Path(args.draft_run_dir).resolve())
    draft_cfg = DFlashDraftModelConfig(**json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    # Draft checkpoints are saved with an EasyDeL 5D mesh convention.
    # Draft weights are replicated (partition_rules = [(".*", P())]), so any
    # mesh shape that covers all devices is fine.
    devices = jax.devices()
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape((1, len(devices), 1, 1, 1)),
        axis_names=("dp", "fsdp", "ep", "tp", "sp"),
    )
    draft = load_dflash_draft_from_run_dir(run_dir=run_dir, cfg=draft_cfg, mesh=mesh)

    def _argmax_chunked(logits_fn, *, hs: jax.Array) -> jax.Array:
        # hs: [B, S, H] -> ids [B, S]
        hs_f = hs.astype(jnp.bfloat16)
        bsz = int(hs_f.shape[0])
        seqlen = int(hs_f.shape[1])
        if vocab_chunk <= 0:
            logits = logits_fn(hs_f, lm_w)
            return jnp.argmax(logits, axis=-1).astype(jnp.int32)
        best_val = jnp.full((bsz, seqlen), -jnp.inf, dtype=jnp.float32)
        best_id = jnp.zeros((bsz, seqlen), dtype=jnp.int32)
        for start in range(0, vocab_size, vocab_chunk):
            end = start + vocab_chunk
            w = lm_w[start:end, :]
            chunk = logits_fn(hs_f, w).astype(jnp.float32)  # [B,S,chunk]
            local = jnp.argmax(chunk, axis=-1).astype(jnp.int32)
            local_val = jnp.take_along_axis(chunk, local[..., None], axis=-1)[..., 0]
            take = local_val > best_val
            best_val = jnp.where(take, local_val, best_val)
            best_id = jnp.where(take, local + jnp.int32(start), best_id)
        return best_id

    @jax.jit
    def forward_metrics(batch_ctx_u16, batch_anchor_u16, batch_labels, batch_pos_start):
        # Inputs:
        #  batch_ctx_u16: [B, ctx, K*H] uint16
        #  batch_anchor_u16: [B, H] uint16
        #  batch_labels: [B, block] int32
        ctx = jax.lax.bitcast_convert_type(batch_ctx_u16.astype(jnp.uint16), jnp.bfloat16)
        anc = jax.lax.bitcast_convert_type(batch_anchor_u16.astype(jnp.uint16), jnp.bfloat16)
        out = draft(
            context_features=ctx,
            anchor_embedding=anc,
            rope=rope,
            ctx_pos_start=batch_pos_start,
        )  # [B, block+1, H]
        hs = out[:, 1:, :]  # [B, block, H]

        def logits_fn(hs_in, w_in):
            return jnp.einsum("bsh,vh->bsv", hs_in, w_in, precision=jax.lax.Precision.HIGHEST)

        pred = _argmax_chunked(logits_fn, hs=hs)  # [B, block]
        matches = (pred == batch_labels).astype(jnp.int32)
        # accept_len = number of consecutive matches from t=0 onward.
        accept = jnp.sum(jnp.cumprod(matches, axis=-1), axis=-1).astype(jnp.int32)  # [B]
        tok_acc = jnp.mean(matches.astype(jnp.float32), axis=-1)  # [B]
        return accept, tok_acc

    # Batched loop.
    idxs = idxs.astype(np.int32)
    batch = max(1, int(args.batch_size))
    accept_all: list[int] = []
    acc_all: list[float] = []
    pos_all: list[int] = []

    for start in range(0, int(idxs.size), batch):
        end = min(int(idxs.size), start + batch)
        b_idx = idxs[start:end]
        ctx_b = np.asarray(ctx_u16[b_idx], dtype=np.uint16)
        anc_b = np.asarray(anchor_u16[b_idx], dtype=np.uint16)
        y_b = np.asarray(target_ids[b_idx], dtype=np.int32)
        if ctx_pos_start_i32 is not None:
            p_b = np.asarray(ctx_pos_start_i32[b_idx], dtype=np.int32)
        else:
            p_b = np.zeros((int(b_idx.shape[0]),), dtype=np.int32)
        with mesh:
            accept, tok_acc = forward_metrics(
                jnp.asarray(ctx_b),
                jnp.asarray(anc_b),
                jnp.asarray(y_b),
                jnp.asarray(p_b),
            )
        accept_all.extend([int(x) for x in np.asarray(accept)])
        acc_all.extend([float(x) for x in np.asarray(tok_acc)])
        pos_all.extend([int(x) for x in np.asarray(p_b)])

    accept_np = np.asarray(accept_all, dtype=np.int32)
    acc_np = np.asarray(acc_all, dtype=np.float32)
    pos_np = np.asarray(pos_all, dtype=np.int32)

    by_pos: dict[str, dict[str, float]] = {}
    for pos in sorted(set(int(x) for x in pos_np.tolist())):
        mask = pos_np == int(pos)
        if not np.any(mask):
            continue
        a = accept_np[mask]
        t = acc_np[mask]
        by_pos[str(int(pos))] = {
            "n": float(a.size),
            "accept_len_mean": float(np.mean(a)),
            "accept_len_p50": float(np.percentile(a, 50)),
            "accept_len_p90": float(np.percentile(a, 90)),
            "tok_acc_mean": float(np.mean(t)),
        }

    summary = {
        "cache_dir": str(cache_dir),
        "draft_run_dir": str(run_dir),
        "teacher_snapshot_dir": str(teacher_snapshot),
        "ctx_len": int(ctx_len),
        "block_size": int(block_size),
        "num_samples": int(accept_np.size),
        "accept_len_mean": float(np.mean(accept_np)),
        "accept_len_p50": float(np.percentile(accept_np, 50)),
        "accept_len_p90": float(np.percentile(accept_np, 90)),
        "accept_len_p99": float(np.percentile(accept_np, 99)),
        "accept_rate_tokens": float(np.sum(accept_np)) / float(max(1, accept_np.size * block)),
        "token_acc_mean": float(np.mean(acc_np)),
        "token_acc_p50": float(np.percentile(acc_np, 50)),
        "token_acc_p90": float(np.percentile(acc_np, 90)),
        "by_ctx_pos_start": by_pos,
    }
    print(json.dumps(summary, indent=2), flush=True)

    if args.out_json:
        out = Path(args.out_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "sample_idxs": [int(x) for x in idxs.tolist()],
                    "accept_len": [int(x) for x in accept_np.tolist()],
                    "token_acc": [float(x) for x in acc_np.tolist()],
                    "ctx_pos_start": [int(x) for x in pos_np.tolist()],
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
