#!/usr/bin/env python3
"""Train a DFlash draft model on TPU from a precomputed teacher cache.

This is the intended TPU workflow:
  1) Build cache once: `tpu_dflash_build_teacher_cache.py`
  2) Train draft many steps from cache (cheap):
     - no teacher prefill forward inside the training loop
     - only teacher LM head matmul for (block_size-1) tokens

IMPORTANT:
- This script is implemented but should only be run once the team confirms TPU
  is free (JAX TPU backend is exclusive per host process).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _load_bf16_from_npz(cache: np.lib.npyio.NpzFile, key_prefix: str):
    """Load a bf16 tensor saved in an .npz.

    We support both formats:
    - New: `{key_prefix}_u16` storing raw bf16 bitpatterns (uint16)
    - Old: `{key_prefix}` saved as ml_dtypes.bfloat16, which becomes dtype '|V2'
    """
    import jax
    import jax.numpy as jnp

    key_u16 = f"{key_prefix}_u16"
    if key_u16 in cache:
        u16 = np.asarray(cache[key_u16], dtype=np.uint16)
        return jax.lax.bitcast_convert_type(jnp.asarray(u16, dtype=jnp.uint16), jnp.bfloat16)

    if key_prefix not in cache:
        raise KeyError(f"Missing {key_prefix!r} (or {key_u16!r}) in cache.")

    arr = cache[key_prefix]
    # Old caches store bf16 as a 2-byte void dtype (|V2).
    if getattr(arr.dtype, "kind", None) == "V" and int(getattr(arr.dtype, "itemsize", 0)) == 2:
        u16 = arr.view(np.uint16)
        return jax.lax.bitcast_convert_type(jnp.asarray(u16, dtype=jnp.uint16), jnp.bfloat16)

    # Fallback: try a normal cast (works if numpy dtype is float16/float32).
    return jnp.asarray(arr, dtype=jnp.bfloat16)


def _load_lm_head_weight(*, snapshot_dir: Path):
    """Load GPT-OSS `lm_head.weight` directly from safetensors.

    Avoids instantiating the full EasyDeL model during draft training (we only
    need the frozen LM head for the CE loss).
    """
    import json
    import os

    from safetensors import safe_open

    idx = json.loads((snapshot_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map: dict[str, str] = idx["weight_map"]
    name = "lm_head.weight"
    shard = weight_map.get(name)
    if shard is None:
        raise KeyError(f"Missing {name!r} in model.safetensors.index.json")

    with safe_open(str(snapshot_dir / shard), framework="flax") as f:
        w = f.get_tensor(name)  # [vocab, hidden], bf16
    return w


def _chunked_nll_from_hidden(
    *,
    hidden: "Any",
    labels: "Any",
    lm_head_weight_vocab_hidden: "Any",
    vocab_chunk_size: int,
):
    """Compute mean NLL without materializing full [B,S,V] logits.

    hidden: [B, S, H]
    labels: [B, S]
    lm_head_weight_vocab_hidden: [V, H]
    """
    import jax
    import jax.numpy as jnp

    if int(vocab_chunk_size) <= 0:
        logits = jnp.einsum(
            "bsh,vh->bsv",
            hidden.astype(jnp.float32),
            lm_head_weight_vocab_hidden.astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
        )
        logp = jax.nn.log_softmax(logits, axis=-1)
        ll = jnp.take_along_axis(logp, labels[..., None].astype(jnp.int32), axis=-1)[..., 0]
        return -jnp.mean(ll)

    hs = hidden.astype(jnp.float32)
    y = labels.astype(jnp.int32)
    w = lm_head_weight_vocab_hidden.astype(jnp.float32)
    v = int(w.shape[0])
    h = int(w.shape[1])
    b = int(hs.shape[0])
    s = int(hs.shape[1])

    chunk = int(vocab_chunk_size)
    n_chunks = (v + chunk - 1) // chunk
    pad = n_chunks * chunk - v
    w_pad = jnp.pad(w, ((0, pad), (0, 0)))  # [n_chunks*chunk, H]
    w_chunks = w_pad.reshape((n_chunks, chunk, h))  # [C, chunk, H]

    def scan_body(carry, w_chunk):
        max_prev, sumexp_prev, true_logit_prev, start = carry
        # logits_chunk: [B,S,chunk]
        logits_c = jnp.einsum(
            "bsh,vh->bsv",
            hs,
            w_chunk,
            precision=jax.lax.Precision.HIGHEST,
        )
        max_c = jnp.max(logits_c, axis=-1)  # [B,S]
        max_new = jnp.maximum(max_prev, max_c)
        sumexp_new = sumexp_prev * jnp.exp(max_prev - max_new) + jnp.sum(
            jnp.exp(logits_c - max_new[..., None]),
            axis=-1,
        )

        in_chunk = (y >= start) & (y < (start + chunk))
        offset = jnp.clip(y - start, 0, chunk - 1)
        gathered = jnp.take_along_axis(logits_c, offset[..., None], axis=-1)[..., 0]
        true_logit_new = jnp.where(in_chunk, gathered, true_logit_prev)

        return (max_new, sumexp_new, true_logit_new, start + chunk), None

    init = (
        jnp.full((b, s), -jnp.inf, dtype=jnp.float32),
        jnp.zeros((b, s), dtype=jnp.float32),
        jnp.full((b, s), -jnp.inf, dtype=jnp.float32),
        jnp.array(0, dtype=jnp.int32),
    )
    (max_final, sumexp_final, true_logit_final, _), _ = jax.lax.scan(scan_body, init, w_chunks)
    logz = max_final + jnp.log(sumexp_final + 1e-9)
    nll = logz - true_logit_final
    return jnp.mean(nll)


def _spec_tree(tree, spec):
    import jax

    return jax.tree_util.tree_map(lambda _: spec, tree)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="Cache .npz from tpu_dflash_build_teacher_cache.py")
    ap.add_argument(
        "--teacher-snapshot-dir",
        required=True,
        help="HF snapshot dir for GPT-OSS target model (EasyDeL from_torch=True).",
    )
    ap.add_argument("--draft-layers", type=int, default=4)
    ap.add_argument("--mlp-ratio", type=float, default=4.0)
    ap.add_argument("--hidden-act", type=str, default="silu")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="In TPU mode this is per-device batch (global=batch_size*num_devices).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--out-dir", default=f"/dev/shm/out/dflash_draft_{_now_tag()}")
    ap.add_argument(
        "--vocab-chunk-size",
        type=int,
        default=16384,
        help="0 disables chunking (materializes full logits). Use chunking for large batches.",
    )
    ap.add_argument(
        "--spmd",
        action="store_true",
        help="Use a 2D dp×tp SPMD mesh (dp=data-parallel, tp=vocab-parallel LM head).",
    )
    ap.add_argument("--dp", type=int, default=0, help="SPMD data-parallel size (dp×tp must equal num_devices).")
    ap.add_argument("--tp", type=int, default=0, help="SPMD vocab-parallel size (dp×tp must equal num_devices).")
    ap.add_argument(
        "--sharding-axis-dims",
        default="1,8,1,1,1",
        help="5D sharding axis dims (dp,fsdp,ep,tp,sp). Default fits v6e-8 single host.",
    )
    ap.add_argument(
        "--platform",
        default="tpu",
        choices=["tpu", "cpu"],
        help="For debugging only. Use 'cpu' if TPU is busy.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Only print cache/model metadata; no JAX compile.")
    args = ap.parse_args()

    from tpu_dflash_lib import (
        DFlashDraftConfig,
        build_rope,
        load_json,
        make_dflash_draft_module,
        require_hf_token,
        set_shm_caches,
    )

    set_shm_caches()
    require_hf_token()
    if args.platform == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    cache_path = Path(args.cache).resolve()
    cache = np.load(str(cache_path), allow_pickle=True)
    meta = json.loads(str(cache["meta"].tolist()))

    teacher_snapshot = Path(args.teacher_snapshot_dir).resolve()
    cfg = load_json(teacher_snapshot / "config.json")

    ctx_len = int(meta["ctx_len"])
    block_size = int(meta["block_size"])
    target_layer_ids = list(meta["target_layer_ids"])
    k = int(len(target_layer_ids))
    hidden = int(cfg["hidden_size"])

    import jax
    import jax.numpy as jnp

    devices = jax.devices()
    num_devices = int(len(devices))
    if args.platform == "tpu" and num_devices < 2:
        raise RuntimeError("Requested TPU platform but only 1 device is visible.")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "cache": str(cache_path),
                    "cache_ctx_len": ctx_len,
                    "cache_block_size": block_size,
                    "cache_K": k,
                    "cache_target_layer_ids": target_layer_ids,
                    "teacher_snapshot_dir": str(teacher_snapshot),
                    "teacher_hidden": hidden,
                    "draft_layers": int(args.draft_layers),
                    "platform": str(args.platform),
                    "num_devices": num_devices,
                    "per_device_batch": int(args.batch_size),
                    "global_batch": int(args.batch_size) * num_devices if args.platform == "tpu" else int(args.batch_size),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return

    import optax

    rope = build_rope(cfg=cfg, dtype=jnp.bfloat16)

    dcfg = DFlashDraftConfig(
        hidden_size=int(hidden),
        num_layers=int(args.draft_layers),
        mlp_ratio=float(args.mlp_ratio),
        hidden_act=str(args.hidden_act),
        num_attention_heads=int(cfg["num_attention_heads"]),
        num_key_value_heads=int(cfg["num_key_value_heads"]),
        head_dim=int(cfg["head_dim"]),
        max_position_embeddings=int(cfg["max_position_embeddings"]),
        rope_theta=float(cfg["rope_theta"]),
        rope_scaling=cfg.get("rope_scaling"),
        rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-5)),
        block_size=int(block_size),
        num_context_features=int(k),
    )
    Draft = make_dflash_draft_module()
    draft = Draft(cfg=dcfg)

    rng = jax.random.PRNGKey(int(args.seed))
    dummy_ctx = jnp.zeros((1, ctx_len, k * hidden), dtype=jnp.bfloat16)
    dummy_anchor = jnp.zeros((1, hidden), dtype=jnp.bfloat16)
    params = draft.init(rng, context_features=dummy_ctx, anchor_embedding=dummy_anchor, rope=rope)

    opt = optax.adamw(learning_rate=float(args.lr), b1=0.9, b2=0.95, weight_decay=0.01)
    opt_state = opt.init(params)

    # Load frozen LM head weight [V,H] directly (no EasyDeL model instantiation).
    lm_head_weight = _load_lm_head_weight(snapshot_dir=teacher_snapshot)
    lm_head_weight = jax.lax.stop_gradient(lm_head_weight)

    # Cached data (bf16) host->device.
    context_features = _load_bf16_from_npz(cache, "context_features")  # [N, ctx, K*hidden]
    anchor_embedding = _load_bf16_from_npz(cache, "anchor_embedding")  # [N, hidden]
    target_ids = jnp.asarray(cache["target_ids"], dtype=jnp.int32)  # [N, block-1]

    n = int(context_features.shape[0])

    def loss_from_batch(p, cf, ae, labels):
        out = draft.apply(p, context_features=cf, anchor_embedding=ae, rope=rope)  # [B, block, hidden]
        hs = out[:, 1:, :]  # predict positions 1..block-1
        return _chunked_nll_from_hidden(
            hidden=hs,
            labels=labels,
            lm_head_weight_vocab_hidden=lm_head_weight,
            vocab_chunk_size=int(args.vocab_chunk_size),
        )

    if args.platform == "tpu" and bool(args.spmd):
        # ----
        # True SPMD: dp×tp mesh.
        #
        # Motivation:
        # - The exact CE loss requires a vocab softmax over V≈200k, which is the
        #   dominant compute. Sharding vocab across tp greatly accelerates.
        # - dp shards the cache/batch for higher global batch size.
        #
        # We shard:
        # - cache tensors along dp
        # - lm_head.weight along tp (vocab dimension)
        # - draft params are replicated
        #
        from jax import shard_map
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

        dp = int(args.dp) if int(args.dp) > 0 else 1
        tp = int(args.tp) if int(args.tp) > 0 else int(num_devices) // int(dp)
        if int(dp) <= 0 or int(tp) <= 0 or int(dp) * int(tp) != int(num_devices):
            raise ValueError(f"Invalid dp/tp: dp={dp} tp={tp} num_devices={num_devices} (need dp*tp==num_devices)")

        dev_mesh = np.array(devices).reshape((dp, tp))
        mesh = Mesh(dev_mesh, axis_names=("dp", "tp"))

        # Pad examples to a multiple of dp.
        shard_n = (n + dp - 1) // dp
        pad_n = shard_n * dp - n

        def _pad_first_dim(x, pad_value=0):
            if pad_n <= 0:
                return x
            pad_width = [(0, pad_n)] + [(0, 0) for _ in range(x.ndim - 1)]
            return jnp.pad(x, pad_width, constant_values=pad_value)

        context_padded = _pad_first_dim(context_features)
        anchor_padded = _pad_first_dim(anchor_embedding)
        target_padded = _pad_first_dim(target_ids)

        # Pad vocab to a multiple of tp.
        v = int(lm_head_weight.shape[0])
        h = int(lm_head_weight.shape[1])
        v_per = (v + tp - 1) // tp
        v_pad = v_per * tp - v
        w_padded = jnp.pad(lm_head_weight, ((0, v_pad), (0, 0)))  # [V_pad, H]

        # Place arrays with NamedSharding.
        with mesh:
            sh_cache = NamedSharding(mesh, P("dp", None, None))
            sh_cache_vec = NamedSharding(mesh, P("dp", None))
            sh_w = NamedSharding(mesh, P("tp", None))
            sh_batch = NamedSharding(mesh, P("dp", None))
            sh_repl = NamedSharding(mesh, P())

            context_padded = jax.device_put(context_padded, sh_cache)
            anchor_padded = jax.device_put(anchor_padded, sh_cache_vec)
            target_padded = jax.device_put(target_padded, sh_cache_vec)
            w_padded = jax.device_put(w_padded, sh_w)

            opt_spec = _spec_tree(opt_state, P())
            params = jax.device_put(params, sh_repl)
            opt_state = jax.device_put(opt_state, sh_repl)

        def _nll_vocab_parallel(*, hs, labels, w_local):
            # hs: [B, S, H] (B is local per dp shard, replicated across tp)
            # labels: [B, S] (replicated across tp)
            # w_local: [V_local, H] (sharded across tp)
            logits_local = jnp.einsum(
                "bsh,vh->bsv",
                hs.astype(jnp.float32),
                w_local.astype(jnp.float32),
                precision=jax.lax.Precision.HIGHEST,
            )
            max_local = jnp.max(logits_local, axis=-1)  # [B,S]
            max_global = jax.lax.pmax(max_local, "tp")
            sumexp_local = jnp.sum(jnp.exp(logits_local - max_global[..., None]), axis=-1)  # [B,S]
            sumexp_global = jax.lax.psum(sumexp_local, "tp")
            logz = max_global + jnp.log(sumexp_global + 1e-9)

            tp_i = jax.lax.axis_index("tp")
            start = tp_i * v_per
            in_range = (labels >= start) & (labels < (start + v_per))
            off = jnp.clip(labels - start, 0, v_per - 1).astype(jnp.int32)
            gathered = jnp.take_along_axis(logits_local, off[..., None], axis=-1)[..., 0]
            true_local = jnp.where(in_range, gathered, 0.0)
            true = jax.lax.psum(true_local, "tp")
            nll = logz - true
            return jnp.mean(nll)

        def _step(p, s, batch_idx, cf_all, ae_all, y_all, w_local):
            # batch_idx: [B_local] indices into this dp shard
            batch_idx = jnp.asarray(batch_idx)
            # With shard_map, if we pass global batch_idx shaped [dp, B], each dp
            # shard sees a [1, B] view. Squeeze it so advanced indexing doesn't
            # introduce extra singleton dimensions.
            if batch_idx.ndim == 2 and batch_idx.shape[0] == 1:
                batch_idx = batch_idx[0]
            batch_idx = batch_idx.reshape((-1,))

            cf = cf_all[batch_idx]  # [B_local, ctx, K*H]
            ae = ae_all[batch_idx]  # [B_local, H]
            labels = y_all[batch_idx]  # [B_local, block-1]

            out = draft.apply(p, context_features=cf, anchor_embedding=ae, rope=rope)
            hs = out[:, 1:, :]

            def _loss(pp):
                return _nll_vocab_parallel(hs=hs, labels=labels, w_local=w_local)

            loss, grads = jax.value_and_grad(_loss)(p)
            grads = jax.lax.pmean(grads, "dp")
            grads = jax.lax.pmean(grads, "tp")
            loss = jax.lax.pmean(loss, "dp")
            loss = jax.lax.pmean(loss, "tp")
            updates, s2 = opt.update(grads, s, p)
            p2 = optax.apply_updates(p, updates)
            return p2, s2, loss

        train_step = jax.jit(
            shard_map(
                _step,
                mesh=mesh,
                in_specs=(
                    P(),  # params replicated
                    opt_spec,  # opt_state replicated
                    P("dp", None),  # batch_idx sharded on dp
                    P("dp", None, None),  # context sharded on dp
                    P("dp", None),  # anchor sharded on dp
                    P("dp", None),  # labels sharded on dp
                    P("tp", None),  # lm_head.weight sharded on tp
                ),
                out_specs=(P(), opt_spec, P()),
            ),
        )

        rng_np = np.random.default_rng(int(args.seed))

        def sample_batch_idx():
            # [dp, batch] so it shards cleanly over dp and replicates over tp.
            # Each dp shard samples within its local cache range.
            idx = rng_np.integers(0, shard_n, size=(dp, int(args.batch_size)), endpoint=False, dtype=np.int32)
            return jnp.asarray(idx, dtype=jnp.int32)

        def to_host(x):
            return float(np.asarray(jax.device_get(x)))

        global_batch = int(dp) * int(args.batch_size)

    elif args.platform == "tpu":
        # Data-parallel pmap across all local TPU devices.
        # Shard the cache across devices (total cache is distributed, not replicated).
        shard_n = (n + num_devices - 1) // num_devices
        pad_n = shard_n * num_devices - n

        def _pad_first_dim(x, pad_value=0):
            if pad_n <= 0:
                return x
            pad_width = [(0, pad_n)] + [(0, 0) for _ in range(x.ndim - 1)]
            return jnp.pad(x, pad_width, constant_values=pad_value)

        context_padded = _pad_first_dim(context_features)
        anchor_padded = _pad_first_dim(anchor_embedding)
        target_padded = _pad_first_dim(target_ids)

        context_sharded = context_padded.reshape((num_devices, shard_n) + context_padded.shape[1:])
        anchor_sharded = anchor_padded.reshape((num_devices, shard_n) + anchor_padded.shape[1:])
        target_sharded = target_padded.reshape((num_devices, shard_n) + target_padded.shape[1:])

        # Replicate params/opt state + LM head to all devices.
        params = jax.device_put_replicated(params, devices)
        opt_state = jax.device_put_replicated(opt_state, devices)
        lm_head_weight = jax.device_put_replicated(lm_head_weight, devices)

        # Move sharded cache to devices.
        context_sharded = jax.device_put_sharded(list(context_sharded), devices)
        anchor_sharded = jax.device_put_sharded(list(anchor_sharded), devices)
        target_sharded = jax.device_put_sharded(list(target_sharded), devices)

        def _pmap_step(p, s, batch_idx, cf_shard, ae_shard, y_shard):
            # batch_idx: [local_B] indices into the local shard.
            cf = cf_shard[batch_idx]
            ae = ae_shard[batch_idx]
            labels = y_shard[batch_idx]

            def _loss(pp):
                return loss_from_batch(pp, cf, ae, labels)

            loss, grads = jax.value_and_grad(_loss)(p)
            grads = jax.lax.pmean(grads, axis_name="d")
            loss = jax.lax.pmean(loss, axis_name="d")
            updates, s2 = opt.update(grads, s, p)
            p2 = optax.apply_updates(p, updates)
            return p2, s2, loss

        pmap_step = jax.pmap(_pmap_step, axis_name="d", in_axes=(0, 0, 0, 0, 0, 0))
        train_step = lambda p, s, batch_idx: pmap_step(p, s, batch_idx, context_sharded, anchor_sharded, target_sharded)

        rng_np = np.random.default_rng(int(args.seed))

        def sample_batch_idx():
            # [devices, local_batch]
            return jnp.asarray(
                rng_np.integers(0, shard_n, size=(num_devices, int(args.batch_size)), endpoint=False, dtype=np.int32),
                dtype=jnp.int32,
            )

        def to_host(x):
            return float(np.asarray(jax.device_get(x[0])))

    else:
        # Single-device jit (CPU or 1-device TPU).
        @jax.jit
        def train_step(p, s, batch_idx):
            cf = context_features[batch_idx]
            ae = anchor_embedding[batch_idx]
            labels = target_ids[batch_idx]
            loss, grads = jax.value_and_grad(lambda pp: loss_from_batch(pp, cf, ae, labels))(p)
            updates, s2 = opt.update(grads, s, p)
            p2 = optax.apply_updates(p, updates)
            return p2, s2, loss

        rng_np = np.random.default_rng(int(args.seed))

        def sample_batch_idx():
            return jnp.asarray(
                rng_np.integers(0, n, size=(int(args.batch_size),), endpoint=False, dtype=np.int32),
                dtype=jnp.int32,
            )

        def to_host(x):
            return float(np.asarray(jax.device_get(x)))

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "cache": str(cache_path),
                "teacher_snapshot_dir": str(teacher_snapshot),
                "draft_layers": int(args.draft_layers),
                "mlp_ratio": float(args.mlp_ratio),
                "hidden_act": str(args.hidden_act),
                "lr": float(args.lr),
                "steps": int(args.steps),
                "batch_size": int(args.batch_size),
                "global_batch_size": (
                    int(global_batch)
                    if args.platform == "tpu" and bool(args.spmd)
                    else int(args.batch_size) * num_devices
                    if args.platform == "tpu"
                    else int(args.batch_size)
                ),
                "seed": int(args.seed),
                "vocab_chunk_size": int(args.vocab_chunk_size),
                "platform": str(args.platform),
                "num_devices": num_devices,
                "spmd": bool(args.spmd),
                "dp": int(args.dp),
                "tp": int(args.tp),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    t0 = time.time()
    losses = []
    for step in range(1, int(args.steps) + 1):
        batch_idx = sample_batch_idx()
        if args.platform == "tpu" and bool(args.spmd):
            params, opt_state, loss = train_step(
                params, opt_state, batch_idx, context_padded, anchor_padded, target_padded, w_padded
            )
        else:
            params, opt_state, loss = train_step(params, opt_state, batch_idx)
        lf = to_host(loss)
        losses.append(lf)
        if step == 1 or step % int(args.log_every) == 0:
            wall = time.time() - t0
            print(f"[step {step}] loss={lf:.6f} wall_s={wall:.1f}", flush=True)
        if step % int(args.save_every) == 0 or step == int(args.steps):
            # Save flax params as msgpack for later conversion/inference wiring.
            import flax

            step_dir = out_dir / f"step_{step:06d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            save_params = params
            if args.platform == "tpu":
                # params is replicated across devices: take replica 0.
                if bool(args.spmd):
                    # params are replicated across the dp×tp mesh already.
                    save_params = jax.device_get(save_params)
                else:
                    save_params = jax.tree_util.tree_map(lambda x: x[0], params)
            (step_dir / "draft_params.msgpack").write_bytes(flax.serialization.to_bytes(save_params))
            np.save(step_dir / "losses.npy", np.asarray(losses, dtype=np.float32))
            print(f"[+] saved {step_dir}", flush=True)

    print(f"[done] out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
