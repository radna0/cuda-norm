from __future__ import annotations

import json
import os
import typing as tp
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Thread

# Must be set before importing JAX for persistent compilation cache to work.
os.environ.setdefault("HF_HOME", "/dev/shm/hf")
os.environ.setdefault("HF_HUB_CACHE", "/dev/shm/hf/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/dev/shm/hf/transformers")
os.environ.setdefault("XDG_CACHE_HOME", "/dev/shm/xdg")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/dev/shm/jax_compilation_cache_dflash")
os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")
os.environ.setdefault("TMPDIR", "/dev/shm/tmp")

import jax
import numpy as np
from eformer.loggings import get_logger
from flax import nnx
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossMetrics
from easydel.trainers.trainer import Trainer
from easydel.trainers.trainer_protocol import (
    TrainerConfigureDataloaderOutput,
    TrainerConfigureFunctionOutput,
    TrainerConfigureModelOutput,
)
from easydel.utils import Registry

from .easydel_dflash_cache import DFlashTeacherCacheDataset
from .easydel_dflash_config import DFlashConfig
from .easydel_dflash_draft_model import DFlashDraftModel, DFlashDraftModelConfig

logger = get_logger(__name__)


def _set_shm_caches() -> None:
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["JAX_COMPILATION_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    xla_flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_tpu_enable_latency_hiding_scheduler" not in xla_flags:
        os.environ["XLA_FLAGS"] = (xla_flags + " --xla_tpu_enable_latency_hiding_scheduler=true").strip()


def _require_token_present() -> None:
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
        raise RuntimeError("Missing HF token in env (HF_TOKEN or HUGGINGFACE_HUB_TOKEN).")


def _load_lm_head_weight(snapshot_dir: Path) -> jax.Array:
    """Load `lm_head.weight` as a JAX array [V,H] from safetensors."""
    from safetensors import safe_open

    name_candidates = ("lm_head.weight", "model.lm_head.weight")

    index_path = snapshot_dir / "model.safetensors.index.json"
    if index_path.exists():
        idx = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = idx.get("weight_map", {})
        for name in name_candidates:
            shard = weight_map.get(name)
            if shard is None:
                continue
            with safe_open(str(snapshot_dir / shard), framework="flax") as f:
                return f.get_tensor(name)
        raise KeyError(f"Missing {name_candidates} in {index_path.name}")

    single_path = snapshot_dir / "model.safetensors"
    if not single_path.exists():
        raise FileNotFoundError(f"Missing {index_path.name} and {single_path.name} in {snapshot_dir}")
    with safe_open(str(single_path), framework="flax") as f:
        for name in name_candidates:
            if name in f.keys():
                return f.get_tensor(name)
    raise KeyError(f"Missing {name_candidates} in {single_path.name}")


def _build_rope(*, cfg: dict, dtype):
    from easydel.layers.rotary_embedding import get_rope

    return get_rope(
        head_size=int(cfg["head_dim"]),
        rotary_dim=int(cfg["head_dim"]),
        max_position=int(cfg["max_position_embeddings"]),
        base=int(cfg["rope_theta"]),
        is_neox_style=True,
        rope_scaling=cfg.get("rope_scaling"),
        dtype=dtype,
    )


def _bf16_from_u16(x_u16: jax.Array) -> jax.Array:
    return jax.lax.bitcast_convert_type(x_u16.astype(jnp.uint16), jnp.bfloat16)


@dataclass(frozen=True)
class _DFlashBatch:
    context_u16: jax.Array  # [B, ctx, K*H] uint16
    anchor_u16: jax.Array  # [B, H] uint16
    target_ids: jax.Array  # [B, block-1] int32


def _batch_from_dict(batch: dict) -> _DFlashBatch:
    return _DFlashBatch(
        context_u16=jnp.asarray(batch["context_features_u16"], dtype=jnp.uint16),
        anchor_u16=jnp.asarray(batch["anchor_embedding_u16"], dtype=jnp.uint16),
        target_ids=jnp.asarray(batch["target_ids"], dtype=jnp.int32),
    )


def _choose_vocab_chunk(*, vocab_size: int, requested: int) -> int:
    if requested <= 0:
        return 0
    if vocab_size % requested == 0:
        return requested
    for d in range(requested, 0, -1):
        if vocab_size % d == 0:
            return d
    return 0


@Registry.register("trainer", "dflash")
class DFlashTrainer(Trainer):
    """Cache-first DFlash draft training.

    - Training never runs the teacher forward; it uses cached teacher features.
    - dp-only pmap path is the stable default on single-host TPU (8 devices).
    """

    arguments: DFlashConfig

    def __init__(
        self,
        arguments: DFlashConfig,
        *,
        processing_class,
        train_dataset: tp.Any | None = None,
        eval_dataset: tp.Any | None = None,
    ):
        if not isinstance(arguments, DFlashConfig):
            raise TypeError("arguments must be a DFlashConfig")

        _set_shm_caches()
        _require_token_present()

        if not arguments.cache_dir:
            raise ValueError("DFlashConfig.cache_dir is required")
        if not arguments.teacher_snapshot_dir:
            raise ValueError("DFlashConfig.teacher_snapshot_dir is required")

        self.arguments = arguments
        self.cache = DFlashTeacherCacheDataset(arguments.cache_dir)
        self.teacher_snapshot = Path(arguments.teacher_snapshot_dir).resolve()

        meta = self.cache.meta
        if meta.dtype not in ("bf16_u16",):
            raise ValueError(f"Unsupported cache dtype {meta.dtype!r}; expected bf16_u16")

        teacher_cfg = json.loads((self.teacher_snapshot / "config.json").read_text(encoding="utf-8"))
        self._rope = _build_rope(cfg=teacher_cfg, dtype=jnp.bfloat16)
        self._lm_head_weight = jax.lax.stop_gradient(_load_lm_head_weight(self.teacher_snapshot))

        dcfg = DFlashDraftModelConfig(
            hidden_size=int(meta.hidden_size),
            num_layers=int(arguments.draft_layers),
            mlp_ratio=float(arguments.mlp_ratio),
            hidden_act=str(arguments.hidden_act),
            num_attention_heads=int(teacher_cfg["num_attention_heads"]),
            num_key_value_heads=int(teacher_cfg["num_key_value_heads"]),
            head_dim=int(teacher_cfg["head_dim"]),
            rms_norm_eps=float(teacher_cfg.get("rms_norm_eps", 1e-5)),
            block_size=int(meta.block_size),
            num_context_features=int(meta.num_context_features),
            qk_norm=bool(arguments.qk_norm),
            remat=bool(getattr(arguments, "remat", True)),
        )

        rngs = nnx.Rngs(0)
        draft_model = DFlashDraftModel(dcfg, rngs=rngs)
        draft_model.mesh = self._make_mesh()

        tx, _scheduler = arguments.get_optimizer_and_scheduler(int(arguments.max_training_steps or 1))
        state = EasyDeLState.create(model=draft_model, tx=tx, init_opt_state=True)

        if train_dataset is None:
            from datasets import Dataset

            train_dataset = Dataset.from_dict({"idx": np.arange(len(self.cache), dtype=np.int64)})

        super().__init__(
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            model_state=state,
            processing_class=processing_class,
            data_collator=_dflash_or_idx_collate,
        )

    def configure_model(self):
        tx, scheduler = self.arguments.get_optimizer_and_scheduler(self.max_training_steps)
        return TrainerConfigureModelOutput(
            model=self.model,
            tx=tx,
            scheduler=scheduler,
            config=getattr(self.model, "config", None),
        )

    def _configure_state(self):
        mesh = self._make_mesh()
        empty = NamedSharding(mesh, P())

        if getattr(self.model_state, "opt_state", None) is None:
            with mesh:
                self.model_state = self.model_state.replace(opt_state=self.tx.init(self.model_state.graphstate))

        graphstate_sh = jax.tree_util.tree_map(lambda _: empty, self.model_state.graphstate)
        graphother_sh = jax.tree_util.tree_map(lambda _: empty, self.model_state.graphother)
        opt_sh = jax.tree_util.tree_map(lambda _: empty, self.model_state.opt_state)
        step_sh = empty

        with mesh:
            graphstate = jax.device_put(self.model_state.graphstate, graphstate_sh)
            graphother = jax.device_put(self.model_state.graphother, graphother_sh)
            opt_state = jax.device_put(self.model_state.opt_state, opt_sh)
            step = jax.device_put(self.model_state.step, step_sh)

        self.model_state = self.model_state.replace(
            graphstate=graphstate,
            graphother=graphother,
            opt_state=opt_state,
            step=step,
        )
        self.state_shardings = self.model_state.replace(
            graphstate=graphstate_sh,
            graphother=graphother_sh,
            opt_state=opt_sh,
            step=step_sh,
        )

    def _make_mesh(self) -> Mesh:
        devices = jax.devices()
        n = int(len(devices))
        if not self.arguments.spmd:
            return Mesh(np.array(devices).reshape((n, 1)), axis_names=("dp", "tp"))
        dp = int(self.arguments.dp)
        tp_size = int(self.arguments.tp)
        need = dp * tp_size
        if need > n:
            raise ValueError(f"Invalid dp/tp for devices: dp={dp} tp={tp_size} devices={n}")
        dev = np.array(devices[:need]).reshape((dp, tp_size))
        return Mesh(dev, axis_names=("dp", "tp"))

    def configure_dataloaders(self) -> TrainerConfigureDataloaderOutput:
        bs = int(self.training_batch_size)
        if bs <= 0:
            raise ValueError(f"Invalid training_batch_size={bs}")

        use_spmd = bool(self.arguments.spmd)
        dp = int(self.arguments.dp) if use_spmd else int(jax.local_device_count())
        if bs % dp != 0:
            raise ValueError(f"training_batch_size={bs} must be divisible by dp={dp}")

        steps = int(self.arguments.max_training_steps or 0)
        if steps <= 0:
            epochs = float(self.arguments.num_train_epochs or 1.0)
            steps = int(max(1, (len(self.cache) * epochs) // bs))

        seed = int(getattr(self.arguments, "seed", 0) or 0)
        shuffle = bool(getattr(self.arguments, "shuffle_train_dataset", True))
        prefetch = int(getattr(self.arguments, "dataloader_prefetch", 128) or 128)
        workers = max(1, int(getattr(self.arguments, "dataloader_workers", 8) or 8))

        class _CachePrefetchLoader:
            def __iter__(self_inner):
                rng = np.random.default_rng(seed)
                n = len(self.cache)
                order = np.arange(n, dtype=np.int64)
                if shuffle:
                    rng.shuffle(order)
                pos = 0

                q: Queue = Queue(maxsize=prefetch)

                def _worker():
                    nonlocal pos
                    while True:
                        if bs > n:
                            idx = rng.integers(0, n, size=bs, dtype=np.int64)
                            q.put(self.cache.get_batch(idx))
                            continue
                        if pos + bs > n:
                            if shuffle:
                                rng.shuffle(order)
                            pos = 0
                        idx = order[pos : pos + bs]
                        pos += bs
                        q.put(self.cache.get_batch(idx))

                for _ in range(workers):
                    Thread(target=_worker, daemon=True).start()

                per = bs // dp
                while True:
                    batch = q.get()
                    if not use_spmd:
                        batch = {
                            "context_features_u16": batch["context_features_u16"].reshape(
                                (dp, per) + tuple(batch["context_features_u16"].shape[1:])
                            ),
                            "anchor_embedding_u16": batch["anchor_embedding_u16"].reshape(
                                (dp, per) + tuple(batch["anchor_embedding_u16"].shape[1:])
                            ),
                            "target_ids": batch["target_ids"].reshape((dp, per) + tuple(batch["target_ids"].shape[1:])),
                        }
                    yield batch

        return TrainerConfigureDataloaderOutput(
            dataloader_train=_CachePrefetchLoader(),
            max_training_steps=steps,
            dataloader_eval=None,
            max_evaluation_steps=0,
        )

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        if bool(self.arguments.spmd):
            raise NotImplementedError("SPMD path is not wired for this trainer yet. Use --spmd false.")

        mesh = self._make_mesh()
        dp_size = int(mesh.shape["dp"])
        if dp_size != int(jax.local_device_count()):
            raise ValueError("dp-only expects dp == local_device_count")

        grad_accum = int(getattr(self.arguments, "gradient_accumulation_steps", 1) or 1)
        if grad_accum < 1:
            raise ValueError(f"Invalid gradient_accumulation_steps={grad_accum}")

        rope = self._rope

        w = jax.lax.stop_gradient(self._lm_head_weight)
        vocab = int(w.shape[0])
        h = int(w.shape[1])
        requested_chunk = int(self.arguments.vocab_chunk_size or 0)
        vocab_chunk = _choose_vocab_chunk(vocab_size=vocab, requested=requested_chunk)
        if vocab_chunk <= 0:
            raise ValueError("Set --vocab-chunk-size to a reasonable value (e.g. 8192).")

        w_repl = jax.device_put(w, NamedSharding(mesh, P()))

        def _nll_chunked(*, hs, labels):
            hs_f = hs.astype(jnp.float32)
            y = labels.astype(jnp.int32)
            w_bf16 = w_repl

            chunk = int(vocab_chunk)
            n_chunks = vocab // chunk

            def scan_body(carry, i):
                max_prev, sumexp_prev, true_logit_prev = carry
                start = i * chunk
                w_c = jax.lax.dynamic_slice(w_bf16, (start, 0), (chunk, int(h)))
                logits_c = jnp.einsum(
                    "bsh,vh->bsv",
                    hs_f,
                    w_c.astype(jnp.float32),
                    precision=jax.lax.Precision.HIGHEST,
                )

                max_c = jnp.max(logits_c, axis=-1)
                max_new = jnp.maximum(max_prev, max_c)
                sumexp_new = sumexp_prev * jnp.exp(max_prev - max_new) + jnp.sum(
                    jnp.exp(logits_c - max_new[..., None]), axis=-1
                )

                in_chunk = (y >= start) & (y < (start + chunk))
                off = jnp.clip(y - start, 0, chunk - 1).astype(jnp.int32)
                gathered = jnp.take_along_axis(logits_c, off[..., None], axis=-1)[..., 0]
                true_logit_new = jnp.where(in_chunk, gathered, true_logit_prev)
                return (max_new, sumexp_new, true_logit_new), None

            init = (
                jnp.full(hs.shape[:2], -jnp.inf, dtype=jnp.float32),
                jnp.zeros(hs.shape[:2], dtype=jnp.float32),
                jnp.full(hs.shape[:2], -jnp.inf, dtype=jnp.float32),
            )
            (max_final, sumexp_final, true_final), _ = jax.lax.scan(
                scan_body, init, jnp.arange(n_chunks, dtype=jnp.int32)
            )
            logz = max_final + jnp.log(sumexp_final + 1e-9)
            return jnp.mean(logz - true_final)

        def _micro_loss_and_grads(state: EasyDeLState, batch: dict):
            batch_obj = _batch_from_dict(batch)
            context = _bf16_from_u16(batch_obj.context_u16)
            anchor = _bf16_from_u16(batch_obj.anchor_u16)
            labels = batch_obj.target_ids.astype(jnp.int32)

            def loss_fn(graphstate):
                module = nnx.merge(state.graphdef, graphstate, state.graphother)
                out = module(context_features=context, anchor_embedding=anchor, rope=rope)
                hs = out[:, 1:, :]
                loss = _nll_chunked(hs=hs, labels=labels)
                loss = jax.lax.pmean(loss, "dp")
                return loss

            loss, grads = jax.value_and_grad(loss_fn)(state.graphstate)
            grads = jax.tree_util.tree_map(lambda g: jax.lax.pmean(g, "dp"), grads)
            return loss, grads

        def dflash_train_step(state: EasyDeLState, batch: dict):
            if grad_accum == 1:
                loss, grads = _micro_loss_and_grads(state, batch)
                state2 = state.apply_gradients(grads=grads)
                return state2, LossMetrics(loss=loss, accuracy=jnp.array(0.0, dtype=jnp.float32))

            def split(x):
                b0 = x.shape[0]
                if b0 % grad_accum != 0:
                    raise ValueError(f"per-device batch {b0} must be divisible by grad_accum={grad_accum}")
                return x.reshape((grad_accum, b0 // grad_accum) + x.shape[1:])

            batch_s = {
                "context_features_u16": split(batch["context_features_u16"]),
                "anchor_embedding_u16": split(batch["anchor_embedding_u16"]),
                "target_ids": split(batch["target_ids"]),
            }

            def body(carry, micro):
                loss_sum, grads_sum = carry
                loss, grads = _micro_loss_and_grads(state, micro)
                loss_sum = loss_sum + loss
                grads_sum = grads if grads_sum is None else jax.tree_util.tree_map(lambda a, b: a + b, grads_sum, grads)
                return (loss_sum, grads_sum), None

            init = (jnp.array(0.0, dtype=jnp.float32), None)
            (loss_sum, grads_sum), _ = jax.lax.scan(body, init, batch_s)
            loss = loss_sum / float(grad_accum)
            grads = jax.tree_util.tree_map(lambda g: g / float(grad_accum), grads_sum)
            state2 = state.apply_gradients(grads=grads)
            return state2, LossMetrics(loss=loss, accuracy=jnp.array(0.0, dtype=jnp.float32))

        sharded_training_step_function = jax.pmap(
            dflash_train_step,
            axis_name="dp",
            in_axes=(None, 0),
            out_axes=(None, None),
            donate_argnums=(0,),
        )

        def dflash_eval_step(state: EasyDeLState, batch: dict):
            batch_obj = _batch_from_dict(batch)
            context = _bf16_from_u16(batch_obj.context_u16)
            anchor = _bf16_from_u16(batch_obj.anchor_u16)
            labels = batch_obj.target_ids.astype(jnp.int32)
            module = nnx.merge(state.graphdef, state.graphstate, state.graphother)
            out = module(context_features=context, anchor_embedding=anchor, rope=rope)
            hs = out[:, 1:, :]
            loss = _nll_chunked(hs=hs, labels=labels)
            loss = jax.lax.pmean(loss, "dp")
            return LossMetrics(loss=loss, accuracy=jnp.array(0.0, dtype=jnp.float32))

        sharded_evaluation_step_function = jax.pmap(
            dflash_eval_step,
            axis_name="dp",
            in_axes=(None, 0),
            out_axes=None,
        )

        self.arguments.ensure_checkpoint_path()
        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=mesh,
            checkpoint_manager=self.arguments.get_streaming_checkpointer(),
        )

    def on_step_end(self, state: EasyDeLState, metrics: LossMetrics, step: int):
        try:
            if jax.process_index() == 0:
                every = int(getattr(self.arguments, "report_steps", 0) or 0)
                if every > 0 and (int(step) % every == 0):
                    dev = jax.devices()[0]
                    mem = getattr(dev, "memory_stats", None)
                    if callable(mem):
                        st = mem()
                        used = st.get("memory_used", None) or st.get("hbm_memory_used", None)
                        limit = st.get("memory_limit", None) or st.get("hbm_memory_total", None)
                        if used is not None:
                            other = dict(metrics.other_metrics or {})
                            other["tpu_mem_used_gb"] = float(used) / (1024**3)
                            if limit is not None:
                                other["tpu_mem_limit_gb"] = float(limit) / (1024**3)
                                other["tpu_mem_used_pct"] = 100.0 * float(used) / float(limit)
                            metrics = metrics.replace(other_metrics=other)
        except Exception:
            pass
        return state, metrics


def _dflash_collate(examples: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    ctx = np.stack([ex["context_features_u16"] for ex in examples], axis=0).astype(np.uint16, copy=False)
    anc = np.stack([ex["anchor_embedding_u16"] for ex in examples], axis=0).astype(np.uint16, copy=False)
    tgt = np.stack([ex["target_ids"] for ex in examples], axis=0).astype(np.int32, copy=False)
    return {"context_features_u16": ctx, "anchor_embedding_u16": anc, "target_ids": tgt}


def _idx_collate(examples: list[dict[str, tp.Any]]) -> dict[str, np.ndarray]:
    idx = np.asarray([int(ex["idx"]) for ex in examples], dtype=np.int64)
    return {"idx": idx}


def _dflash_or_idx_collate(batch: tp.Any) -> dict[str, np.ndarray]:
    if isinstance(batch, dict):
        return batch
    if isinstance(batch, list):
        if batch and isinstance(batch[0], dict) and "context_features_u16" in batch[0]:
            return _dflash_collate(batch)  # type: ignore[arg-type]
        if batch and isinstance(batch[0], dict) and "idx" in batch[0]:
            return _idx_collate(batch)  # type: ignore[arg-type]
    return batch
