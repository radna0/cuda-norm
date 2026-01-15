from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _set_env_sane_defaults() -> None:
    # Kaggle images often have optional deps (tf/sklearn) that are ABI-mismatched with numpy.
    # Keep Transformers from pulling them in.
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def _maybe_load_repo_dotenv() -> None:
    # Best-effort load of synced `.env` (Kaggle via Versa) without printing secrets.
    for candidate in (
        Path("/kaggle/working/cuda-norm-sync/.env"),
        Path(__file__).resolve().parents[1] / ".env",
    ):
        try:
            if not candidate.exists():
                continue
            for raw in candidate.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                if not key:
                    continue
                val = val.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                    val = val[1:-1]
                os.environ.setdefault(key, val)
            return
        except Exception:
            continue


def _predownload_model_and_data(*, model_id: str, dataset_repo: str, train_files: list[str]) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    print("[train] predownload: model snapshot_download start", flush=True)
    snapshot_download(repo_id=str(model_id), repo_type="model", token=token)
    print("[train] predownload: model snapshot_download done", flush=True)
    for f in train_files:
        print(f"[train] predownload: dataset hf_hub_download start: {f}", flush=True)
        hf_hub_download(repo_id=str(dataset_repo), repo_type="dataset", filename=str(f), token=token)
    print("[train] predownload: dataset downloads done", flush=True)


def _download_dataset_file(dataset_repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    return Path(
        hf_hub_download(
            repo_id=str(dataset_repo),
            repo_type="dataset",
            filename=str(filename),
            token=token,
        )
    )

def _download_model_file(model_id: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    return Path(
        hf_hub_download(
            repo_id=str(model_id),
            repo_type="model",
            filename=str(filename),
            token=token,
        )
    )


def _load_target_config_json(model_id: str) -> dict:
    import json

    cfg_path = _download_model_file(model_id, "config.json")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Invalid target config.json")
    return cfg


class _Tokenizer:
    def __init__(self, tok_json: Path, *, eos_token: str, pad_token: str) -> None:
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(tok_json))
        self.eos_token = str(eos_token)
        self.pad_token = str(pad_token)
        self.eos_token_id = self._tok.token_to_id(self.eos_token)
        self.pad_token_id = self._tok.token_to_id(self.pad_token)
        if self.eos_token_id is None or self.pad_token_id is None:
            raise ValueError("Failed to resolve eos/pad token ids from tokenizer.json")

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text).ids)


def _load_tokenizer(model_id: str) -> _Tokenizer:
    import json

    tok_json = _download_model_file(model_id, "tokenizer.json")
    special_path = _download_model_file(model_id, "special_tokens_map.json")
    special = json.loads(special_path.read_text(encoding="utf-8"))
    eos = (special.get("eos_token") or {}).get("content") if isinstance(special.get("eos_token"), dict) else special.get("eos_token")
    pad = (special.get("pad_token") or {}).get("content") if isinstance(special.get("pad_token"), dict) else special.get("pad_token")
    eos = eos or "<|eos|>"
    pad = pad or "<|pad|>"
    return _Tokenizer(tok_json, eos_token=str(eos), pad_token=str(pad))


def _iter_parquet_texts(parquet_path: Path, *, text_column: str = "text") -> Iterable[str]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(parquet_path))
    for rg in range(pf.num_row_groups):
        tab = pf.read_row_group(rg, columns=[text_column])
        col = tab.column(text_column)
        for v in col.to_pylist():
            if isinstance(v, str) and v.strip():
                yield v


def _union_round_robin(iters: list[Iterable[str]]) -> Iterable[str]:
    active = [iter(it) for it in iters]
    while active:
        nxt = []
        for it in active:
            try:
                yield next(it)
                nxt.append(it)
            except StopIteration:
                pass
        active = nxt


@dataclass(frozen=True)
class TrainBatch:
    context_ids: "torch.LongTensor"
    target_ids: "torch.LongTensor"


def _make_training_stream(
    tok: _Tokenizer,
    *,
    eos_id: int | None,
    seq_len: int,
    block_size: int,
    texts: Iterable[str],
    mask_token_id: int,
) -> Iterable[TrainBatch]:
    import torch

    # In SGLang DFLASH, `block_size` counts the anchor token (verified_id) + (block_size-1)
    # drafted tokens. Training should match that regime:
    #   - input block token0 = last token of context (verified_id)
    #   - input block token1..B-1 = mask
    #   - labels are the *next* (B-1) tokens after the context.
    need = int(seq_len) + max(1, int(block_size) - 1)
    buf: list[int] = []
    for t in texts:
        ids = tok.encode(t)
        if eos_id is not None:
            ids = ids + [int(eos_id)]
        buf.extend(ids)
        while len(buf) >= need:
            chunk = buf[:need]
            buf = buf[need:]
            context = torch.tensor(chunk[:seq_len], dtype=torch.long)
            targets = torch.tensor(chunk[seq_len:], dtype=torch.long)
            yield TrainBatch(context_ids=context, target_ids=targets)


def _infer_mask_token_id(tok) -> int:
    return int(tok.pad_token_id)


def main() -> None:
    _set_env_sane_defaults()
    _maybe_load_repo_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--target-model", default="openai/gpt-oss-20b")
    ap.add_argument("--dataset-repo", required=True)
    ap.add_argument("--train-files-csv", required=True, help="Comma-separated dataset file paths (parquet).")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--num-hidden-layers", type=int, default=4)
    ap.add_argument("--mlp-ratio", type=float, default=4.0)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--out-root", default="/kaggle/working/dflash_gptoss20b")
    ap.add_argument("--predownload-only", action="store_true")
    ap.add_argument("--teacher-attn-backend", default="fa3", choices=["fa3", "flashinfer", "trtllm"])
    ap.add_argument("--teacher-mem-fraction", type=float, default=0.75)
    args = ap.parse_args()

    train_files = [s.strip() for s in str(args.train_files_csv).split(",") if s.strip()]
    if not train_files:
        raise ValueError("No train files provided")

    token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    print(
        "[train] start "
        f"target_model={args.target_model} seq_len={int(args.seq_len)} block_size={int(args.block_size)} "
        f"hidden_layers={int(args.num_hidden_layers)} max_steps={int(args.max_steps)} "
        f"teacher_backend={args.teacher_attn_backend} hf_token_present={token_present}",
        flush=True,
    )

    _predownload_model_and_data(
        model_id=str(args.target_model),
        dataset_repo=str(args.dataset_repo),
        train_files=train_files,
    )
    if args.predownload_only:
        print("[+] predownload ok (predownload-only)", flush=True)
        return

    random.seed(int(args.seed))

    import torch
    from torch.nn import functional as F

    from dflash_gptoss.sglang_inproc_teacher import SGLangInprocTeacher
    from dflash_gptoss.torch_draft import GptOssDFlashDraftConfig, TorchGptOssDFlashDraftModel

    tok = _load_tokenizer(str(args.target_model))
    eos_id = int(tok.eos_token_id)
    mask_token_id = _infer_mask_token_id(tok)

    tcfg = _load_target_config_json(str(args.target_model))
    dcfg = GptOssDFlashDraftConfig(
        target_model_id=str(args.target_model),
        vocab_size=int(tcfg.get("vocab_size", 0)),
        hidden_size=int(tcfg["hidden_size"]),
        num_attention_heads=int(tcfg["num_attention_heads"]),
        num_key_value_heads=int(tcfg.get("num_key_value_heads", tcfg["num_attention_heads"])),
        head_dim=int(tcfg.get("head_dim", int(tcfg["hidden_size"]) // int(tcfg["num_attention_heads"]))),
        max_position_embeddings=int(tcfg.get("max_position_embeddings", 0)),
        rope_theta=float(tcfg.get("rope_theta", 150000.0)),
        rms_norm_eps=float(tcfg.get("rms_norm_eps", 1e-5)),
        attention_bias=bool(tcfg.get("attention_bias", True)),
        num_hidden_layers=int(args.num_hidden_layers),
        num_target_layers=int(tcfg.get("num_hidden_layers", 0)),
        block_size=int(args.block_size),
        mlp_ratio=float(args.mlp_ratio),
        hidden_act=str(tcfg.get("hidden_act", "silu")),
    )
    draft = TorchGptOssDFlashDraftModel(dcfg).to(device="cuda:0", dtype=torch.bfloat16)
    draft.train()

    opt = torch.optim.AdamW(
        draft.parameters(),
        lr=float(args.lr),
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    teacher = SGLangInprocTeacher(
        model_path=str(args.target_model),
        attention_backend=str(args.teacher_attn_backend),
        context_length=int(args.seq_len),
        dtype="bfloat16",
        mem_fraction_static=float(args.teacher_mem_fraction),
        layers_to_capture=list(draft.target_layer_ids),
    )

    paths = [_download_dataset_file(str(args.dataset_repo), f) for f in train_files]
    texts = _union_round_robin([_iter_parquet_texts(p) for p in paths])
    stream = _make_training_stream(
        tok,
        eos_id=eos_id,
        seq_len=int(args.seq_len),
        block_size=int(args.block_size),
        texts=texts,
        mask_token_id=int(mask_token_id),
    )

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(str(args.out_root)).expanduser().resolve() / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "meta.json").write_text(
        __import__("json").dumps(
            {
                "target_model": str(args.target_model),
                "dataset_repo": str(args.dataset_repo),
                "train_files": train_files,
                "seq_len": int(args.seq_len),
                "block_size": int(args.block_size),
                "num_hidden_layers": int(args.num_hidden_layers),
                "mlp_ratio": float(args.mlp_ratio),
                "teacher_attn_backend": str(args.teacher_attn_backend),
                "teacher_mem_fraction": float(args.teacher_mem_fraction),
                "seed": int(args.seed),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    t0 = time.time()
    for step in range(1, int(args.max_steps) + 1):
        batch = next(stream)
        device = torch.device("cuda:0")
        context_ids = batch.context_ids.unsqueeze(0).to(device)
        target_ids = batch.target_ids.unsqueeze(0).to(device)

        # DFlash/SGLang regime: token0 is the last verified token in the prefix.
        # Draft tokens are positions 1..B-1, starting at the *next* token after the prefix.
        anchor_id = context_ids[:, -1:].contiguous()  # [1, 1]
        noise_block_ids = torch.empty((1, int(args.block_size)), device=device, dtype=torch.long)
        noise_block_ids[:, 0:1].copy_(anchor_id)
        noise_block_ids[:, 1:].fill_(int(mask_token_id))

        # Teacher prefill features + embeddings.
        with torch.no_grad():
            t_out = teacher.prefill_hidden_states(context_ids)
            target_hidden = t_out.hidden_states.unsqueeze(0)
            expected_feat = int(len(draft.target_layer_ids) * int(dcfg.hidden_size))
            got_feat = int(target_hidden.shape[-1])
            if got_feat != expected_feat:
                raise RuntimeError(
                    "SGLang teacher hidden-state feature dim mismatch for DFlash. "
                    f"Expected last_dim={expected_feat} (=K*hidden, K={len(draft.target_layer_ids)}, hidden={int(dcfg.hidden_size)}), "
                    f"but got last_dim={got_feat}. "
                    "This usually means the SGLang target model is not capturing the requested target-layer features "
                    "(dflash/eagle hidden capture not enabled, overlay not applied, or layers_to_capture not set)."
                )
            base_noise_embedding = (
                teacher.embed_tokens(noise_block_ids)
                .reshape(1, int(args.block_size), -1)
                .contiguous()
            )

        # Learned mask embedding (differentiable).
        mask = (noise_block_ids == int(mask_token_id)).view(1, int(args.block_size), 1)
        mask_embed = draft.mask_embedding.to(base_noise_embedding.dtype).view(1, 1, -1)
        noise_embedding = torch.where(mask, mask_embed, base_noise_embedding)

        rope_cos, rope_sin = draft.build_rope_for_lengths(
            ctx_len=int(args.seq_len),
            block_size=int(args.block_size),
            head_dim=int(dcfg.head_dim),
            rope_theta=float(dcfg.rope_theta),
            device=device,
            dtype=base_noise_embedding.dtype,
        )
        draft_out = draft(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )

        # Project to vocab using teacher lm_head weights.
        flat = draft_out[:, 1:, :].reshape(-1, int(draft_out.size(-1)))
        flat_logits = teacher.lm_head_logits(flat)
        logits = flat_logits.view(1, -1, flat_logits.size(-1))

        labels = target_ids
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="mean",
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
        opt.step()

        if step % int(args.log_every) == 0 or step == 1:
            dt = max(1e-9, time.time() - t0)
            # proxy tokens/sec (teacher prefill tokens + draft block tokens)
            tok_s = (step * int(args.seq_len + args.block_size)) / dt
            print(f"[step {step}] loss={loss.item():.4f} tok_s(train_proxy)={tok_s:.1f}", flush=True)

        if step % int(args.save_every) == 0 or step == int(args.max_steps):
            ckpt = out_root / f"step_{step:06d}"
            ckpt.mkdir(parents=True, exist_ok=True)
            # Save in HF-like layout expected by our converter (config.json + model.safetensors + tokenizer files).
            import json
            from safetensors.torch import save_file

            cfg_out = {
                "target_model_id": str(dcfg.target_model_id),
                "vocab_size": int(dcfg.vocab_size),
                "hidden_size": int(dcfg.hidden_size),
                "num_attention_heads": int(dcfg.num_attention_heads),
                "num_key_value_heads": int(dcfg.num_key_value_heads),
                "head_dim": int(dcfg.head_dim),
                "max_position_embeddings": int(dcfg.max_position_embeddings),
                "rope_theta": float(dcfg.rope_theta),
                "rope_scaling": None,
                "rms_norm_eps": float(dcfg.rms_norm_eps),
                "attention_bias": bool(dcfg.attention_bias),
                "num_hidden_layers": int(dcfg.num_hidden_layers),
                "num_target_layers": int(dcfg.num_target_layers),
                "block_size": int(dcfg.block_size),
                "mlp_ratio": float(dcfg.mlp_ratio),
                "hidden_act": str(dcfg.hidden_act),
            }
            (ckpt / "config.json").write_text(json.dumps(cfg_out, indent=2), encoding="utf-8")

            state = {k: v.detach().to("cpu") for k, v in draft.state_dict().items()}
            save_file(state, str(ckpt / "model.safetensors"), metadata={"format": "torch_draft"})

            # Copy tokenizer + special tokens metadata for converter mask-token resolution.
            shutil_copy = __import__("shutil").copy2
            shutil_copy(_download_model_file(str(args.target_model), "tokenizer.json"), ckpt / "tokenizer.json")
            shutil_copy(_download_model_file(str(args.target_model), "special_tokens_map.json"), ckpt / "special_tokens_map.json")
            print(f"[+] saved {ckpt}", flush=True)

    teacher.close()
    print(f"[+] done. out_root={out_root}", flush=True)


if __name__ == "__main__":
    main()
