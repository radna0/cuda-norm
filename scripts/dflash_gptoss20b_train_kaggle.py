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
    snapshot_download(repo_id=str(model_id), repo_type="model", token=token)
    for f in train_files:
        hf_hub_download(repo_id=str(dataset_repo), repo_type="dataset", filename=str(f), token=token)


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
    block_ids: "torch.LongTensor"
    noise_block_ids: "torch.LongTensor"


def _make_training_stream(
    tok,
    *,
    eos_id: int | None,
    seq_len: int,
    block_size: int,
    texts: Iterable[str],
    mask_token_id: int,
) -> Iterable[TrainBatch]:
    import torch

    need = int(seq_len) + int(block_size)
    buf: list[int] = []
    for t in texts:
        ids = tok(t, add_special_tokens=False).input_ids
        if eos_id is not None:
            ids = ids + [int(eos_id)]
        buf.extend(ids)
        while len(buf) >= need:
            chunk = buf[:need]
            buf = buf[need:]
            context = torch.tensor(chunk[:seq_len], dtype=torch.long)
            block = torch.tensor(chunk[seq_len:], dtype=torch.long)

            # Inference-time DFlash regime: token0 is anchor/current token; all future tokens masked.
            noise = block.clone()
            noise[1:] = int(mask_token_id)

            yield TrainBatch(context_ids=context, block_ids=block, noise_block_ids=noise)


def _infer_mask_token_id(tok) -> int:
    # For GPT-OSS + SGLang we must not resize vocab. Use an existing token id.
    if getattr(tok, "mask_token_id", None) is not None:
        return int(tok.mask_token_id)
    if getattr(tok, "pad_token_id", None) is not None:
        return int(tok.pad_token_id)
    raise ValueError("Tokenizer must define pad_token_id or mask_token_id")


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
    from transformers import AutoConfig, AutoTokenizer

    from dflash_gptoss.modeling_gptoss_dflash import GptOssDFlashDraftModel
    from dflash_gptoss.sglang_inproc_teacher import SGLangInprocTeacher

    tok = AutoTokenizer.from_pretrained(str(args.target_model))
    eos_id = int(tok.eos_token_id) if tok.eos_token_id is not None else None
    mask_token_id = _infer_mask_token_id(tok)

    cfg = AutoConfig.from_pretrained(str(args.target_model))
    draft = GptOssDFlashDraftModel.from_target_config(
        target_model_id=str(args.target_model),
        target_config=cfg,
        num_hidden_layers=int(args.num_hidden_layers),
        block_size=int(args.block_size),
        mlp_ratio=float(args.mlp_ratio),
    ).to(device="cuda:0", dtype=torch.bfloat16)
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
        block_ids = batch.block_ids.unsqueeze(0).to(device)
        noise_block_ids = batch.noise_block_ids.unsqueeze(0).to(device)

        # Teacher prefill features + embeddings.
        with torch.no_grad():
            t_out = teacher.prefill_hidden_states(context_ids)
            target_hidden = t_out.hidden_states.unsqueeze(0)
            base_noise_embedding = (
                teacher.embed_tokens(noise_block_ids)
                .reshape(1, int(args.block_size), -1)
                .contiguous()
            )

        # Learned mask embedding (differentiable).
        mask = (noise_block_ids == int(mask_token_id)).view(1, int(args.block_size), 1)
        mask_embed = draft.mask_embedding.to(base_noise_embedding.dtype).view(1, 1, -1)
        noise_embedding = torch.where(mask, mask_embed, base_noise_embedding)

        pos = torch.arange(int(args.seq_len) + int(args.block_size), device=device).unsqueeze(0)
        draft_out = draft(
            position_ids=pos,
            attention_mask=None,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            use_cache=False,
        )

        # Project to vocab using teacher lm_head weights.
        flat = draft_out[:, 1:, :].reshape(-1, draft_out.size(-1))
        flat_logits = teacher.lm_head_logits(flat)
        logits = flat_logits.view(1, -1, flat_logits.size(-1))

        labels = block_ids[:, 1:]
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
            tok_s = (step * int(args.seq_len + args.block_size)) / dt
            print(f"[step {step}] loss={loss.item():.4f} tok_s(train_proxy)={tok_s:.1f}", flush=True)

        if step % int(args.save_every) == 0 or step == int(args.max_steps):
            ckpt = out_root / f"step_{step:06d}"
            ckpt.mkdir(parents=True, exist_ok=True)
            # Save draft weights (HF format) and tokenizer snapshot for mask/pad metadata.
            draft.save_pretrained(str(ckpt), safe_serialization=True)
            tok.save_pretrained(str(ckpt))
            print(f"[+] saved {ckpt}", flush=True)

    teacher.close()
    print(f"[+] done. out_root={out_root}", flush=True)


if __name__ == "__main__":
    main()
