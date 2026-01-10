# modal/nll_scoring.py
# Compute completion-only NLL/PPL on Harmony-formatted `text` rows.
#
# This is the GPU stage (Modal). CPU stage prepares Parquet candidate pools containing:
#   id, text, loss_mode, meta_*, quality_*
#
# This job:
# - downloads candidate Parquet files from a HF dataset repo
# - masks loss to assistant message bodies (loss_mode=assistant_all)
# - writes Parquet score shards to the persistent data volume

from __future__ import annotations

import os
import time
from pathlib import Path

import modal

APP_NAME = "harmony-nll-scoring"

# Optional: pass your local `HF_TOKEN` into the Modal container at launch time.
_secrets = []
_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    _secrets.append(modal.Secret.from_dict({"HF_TOKEN": _hf_token}))

# Non-secret run configuration: Modal does not automatically propagate local
# environment variables into containers, so we pass the knobs explicitly.
_run_env: dict[str, str | None] = {
    "CANDIDATE_DATASET_ID": os.environ.get("CANDIDATE_DATASET_ID"),
    "CANDIDATE_SUBDIR": os.environ.get("CANDIDATE_SUBDIR"),
    "MODEL_NAME": os.environ.get("MODEL_NAME"),
    "MAX_LENGTH": os.environ.get("MAX_LENGTH"),
    "BATCH_SIZE": os.environ.get("BATCH_SIZE"),
    "MAX_RECORDS": os.environ.get("MAX_RECORDS"),
    "RUN_TAG": os.environ.get("RUN_TAG"),
}

data_volume = modal.Volume.from_name("mhc-data-volume", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("hf-cache-persistent", create_if_missing=True)

BASE_IMAGE = "nvidia/cuda:12.8.0-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.12")
    .apt_install("git", "build-essential", "clang", "python3-dev", "curl")
    .run_commands("python -m pip install -U pip setuptools wheel")
    .run_commands(
        "python -m pip install "
        "torch==2.9.0 "
        "--extra-index-url https://download.pytorch.org/whl/cu128"
    )
    .run_commands(
        "python -m pip install "
        "datasets==3.2.0 pyarrow==22.0.0 requests accelerate "
        "huggingface_hub hf_transfer "
        "transformers==4.56.2 bitsandbytes==0.47.0"
    )
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    env=_run_env,
    gpu="B200:1",
    volumes={"/root/data": data_volume, "/root/hf_cache": hf_cache_volume},
    secrets=_secrets,
    timeout=60 * 60 * 12,
    cpu=16.0,
    memory=262144,
)
def score_nll() -> str:
    import json
    import math
    from dataclasses import dataclass, field
    from hashlib import sha1
    from typing import Any

    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    START_TAG = "<|start|>"
    END_TAG = "<|end|>"
    CALL_TAG = "<|call|>"
    CHANNEL_TAG = "<|channel|>"
    MESSAGE_TAG = "<|message|>"
    RETURN_TAG = "<|return|>"

    def assistant_content_spans(text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        if not isinstance(text, str):
            return spans

        i = 0
        n = len(text)
        while True:
            start = text.find(START_TAG, i)
            if start < 0:
                break

            role_start = start + len(START_TAG)
            msg_tag = text.find(MESSAGE_TAG, role_start)
            if msg_tag < 0:
                raise ValueError("malformed harmony text: missing <|message|>")

            header = text[role_start:msg_tag]
            if CHANNEL_TAG in header:
                role = header.split(CHANNEL_TAG, 1)[0]
            else:
                role = header

            content_start = msg_tag + len(MESSAGE_TAG)
            end_pos = text.find(END_TAG, content_start)
            call_pos = text.find(CALL_TAG, content_start)
            return_pos = text.find(RETURN_TAG, content_start)

            candidates: list[tuple[int, str]] = []
            if end_pos >= 0:
                candidates.append((end_pos, END_TAG))
            if call_pos >= 0:
                candidates.append((call_pos, CALL_TAG))
            if return_pos >= 0:
                candidates.append((return_pos, RETURN_TAG))

            if not candidates:
                raise ValueError(
                    "malformed harmony text: missing <|end|>, <|call|>, or <|return|>"
                )

            delim_pos, delim_tag = min(candidates, key=lambda t: t[0])
            if role == "assistant" or role.startswith("assistant "):
                spans.append((content_start, delim_pos))

            i = delim_pos + len(delim_tag)
            if i >= n:
                break

        return spans

    os.environ.setdefault("HF_HOME", "/root/hf_cache")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    dataset_id = os.environ.get("CANDIDATE_DATASET_ID")
    if not dataset_id:
        raise RuntimeError("Set CANDIDATE_DATASET_ID (HF dataset repo_id with candidate parquet files)")
    subdir = os.environ.get("CANDIDATE_SUBDIR", "").strip() or None

    model_name = os.environ.get("MODEL_NAME", "unsloth/gpt-oss-120b")
    max_length = int(os.environ.get("MAX_LENGTH", "4096"))
    batch_size = int(os.environ.get("BATCH_SIZE", "1"))
    max_records = int(os.environ.get("MAX_RECORDS", "0"))  # 0 = all

    out_tag = os.environ.get("RUN_TAG", "")
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_id = out_tag or sha1(f"{dataset_id}|{subdir}|{model_name}|{ts}".encode("utf-8")).hexdigest()[:12]
    out_dir = Path("/root/data/nll_scores") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = ["**/*.parquet", "README.md"]
    if subdir:
        allow_patterns = [f"{subdir}/**", "README.md"]

    print(f"[*] snapshot_download: {dataset_id} subdir={subdir!r}")
    snap_path = snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
    )
    snap = Path(snap_path)
    scan_root = (snap / subdir) if subdir else snap
    parquet_files = sorted(scan_root.rglob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"no parquet files found under {scan_root}")
    print(f"[*] found {len(parquet_files)} parquet files")

    print(f"[*] loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "Need a fast tokenizer for return_offsets_mapping=True. "
            "Try a different MODEL_NAME or ensure a fast tokenizer is available."
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map={"": 0},
    )
    model.eval()

    @dataclass
    class ScoreWriter:
        out_dir: Path
        rows_per_shard: int = 50_000
        compression: str = "zstd"
        shard_index: int = 0
        rows_in_shard: int = 0
        buf: dict[str, list[Any]] = field(default_factory=dict)

        def add(self, row: dict[str, Any]) -> None:
            for k, v in row.items():
                self.buf.setdefault(k, []).append(v)
            self.rows_in_shard += 1
            if self.rows_in_shard >= self.rows_per_shard:
                self.flush()

        def flush(self) -> None:
            if not self.rows_in_shard:
                return
            table = pa.table(self.buf)
            path = self.out_dir / f"part-{self.shard_index:05d}.parquet"
            pq.write_table(table, path, compression=self.compression)
            print(f"[write] {path} rows={self.rows_in_shard}")
            self.shard_index += 1
            self.rows_in_shard = 0
            self.buf = {}

    writer = ScoreWriter(out_dir=out_dir)

    processed = 0
    for pf in parquet_files:
        parquet = pq.ParquetFile(pf)
        cols = parquet.schema.names
        need_cols = ["id", "text"]
        for c in need_cols:
            if c not in cols:
                raise RuntimeError(f"missing required column {c!r} in {pf}")

        for batch in parquet.iter_batches(columns=["id", "text", "loss_mode"], batch_size=256):
            ids = batch.column(0).to_pylist()
            texts = batch.column(1).to_pylist()
            loss_modes = (
                batch.column(2).to_pylist()
                if batch.num_columns >= 3
                else ["assistant_all"] * len(texts)
            )

            # Micro-batch to avoid OOM.
            for start in range(0, len(texts), batch_size):
                sub_ids = ids[start : start + batch_size]
                sub_texts = texts[start : start + batch_size]
                sub_modes = loss_modes[start : start + batch_size]

                spans_per_ex = []
                for t, mode in zip(sub_texts, sub_modes):
                    if not isinstance(t, str):
                        spans_per_ex.append([])
                        continue
                    if mode != "assistant_all":
                        spans_per_ex.append([])
                        continue
                    try:
                        spans_per_ex.append(assistant_content_spans(t))
                    except Exception:
                        spans_per_ex.append([])

                enc = tokenizer(
                    sub_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                input_ids = enc["input_ids"].to("cuda", non_blocking=True)
                attention_mask = enc["attention_mask"].to("cuda", non_blocking=True)
                offsets = enc["offset_mapping"]  # cpu tensor/list of lists

                labels = input_ids.clone()
                labels[attention_mask == 0] = -100

                # Apply assistant-only mask using char offsets.
                for bi in range(len(sub_texts)):
                    spans = spans_per_ex[bi]
                    if not spans:
                        labels[bi, :] = -100
                        continue
                    for ti, (cs, ce) in enumerate(offsets[bi].tolist()):
                        if cs == 0 and ce == 0:
                            labels[bi, ti] = -100
                            continue
                        keep = False
                        for ss, ee in spans:
                            if ce > ss and cs < ee:
                                keep = True
                                break
                        if not keep:
                            labels[bi, ti] = -100

                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = out.logits

                # Per-example completion-only NLL.
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                bsz, seqlen, vocab = shift_logits.shape
                loss_flat = F.cross_entropy(
                    shift_logits.view(-1, vocab),
                    shift_labels.view(-1),
                    reduction="none",
                    ignore_index=-100,
                )
                loss_tok = loss_flat.view(bsz, seqlen)
                tok_mask = shift_labels != -100
                nll_sum = (loss_tok * tok_mask).sum(dim=1)
                tok_count = tok_mask.sum(dim=1)
                nll_mean = nll_sum / tok_count.clamp_min(1)
                ppl = torch.exp(nll_mean.clamp_max(30.0))

                for bi in range(bsz):
                    tc = int(tok_count[bi].item())
                    writer.add(
                        {
                            "id": str(sub_ids[bi]),
                            "model": model_name,
                            "dataset": dataset_id,
                            "subdir": subdir or "",
                            "max_length": max_length,
                            "nll_sum": float(nll_sum[bi].item()) if tc else math.nan,
                            "nll_mean": float(nll_mean[bi].item()) if tc else math.nan,
                            "ppl": float(ppl[bi].item()) if tc else math.nan,
                            "completion_token_count": tc,
                        }
                    )

                processed += bsz
                if max_records and processed >= max_records:
                    break
            if max_records and processed >= max_records:
                break
        if max_records and processed >= max_records:
            break

    writer.flush()

    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "dataset_id": dataset_id,
        "subdir": subdir,
        "model_name": model_name,
        "max_length": max_length,
        "batch_size": batch_size,
        "processed": processed,
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[ok] wrote {out_dir}/run_manifest.json")

    data_volume.commit()
    hf_cache_volume.commit()

    return str(out_dir)


@app.local_entrypoint()
def main() -> None:
    out_dir = score_nll.remote()
    print(out_dir)
