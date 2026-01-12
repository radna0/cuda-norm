"""
Verify TransMLA-converted GPT-OSS checkpoints in SGLang on Modal.

What this does:
  1) Loads the original GPT-OSS model (GQA) and the converted TransMLA model (MLA)
  2) Computes PPL using SGLang Engine logprobs on the predownloaded Harmony dataset slice
  3) Runs a tiny greedy generation smoke test

Logging:
  - Remote logs are written to a Modal Volume mounted at /logs and committed each run.
  - Also print a short summary to stdout so `modal run ...` captures it.

Expected volumes (same as convert_transmla_sequential.py):
  - /models -> Volume("gpt-oss-model-weights")
  - /data   -> Volume("gpt-oss-harmony-tools")
  - /root/.cache/huggingface -> Volume("hf-hub-cache")
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import modal


app = modal.App("verify-sglang-gptoss-transmla")

model_vol = modal.Volume.from_name("gpt-oss-model-weights", create_if_missing=True)
data_vol = modal.Volume.from_name("gpt-oss-harmony-tools", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
log_vol = modal.Volume.from_name("sglang-gptoss-transmla-verify-logs", create_if_missing=True)

def _find_workspace_root() -> Path:
    """
    Find the local workspace root (the directory that contains 'sglang/' and 'cuda-mla/').

    Note: This must run only on the *local* side. On Modal workers the user code file
    is copied to `/root/<script>.py` and doesn't have the repo checkout.
    """

    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "sglang").exists() and (parent / "cuda-mla").exists():
            return parent
    # Fallback for the expected layout: MODAL/cuda-mla/modal_scripts/<this file>
    if len(here.parents) >= 3:
        return here.parents[2]
    raise RuntimeError(
        "Could not find workspace root. Expected a directory containing both 'sglang/' and 'cuda-mla/'."
    )


# Only compute local paths and attach source trees when running locally.
WORKSPACE_ROOT: Path | None = _find_workspace_root() if modal.is_local() else None
PROJECT_ROOT: Path | None = WORKSPACE_ROOT

def _pick_existing_dir(*candidates: Path) -> Path | None:
    for c in candidates:
        if c and c.exists():
            return c
    return None


FLASHINFER_SRC: Path | None = (
    _pick_existing_dir(
        PROJECT_ROOT / "cuda-flashinfer" / "flashinfer",
        PROJECT_ROOT / "flashinfer",
    )
    if PROJECT_ROOT
    else None
)
SGLANG_PY_SRC: Path | None = (
    _pick_existing_dir(
        # Prefer the conversion team's patched SGLang fork (used by our Modal runs).
        PROJECT_ROOT / "cuda-mla" / "sglang-mla" / "python" / "sglang",
        PROJECT_ROOT / "cuda-flashinfer" / "sglang-flashinfer" / "python" / "sglang",
        PROJECT_ROOT / "sglang-flashinfer" / "python" / "sglang",
    )
    if PROJECT_ROOT
    else None
)
SGL_KERNEL_PY_SRC: Path | None = (
    _pick_existing_dir(
        PROJECT_ROOT / "cuda-mla" / "sglang-mla" / "sgl-kernel" / "python" / "sgl_kernel",
        PROJECT_ROOT
        / "cuda-flashinfer"
        / "sglang-flashinfer"
        / "sgl-kernel"
        / "python"
        / "sgl_kernel",
        PROJECT_ROOT / "sglang-flashinfer" / "sgl-kernel" / "python" / "sgl_kernel",
    )
    if PROJECT_ROOT
    else None
)
CUDA_MLA_SCRIPTS_SRC: Path | None = PROJECT_ROOT / "cuda-mla" / "scripts" if PROJECT_ROOT else None

# Default: do not build/mount FlashInfer for verification runs.
# For truth-PPL runs we typically use FA3, and mounting FlashInfer from a local checkout
# is fragile (local JIT cache churn can break Modal mounts).
USE_LOCAL_FLASHINFER = os.environ.get("VERIFY_SGLANG_USE_LOCAL_FLASHINFER", "0").strip() == "1"


image = (
    # Mirror modal/sglang_benchmark.py: CUDA devel image + build FlashInfer from source.
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04",
        add_python="3.11",
    )
    .apt_install(
        "git",
        "build-essential",
        "clang",
        "cmake",
        "python3-dev",
        "libnuma-dev",
        "numactl",
        "wget",
        "ninja-build",
    )
    .run_commands(
        "pip install --upgrade pip",
        "pip install 'sglang[all]'",
        # Build dependencies for FlashInfer
        "pip install ninja cmake wheel setuptools packaging",
        # Match other cuda-mla modal scripts (CUDA 12.8).
        # Keep torch in sync with the sglang pinned requirement to avoid pip reinstall churn.
        "pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128",
    )
    .env(
        {
            # Allow using local sgl_kernel python sources even if wheel metadata mismatches.
            "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
            # Keep HF cache persistent across runs.
            "HF_HOME": "/root/.cache/huggingface",
        }
    )
)

if modal.is_local():
    assert PROJECT_ROOT is not None
    assert SGLANG_PY_SRC is not None
    assert SGL_KERNEL_PY_SRC is not None
    assert CUDA_MLA_SCRIPTS_SRC is not None
    if USE_LOCAL_FLASHINFER:
        assert FLASHINFER_SRC is not None

    image = (
        image
        .run_commands(
            # Install remaining runtime deps (keep minimal; avoid unrelated backend churn).
            "pip install 'transformers>=4.57.1,!=4.57.2' sentence-transformers numpy==2.2.0 pandas polars datasets==3.2.0 scipy openai-harmony==0.0.4 sentencepiece protobuf msgspec",
        )
        # Inject patched SGLang source + sgl_kernel python package.
        .add_local_dir(
            str(SGLANG_PY_SRC),
            remote_path="/root/sglang-src",
            copy=True,
            ignore=[
                "**/__pycache__",
                "**/__pycache__/**",
                # Avoid build-churn in unrelated backends; we don't need these for FA3 PPL runs.
                "**/srt/layers/attention/flashinfer_backend.py",
            ],
        )
        .add_local_dir(
            str(SGL_KERNEL_PY_SRC),
            remote_path="/root/sgl-kernel-src",
            copy=True,
            ignore=["**/__pycache__", "**/__pycache__/**"],
        )
        .add_local_dir(
            str(CUDA_MLA_SCRIPTS_SRC),
            remote_path="/root/cuda-mla-scripts",
            copy=True,
            ignore=["**/__pycache__", "**/__pycache__/**"],
        )
        .run_commands(
            # Override installed python sources with our local patched versions.
            "cp -rfv /root/sglang-src/* /usr/local/lib/python3.11/site-packages/sglang/",
            "find /usr/local/lib/python3.11/site-packages/sglang -name '__pycache__' -type d -exec rm -rf {} +",
            "cp -rfv /root/sgl-kernel-src/* /usr/local/lib/python3.11/site-packages/sgl_kernel/",
            "find /usr/local/lib/python3.11/site-packages/sgl_kernel -name '__pycache__' -type d -exec rm -rf {} +",
        )
    )

    if USE_LOCAL_FLASHINFER:
        image = (
            image
            # Inject patched FlashInfer source and build from source.
            .add_local_dir(
                str(FLASHINFER_SRC),
                remote_path="/root/flashinfer",
                copy=True,
                ignore=[
                    ".git",
                    ".git/**",
                    "**/.git",
                    "**/.git/**",
                    # FlashInfer JIT cache is mutable (can be touched by other local runs) and
                    # causes Modal mount build failures ("file was modified during build process").
                    "flashinfer-jit-cache",
                    "flashinfer-jit-cache/**",
                    "**/flashinfer-jit-cache",
                    "**/flashinfer-jit-cache/**",
                ],
            )
            .run_commands(
                "echo '=== Building FlashInfer from source (SM90-only) ==='",
                # Ensure no stale installs shadow the custom build.
                "pip uninstall -y flashinfer flashinfer-python flashinfer-cubin || true",
                # Only build/install the SM90 subset of cubins to keep builds reasonable.
                # NOTE: FlashInfer expects the 'major.minor[a]' form (e.g. '9.0a'), not '90'.
                "cd /root/flashinfer/flashinfer-cubin && FLASHINFER_CUDA_ARCH_LIST=9.0a pip install . --no-build-isolation -v > /root/flashinfer_cubin_build.log 2>&1",
                "cd /root/flashinfer && pip install . --no-build-isolation -v > /root/flashinfer_build.log 2>&1",
                "echo '=== FlashInfer build complete ==='",
                "python3 -c \"import flashinfer; print('FlashInfer version:', getattr(flashinfer, '__version__', 'unknown'))\"",
            )
        )

else:
    # Worker-side imports only execute the function body; image mutation isn't needed.
    pass


def _pick_default_converted_model(models_root: str) -> str | None:
    # Prefer the "latest" sequential bf16 directory by name.
    try:
        candidates = []
        for name in os.listdir(models_root):
            if name.startswith("gptoss-transmla-sequential-bf16"):
                candidates.append(name)
        if candidates:
            candidates.sort()
            return os.path.join(models_root, candidates[-1])
    except Exception:
        pass
    # Fallback to the non-suffixed path used by early conversion runs.
    fallback = os.path.join(models_root, "gptoss-transmla-sequential-bf16")
    return fallback if os.path.exists(fallback) else None


@app.function(
    image=image,
    gpu="H100:1",
    timeout=3 * 60 * 60,
    memory=128 * 1024,
    volumes={
        "/models": model_vol,
        "/data": data_vol,
        "/root/.cache/huggingface": hf_cache_vol,
        "/logs": log_vol,
    },
    retries=0,
)
def verify(
    converted_model_path: str = "",
    seq_len: int = 1024,
    num_samples: int = 64,
    conv_attention_backend: str = "fa3",
    force_mla_backend: bool = False,
    disable_sinks_orig: bool = False,
    disable_sinks_converted: bool = False,
    transmla_scale_dim: str = "",
):
    import math
    import sys
    from contextlib import redirect_stderr, redirect_stdout

    from transformers import AutoTokenizer

    # Import shared PPL logic from cuda-mla/scripts
    sys.path.insert(0, "/root/cuda-mla-scripts")
    from sglang_ppl_eval import _iter_token_blocks, compute_ppl_with_sglang

    run_id = time.strftime("%Y%m%d_%H%M%S")
    remote_log = f"/logs/verify_sglang_gptoss_transmla_{run_id}.log"

    orig_model = os.getenv("ORIG_MODEL_PATH", "/models/openai/gpt-oss-20b")
    converted_model = str(converted_model_path).strip() or os.getenv(
        "CONVERTED_MODEL_PATH", ""
    ).strip()
    if not converted_model:
        converted_model = _pick_default_converted_model("/models") or ""

    dataset_file = os.getenv(
        "DATASET_FILE",
        "/data/radna0__nemotron-math-v2-harmony-tools__high_part00.jsonl",
    )

    seq_len = int(seq_len)
    num_samples = int(num_samples)
    batch_size = int(os.getenv("BATCH_SIZE", "1"))
    tp_size = int(os.getenv("TP_SIZE", "1"))
    orig_attention_backend = os.getenv("ORIG_ATTN_BACKEND", "fa3")
    conv_attention_backend = (
        str(conv_attention_backend).strip()
        or os.getenv("CONV_ATTN_BACKEND", os.getenv("ATTN_BACKEND", "fa3"))
    )
    orig_kv_cache_dtype = os.getenv("ORIG_KV_CACHE_DTYPE", "auto")
    conv_kv_cache_dtype = os.getenv("CONV_KV_CACHE_DTYPE", os.getenv("KV_CACHE_DTYPE", "auto"))
    sglang_dtype = os.getenv("SGLANG_DTYPE", "bfloat16")
    context_length = int(os.getenv("CONTEXT_LENGTH", "0")) or (seq_len + 1)

    transmla_scale_dim = str(transmla_scale_dim).strip()
    if transmla_scale_dim:
        os.environ["SGLANG_GPTOSS_TRANSMLA_SCALE_DIM"] = transmla_scale_dim
    else:
        os.environ.pop("SGLANG_GPTOSS_TRANSMLA_SCALE_DIM", None)

    if bool(force_mla_backend):
        # Force GPT-OSS TransMLA checkpoints (use_transmla=true) to run with SGLang's MLA
        # execution mode (latent KV cache) even when using FA3 as the attention backend.
        os.environ["SGLANG_GPTOSS_TRANSMLA_FORCE_MLA_BACKEND"] = "1"

    with open(remote_log, "a") as f, redirect_stdout(f), redirect_stderr(f):
        print("=" * 80)
        print("SGLang TransMLA Verification (GPT-OSS)")
        print("=" * 80)
        print(f"[ENV] orig_model={orig_model}")
        print(f"[ENV] converted_model={converted_model}")
        print(f"[ENV] dataset_file={dataset_file}")
        print(
            f"[ENV] seq_len={seq_len} num_samples={num_samples} batch_size={batch_size} tp_size={tp_size}"
        )
        print(
            f"[ENV] orig_attention_backend={orig_attention_backend} orig_kv_cache_dtype={orig_kv_cache_dtype}"
        )
        print(
            f"[ENV] conv_attention_backend={conv_attention_backend} conv_kv_cache_dtype={conv_kv_cache_dtype} context_length={context_length}"
        )
        print(f"[ENV] sglang_dtype={sglang_dtype}")
        print(f"[ENV] force_mla_backend={bool(force_mla_backend)}")
        print(f"[ENV] disable_sinks_orig={bool(disable_sinks_orig)}")
        print(f"[ENV] disable_sinks_converted={bool(disable_sinks_converted)}")
        print(f"[ENV] transmla_scale_dim={transmla_scale_dim or '(default)'}")

        # If evaluating a partial (mixed-layer) checkpoint, enable per-layer KV-cache
        # shapes so baseline GPT-OSS (GQA KV) and TransMLA materialized (MHA KV) layers
        # can coexist in the same model.
        try:
            import json as _json

            cfg_path = os.path.join(converted_model, "config.json")
            with open(cfg_path, "r", encoding="utf-8") as _f:
                _cfg = _json.load(_f)
            try:
                import json as _json2

                print(
                    "[CONVERTED_CONFIG] "
                    + _json2.dumps(
                        {
                            "architectures": _cfg.get("architectures"),
                            "use_transmla": _cfg.get("use_transmla", False),
                            "use_transmla_partial": _cfg.get("use_transmla_partial", False),
                            "transmla_layer_limit": _cfg.get("transmla_layer_limit", None),
                            "transmla_layer_ids_len": (
                                len(_cfg.get("transmla_layer_ids") or [])
                                if isinstance(_cfg.get("transmla_layer_ids"), (list, tuple))
                                else None
                            ),
                            "transmla_layer_ids_head": (
                                (_cfg.get("transmla_layer_ids") or [])[:8]
                                if isinstance(_cfg.get("transmla_layer_ids"), (list, tuple))
                                else None
                            ),
                            "qk_nope_head_dim": _cfg.get("qk_nope_head_dim"),
                            "qk_rope_head_dim": _cfg.get("qk_rope_head_dim"),
                            "qk_head_dim": _cfg.get("qk_head_dim"),
                            "v_head_dim": _cfg.get("v_head_dim"),
                            "kv_lora_rank": _cfg.get("kv_lora_rank"),
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                )
            except Exception:
                pass
            _use_transmla = bool(_cfg.get("use_transmla", False))
            _partial = bool(_cfg.get("use_transmla_partial", False)) or (
                bool(_cfg.get("transmla_layer_ids")) and not _use_transmla
            )
            if _partial:
                os.environ["SGLANG_ALLOW_PER_LAYER_KV_SHAPES"] = "1"
                print("[ENV] SGLANG_ALLOW_PER_LAYER_KV_SHAPES=1 (partial checkpoint)")
                # Debug: partial checkpoints are currently producing nonsensical PPL.
                # Enable a small engine/KV-pool probe in sglang_ppl_eval.py.
                os.environ["SGLANG_PPL_DEBUG_ENGINE"] = "1"
                print("[ENV] SGLANG_PPL_DEBUG_ENGINE=1")
        except Exception as e:
            print(f"[ENV] partial-checkpoint detection failed: {e}")

        if not os.path.exists(orig_model):
            raise FileNotFoundError(
                f"Original model path not found: {orig_model}. Did you run predownload_model?"
            )
        if not converted_model or not os.path.exists(converted_model):
            raise FileNotFoundError(
                f"Converted model path not found: {converted_model}. Did you run the conversion?"
            )
        if not os.path.exists(dataset_file):
            raise FileNotFoundError(
                f"Dataset file not found: {dataset_file}. Did you run predownload_dataset?"
            )

        # Build token blocks once (use original tokenizer; converted should be identical).
        # GPT-OSS tokenizers can fail slow->fast conversion (SentencePiece/TikToken mismatch).
        # We only need stable tokenization to build token blocks; force the slow tokenizer.
        tok = AutoTokenizer.from_pretrained(
            orig_model, trust_remote_code=True, use_fast=False
        )
        eos = tok.eos_token_id

        def token_ids_iter():
            import json

            with open(dataset_file, "r", encoding="utf-8") as rf:
                for line in rf:
                    try:
                        ex = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = ex.get("text")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    yield tok.encode(text, add_special_tokens=False)

        token_blocks = _iter_token_blocks(
            token_ids_iter=token_ids_iter(),
            eos_token_id=eos,
            seq_len_plus_one=seq_len + 1,
            num_blocks=num_samples,
        )
        print(f"[DATA] Built {len(token_blocks)} token blocks.")

        print("\n--- PPL: Original (GQA) ---")
        os.environ["SGLANG_GPTOSS_DIAG_DISABLE_SINKS"] = (
            "1" if bool(disable_sinks_orig) else "0"
        )
        ppl_orig = compute_ppl_with_sglang(
            model_path=orig_model,
            token_blocks=token_blocks,
            tp_size=tp_size,
            attention_backend=orig_attention_backend,
            kv_cache_dtype=orig_kv_cache_dtype,
            dtype=sglang_dtype,
            context_length=context_length,
            batch_size=batch_size,
        )
        print(f"[PPL] orig ppl={ppl_orig.ppl:.6f} tokens={ppl_orig.token_count}")

        print("\n--- PPL: Converted (TransMLA / MLA) ---")
        os.environ["SGLANG_GPTOSS_DIAG_DISABLE_SINKS"] = (
            "1" if bool(disable_sinks_converted) else "0"
        )
        ppl_conv = compute_ppl_with_sglang(
            model_path=converted_model,
            token_blocks=token_blocks,
            tp_size=tp_size,
            attention_backend=conv_attention_backend,
            kv_cache_dtype=conv_kv_cache_dtype,
            dtype=sglang_dtype,
            context_length=context_length,
            batch_size=batch_size,
        )
        print(f"[PPL] conv ppl={ppl_conv.ppl:.6f} tokens={ppl_conv.token_count}")

        if math.isfinite(ppl_orig.ppl) and math.isfinite(ppl_conv.ppl):
            delta = (ppl_conv.ppl - ppl_orig.ppl) / max(ppl_orig.ppl, 1e-9)
            print(f"[PPL] delta={(delta * 100):.2f}%")

        print("\n--- Generation smoke test (converted) ---")
        import sglang as sgl

        engine = sgl.Engine(
            model_path=converted_model,
            tp_size=tp_size,
            attention_backend=conv_attention_backend,
            kv_cache_dtype=conv_kv_cache_dtype,
            dtype=sglang_dtype,
            context_length=min(context_length, 4096),
            max_running_requests=1,
            max_total_tokens=min(8192, int(min(context_length, 4096) * 4)),
            disable_cuda_graph=True,
            allow_auto_truncate=True,
        )
        out = engine.generate(
            prompt="Write a short proof that 1+1=2.",
            sampling_params={"temperature": 0, "max_new_tokens": 128},
        )
        print(out.get("text", out))
        engine.shutdown()

        log_vol.commit()
        print(f"[OK] Remote log saved: {remote_log}")

    # Minimal summary printed to the local `modal run` output.
    return {
        "orig_model": orig_model,
        "converted_model": converted_model,
        "ppl_orig": ppl_orig.ppl,
        "ppl_converted": ppl_conv.ppl,
        "remote_log": remote_log,
    }


@app.local_entrypoint()
def main(
    converted_model_path: str = "",
    seq_len: int = 1024,
    num_samples: int = 64,
    conv_attention_backend: str = "fa3",
    force_mla_backend: bool = False,
    disable_sinks_orig: bool = False,
    disable_sinks_converted: bool = False,
    transmla_scale_dim: str = "",
):
    result = verify.remote(
        converted_model_path=converted_model_path,
        seq_len=seq_len,
        num_samples=num_samples,
        conv_attention_backend=conv_attention_backend,
        force_mla_backend=force_mla_backend,
        disable_sinks_orig=disable_sinks_orig,
        disable_sinks_converted=disable_sinks_converted,
        transmla_scale_dim=transmla_scale_dim,
    )
    print("[RESULT]", result)
