"""
Structural-pruned GPT-OSS-20B smoke test (Modal, H100).

Deliverable (manager Step C):
- `reports/20b_structural_prune_smoke.md`
  - load test
  - 5 prompt generation sanity
  - router stats summary (experts activated)

Assumes the pruned models were built into the pruning model volume at:
  /root/model/artifacts/20b_pruned_models/general_50pct_experts
  /root/model/artifacts/20b_pruned_models/math_25pct_experts

Run (always log to unsloth_logs/):
  mkdir -p unsloth_logs
  ts=$(date +%Y%m%d_%H%M%S)
  nohup modal run modal/structural_prune_smoke_20b.py > "unsloth_logs/20b_pruned_smoke_${ts}.log" 2>&1 &
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "gpt-oss-20b-pruned-smoke"


def _maybe_load_repo_dotenv() -> None:
    try:
        dotenv_path = Path(__file__).resolve().parent.parent / ".env"
        if not dotenv_path.exists():
            return
        for raw in dotenv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
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
    except Exception:
        return


_maybe_load_repo_dotenv()

_secrets: list[modal.Secret] = []
if os.environ.get("HF_TOKEN"):
    _secrets.append(modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]}))

model_volume = modal.Volume.from_name("pruning-models", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("hf-cache-persistent", create_if_missing=True)

BASE_IMAGE = "nvidia/cuda:12.8.0-devel-ubuntu24.04"
image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.12")
    .apt_install("git", "python3-dev", "build-essential", "curl")
    .run_commands("python -m pip install -U pip setuptools wheel")
    .run_commands(
        "python -m pip install "
        "torch==2.9.0 "
        "--extra-index-url https://download.pytorch.org/whl/cu128"
    )
    .run_commands(
        "python -m pip install "
        "numpy==2.2.0 accelerate==1.10.1 "
        "transformers==4.56.2 tokenizers safetensors "
        "kernels==0.11.7 "
        "hf_transfer huggingface-hub==0.34.0"
    )
)

app = modal.App(APP_NAME)


def _ensure_hf_env() -> None:
    os.environ.setdefault("HF_HOME", "/root/hf_cache")
    os.environ.setdefault("XDG_CACHE_HOME", "/root/hf_cache/.cache")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for p in ("/root/hf_cache", "/root/hf_cache/.cache", "/root/model"):
        try:
            Path(p).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


def _iter_gpt_oss_layers(model: Any) -> list[Any]:
    base = getattr(model, "model", None)
    if base is None:
        raise RuntimeError("Expected model.model to exist.")
    layers = getattr(base, "layers", None)
    if layers is None:
        raise RuntimeError("Expected model.model.layers to exist.")
    return list(layers)


def _as_probs(scores, *, torch_mod):
    s = scores.float()
    try:
        row_sum = float(s.sum(dim=-1).mean().detach().cpu().item())
        s_min = float(s.min().detach().cpu().item())
        s_max = float(s.max().detach().cpu().item())
        if (s_min >= -1e-3) and (s_max <= 1.0 + 1e-3) and (0.98 <= row_sum <= 1.02):
            return s
    except Exception:
        pass
    return torch_mod.softmax(s, dim=-1)


@app.function(
    image=image,
    gpu="H100:1",
    timeout=21600,
    cpu=16.0,
    memory=262144,
    volumes={
        "/root/model": model_volume,
        "/root/hf_cache": hf_cache_volume,
    },
    secrets=_secrets,
)
def smoke_pruned_model(model_dir: str, prompts: list[str]) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _ensure_hf_env()
    try:
        model_volume.reload()
        hf_cache_volume.reload()
    except Exception:
        pass

    mp = Path(model_dir)
    if not mp.exists():
        raise RuntimeError(f"model_dir not found: {model_dir}")

    tok = AutoTokenizer.from_pretrained(str(mp), trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        str(mp),
        torch_dtype="auto",
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()

    dtype = str(next(model.parameters()).dtype)
    num_experts = int(getattr(model.config, "num_local_experts", 0) or 0)
    cfg_top_k = int(
        getattr(model.config, "num_experts_per_tok", 0) or getattr(model.config, "experts_per_token", 4)
    )
    cfg_top_k_ok = bool(1 <= cfg_top_k <= max(1, num_experts))

    def _render_harmony_prompt(user_text: str) -> str:
        # Match GPT-OSS Harmony formatting used in our datasets:
        # start a new assistant final message so `generate()` continues it.
        return (
            "<|start|>user<|message|>"
            + user_text
            + "<|end|><|start|>assistant<|channel|>final<|message|>"
        )

    # Generation sanity (5 prompts).
    gens: list[dict[str, str]] = []
    for p in prompts:
        prompt_text = _render_harmony_prompt(p)
        enc = tok(prompt_text, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=96,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        full = tok.decode(out[0], skip_special_tokens=False)
        # Show only the newly generated portion (best effort).
        gen = full[len(prompt_text) :] if full.startswith(prompt_text) else full
        gens.append({"prompt": p, "gen": gen[:1200]})

    # Router stats summary: call routers manually in mlp pre-hooks, like our working
    # profiling path (Transformers doesn't always surface router logits).
    layers = _iter_gpt_oss_layers(model)
    num_layers = len(layers)
    hist = torch.zeros((num_layers, num_experts), dtype=torch.int64, device="cpu")
    prob_sum = torch.zeros((num_layers,), dtype=torch.float64, device="cpu")
    prob_count = torch.zeros((num_layers,), dtype=torch.int64, device="cpu")

    hooks = []

    def _make_hook(li: int):
        router = layers[li].mlp.router

        def _hook(_module, inputs):
            if not inputs:
                return
            hidden = inputs[0]
            if not torch.is_tensor(hidden):
                return
            hs = hidden
            if hs.dim() == 3:
                hs2 = hs.reshape(-1, int(hs.shape[-1]))
            elif hs.dim() == 2:
                hs2 = hs
            else:
                return

            out = router(hs2)
            if not isinstance(out, (tuple, list)) or len(out) != 2:
                return
            scores, idx = out
            if not torch.is_tensor(scores) or not torch.is_tensor(idx):
                return
            if idx.numel() == 0:
                return
            if scores.dim() == 1:
                scores2 = scores.unsqueeze(0)
            elif scores.dim() == 2:
                scores2 = scores
            else:
                return
            if int(scores2.shape[-1]) != int(num_experts):
                return
            if idx.dim() == 1:
                idx2 = idx.unsqueeze(0)
            elif idx.dim() == 2:
                idx2 = idx
            else:
                return
            k = int(idx2.shape[1])
            if k <= 0:
                return
            probs = _as_probs(scores2, torch_mod=torch)
            kk = max(1, min(int(k), 8))
            idxk = idx2[:, :kk].to(torch.int64)
            flat = idxk.reshape(-1)
            h = torch.bincount(flat, minlength=num_experts).to("cpu")
            hist[li] += h
            sel = probs.gather(1, idxk).reshape(-1)
            sel = torch.clamp(sel, 0.0, 1.0)
            if sel.numel():
                prob_sum[li] += float(sel.sum().item())
                prob_count[li] += int(sel.numel())

        return _hook

    for li in range(num_layers):
        hooks.append(layers[li].mlp.register_forward_pre_hook(_make_hook(li)))

    t0 = time.time()
    total_tokens = 0
    try:
        for p in prompts:
            prompt_text = _render_harmony_prompt(p)
            enc = tok(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
            enc = {k: v.to("cuda") for k, v in enc.items()}
            with torch.inference_mode():
                _ = model(**enc, use_cache=False, return_dict=False)
            total_tokens += int(enc["input_ids"].numel())
    finally:
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

    dt = max(1e-9, time.time() - t0)
    toks_s = float(total_tokens / dt)

    top_by_layer: list[list[int]] = []
    mean_selected_prob: list[float] = []
    for li in range(num_layers):
        counts = hist[li].tolist()
        order = sorted(range(num_experts), key=lambda e: counts[e], reverse=True)
        top_by_layer.append(order[:8])
        mean_selected_prob.append(float(prob_sum[li].item()) / max(1.0, float(prob_count[li].item())))

    return {
        "model_dir": str(model_dir),
        "dtype": dtype,
        "num_layers": int(num_layers),
        "num_experts": int(num_experts),
        "cfg_top_k": int(cfg_top_k),
        "cfg_top_k_ok": bool(cfg_top_k_ok),
        "router_tokens_per_s": float(toks_s),
        "top_experts_by_layer": top_by_layer,
        "mean_selected_prob_by_layer": mean_selected_prob,
        "generations": gens,
    }


@app.local_entrypoint()
def main():
    prompts = [
        "Solve: 17*23. Show work.",
        "Explain what Mixture-of-Experts routing is in one paragraph.",
        "Write a Python function to compute Fibonacci numbers iteratively.",
        "Use a tool to get weather in SF. (Just describe the tool call you would make.)",
        "Prove that the sum of two even numbers is even.",
    ]

    general_dir = "/root/model/artifacts/20b_pruned_models/general_50pct_experts"
    math_dir = "/root/model/artifacts/20b_pruned_models/math_25pct_experts"

    general = smoke_pruned_model.remote(model_dir=general_dir, prompts=prompts)
    math = smoke_pruned_model.remote(model_dir=math_dir, prompts=prompts)

    out = Path("reports/20b_structural_prune_smoke.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(
            [
                "# 20B structural prune smoke",
                "",
                "## Variants",
                "",
                f"- general_50pct_experts: `{general_dir}`",
                f"- math_25pct_experts: `{math_dir}`",
                "",
                "## Load + config checks",
                "",
                f"- general: dtype={general['dtype']} experts={general['num_experts']} top_k={general['cfg_top_k']} ok={general['cfg_top_k_ok']}",
                f"- math: dtype={math['dtype']} experts={math['num_experts']} top_k={math['cfg_top_k']} ok={math['cfg_top_k_ok']}",
                "",
                "## Router stats (summary)",
                "",
                f"- general router tokens/s (forward-only, 5 prompts): {general['router_tokens_per_s']:.0f}",
                f"- math router tokens/s (forward-only, 5 prompts): {math['router_tokens_per_s']:.0f}",
                "",
                "Top experts (layer 0):",
                "",
                f"- general: {general['top_experts_by_layer'][0]}",
                f"- math: {math['top_experts_by_layer'][0]}",
                "",
                "Mean selected prob (layer 0):",
                "",
                f"- general: {general['mean_selected_prob_by_layer'][0]:.4f}",
                f"- math: {math['mean_selected_prob_by_layer'][0]:.4f}",
                "",
                "## Generation (first 5 prompts)",
                "",
                "### general_50pct_experts",
                "",
            ]
            + [
                f"**Prompt:** {g['prompt']}\n\n```text\n{g['gen']}\n```\n"
                for g in general["generations"]
            ]
            + [
                "",
                "### math_25pct_experts",
                "",
            ]
            + [
                f"**Prompt:** {g['prompt']}\n\n```text\n{g['gen']}\n```\n"
                for g in math["generations"]
            ]
            + [
                "",
                "## Reproduce",
                "",
                "```bash",
                "modal run modal/structural_prune_smoke_20b.py",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[+] Wrote {out}")
    print("[RESULT] general", {k: general[k] for k in ('dtype','num_experts','cfg_top_k','cfg_top_k_ok','router_tokens_per_s')})
    print("[RESULT] math", {k: math[k] for k in ('dtype','num_experts','cfg_top_k','cfg_top_k_ok','router_tokens_per_s')})
