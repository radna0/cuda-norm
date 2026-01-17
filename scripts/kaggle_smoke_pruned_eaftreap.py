from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_variants() -> dict[str, str]:
    manifest_keepfrac = Path("artifacts/20b_pruned_models_eaftreap_keepfrac/manifest_eaftreap_keepfrac.json")
    if manifest_keepfrac.exists():
        data = json.loads(manifest_keepfrac.read_text(encoding="utf-8"))
        variants = data.get("variants") or {}
        if isinstance(variants, dict) and variants:
            return {str(k): str(v) for k, v in variants.items()}
    manifest = Path("artifacts/20b_pruned_models_eaftreap/manifest_eaftreap.json")
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        variants = data.get("variants") or {}
        if isinstance(variants, dict) and variants:
            return {str(k): str(v) for k, v in variants.items()}
    return {
        "general_50pct_experts_eaftreap": "/tmp/harmony_pruning_cache/model/artifacts/20b_pruned_models_eaftreap/general_50pct_experts_eaftreap",
        "math_25pct_experts_eaftreap": "/tmp/harmony_pruning_cache/model/artifacts/20b_pruned_models_eaftreap/math_25pct_experts_eaftreap",
    }


def main() -> None:
    variants = _load_variants()
    prompts = [
        "<|start|>user<|message|>Solve: If f(x)=x^2+1, compute f(10).<|end|>",
        "<|start|>user<|message|>Write a bash one-liner to count lines in all .py files recursively.<|end|>",
        "<|start|>user<|message|>Explain what a mutex is in 2 sentences.<|end|>",
        "<|start|>user<|message|>Plan a 3-step tool-using approach to scrape a webpage then summarize it (no code).<|end|>",
        "<|start|>user<|message|>What is the derivative of sin(x)?<|end|>",
    ]

    for name, model_dir in variants.items():
        p = Path(model_dir)
        if not (p / "config.json").exists():
            raise SystemExit(f"Missing model at {p}")

        tok = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(str(p), torch_dtype=torch.bfloat16, device_map="cuda")
        model.eval()

        cfg = model.config
        print(f"\n=== {name} ===", flush=True)
        print(
            "num_local_experts=",
            getattr(cfg, "num_local_experts", None),
            "top_k=",
            getattr(cfg, "num_experts_per_tok", None),
            flush=True,
        )

        with torch.no_grad():
            for i, pr in enumerate(prompts):
                ids = tok(pr, return_tensors="pt").to("cuda")
                out = model.generate(**ids, max_new_tokens=64, do_sample=False)
                text = tok.decode(out[0], skip_special_tokens=False)
                print(f"--- prompt {i} ---", flush=True)
                print(text[-400:], flush=True)


if __name__ == "__main__":
    main()
