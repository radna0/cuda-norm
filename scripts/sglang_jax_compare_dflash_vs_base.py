#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CompareResult:
    total: int
    matched: int
    mismatched: int


def _build_random_inputs(*, num_prompts: int, prompt_len: int, vocab_size: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    inputs: list[list[int]] = []
    for _ in range(int(num_prompts)):
        inputs.append([rng.randrange(1, int(vocab_size)) for _ in range(int(prompt_len))])
    return inputs


def _run_engine(
    *,
    model_path: str,
    tokenizer_path: str,
    input_ids: list[list[int]],
    max_new_tokens: int,
    ignore_eos: bool,
    speculative_algorithm: str,
    draft_model_path: str | None,
    block_size: int,
    dtype: str,
    page_size: int,
    mem_fraction_static: float,
    context_length: int,
    max_total_tokens: int | None,
) -> list[list[int]]:
    from sgl_jax.srt.entrypoints.engine import Engine

    engine_kwargs: dict = dict(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        dtype=dtype,
        page_size=int(page_size),
        mem_fraction_static=float(mem_fraction_static),
        context_length=int(context_length),
        disable_overlap_schedule=True,
        skip_server_warmup=True,
        disable_precompile=True,
        log_level="error",
    )
    if max_total_tokens is not None:
        engine_kwargs["max_total_tokens"] = int(max_total_tokens)

    if speculative_algorithm.lower() != "none":
        engine_kwargs["speculative_algorithm"] = speculative_algorithm
        engine_kwargs["speculative_num_draft_tokens"] = int(block_size)
        engine_kwargs["speculative_dflash_block_size"] = int(block_size)
        # sglang-jax DFLASH currently requires radix cache disabled.
        engine_kwargs["disable_radix_cache"] = True
        if draft_model_path is not None:
            engine_kwargs["speculative_draft_model_path"] = draft_model_path

    engine = None
    try:
        engine = Engine(**engine_kwargs)
        sampling_params = [
            {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "max_new_tokens": int(max_new_tokens),
                "ignore_eos": bool(ignore_eos),
            }
            for _ in range(len(input_ids))
        ]
        out = engine.generate(input_ids=input_ids, sampling_params=sampling_params, stream=False)
        if isinstance(out, dict):
            output_ids = out.get("output_ids")
            if output_ids is None:
                raise RuntimeError(f"Engine output missing output_ids keys: {sorted(out.keys())}")
            if not isinstance(output_ids, list):
                raise RuntimeError(f"Unexpected output_ids type: {type(output_ids)}")
            return output_ids
        if isinstance(out, list):
            outputs: list[list[int]] = []
            for item in out:
                if not isinstance(item, dict) or "output_ids" not in item:
                    raise RuntimeError(f"Unexpected engine item: {type(item)} keys={getattr(item,'keys',lambda:[])()}")
                outputs.append(item["output_ids"])
            return outputs
        raise RuntimeError(f"Unexpected engine output type: {type(out)}")
    finally:
        if engine is not None:
            engine.shutdown()


def _compare_lists(a: list[int], b: list[int]) -> tuple[bool, int]:
    n = min(len(a), len(b))
    for i in range(n):
        if int(a[i]) != int(b[i]):
            return False, i
    if len(a) != len(b):
        return False, n
    return True, -1


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare baseline greedy vs DFLASH greedy outputs (sglang-jax, TPU).")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--draft-model-path", default="")
    ap.add_argument("--num-prompts", type=int, default=2)
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--vocab-size", type=int, default=201088)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--page-size", type=int, default=1)
    ap.add_argument("--mem-fraction-static", type=float, default=0.45)
    ap.add_argument("--context-length", type=int, default=2048)
    ap.add_argument("--max-total-tokens", type=int, default=65536)
    ap.add_argument("--ignore-eos", action="store_true")
    args = ap.parse_args()
    print_prefix = 16

    model_path = os.path.abspath(args.model_path)
    draft_path = os.path.abspath(args.draft_model_path) if args.draft_model_path else ""

    inputs = _build_random_inputs(
        num_prompts=args.num_prompts,
        prompt_len=args.prompt_len,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )

    baseline = _run_engine(
        model_path=model_path,
        tokenizer_path=model_path,
        input_ids=inputs,
        max_new_tokens=args.max_new_tokens,
        ignore_eos=args.ignore_eos,
        speculative_algorithm="NONE",
        draft_model_path=None,
        block_size=args.block_size,
        dtype=args.dtype,
        page_size=args.page_size,
        mem_fraction_static=args.mem_fraction_static,
        context_length=args.context_length,
        max_total_tokens=args.max_total_tokens,
    )

    if not draft_path:
        raise SystemExit("--draft-model-path is required for DFLASH compare")

    dflash = _run_engine(
        model_path=model_path,
        tokenizer_path=model_path,
        input_ids=inputs,
        max_new_tokens=args.max_new_tokens,
        ignore_eos=args.ignore_eos,
        speculative_algorithm="DFLASH",
        draft_model_path=draft_path,
        block_size=args.block_size,
        dtype=args.dtype,
        page_size=args.page_size,
        mem_fraction_static=args.mem_fraction_static,
        context_length=args.context_length,
        max_total_tokens=args.max_total_tokens,
    )

    matched = 0
    for i, (base_ids, dflash_ids) in enumerate(zip(baseline, dflash, strict=True)):
        ok, idx = _compare_lists(base_ids, dflash_ids)
        if ok:
            matched += 1
        else:
            print(f"[mismatch] prompt={i} first_mismatch_at={idx} base_len={len(base_ids)} dflash_len={len(dflash_ids)}")
            if idx >= 0:
                print(f"  base[{idx}]={base_ids[idx] if idx < len(base_ids) else None}")
                print(f"  dflash[{idx}]={dflash_ids[idx] if idx < len(dflash_ids) else None}")
            print(f"  base[:{print_prefix}]={base_ids[:print_prefix]}")
            print(f"  dflash[:{print_prefix}]={dflash_ids[:print_prefix]}")

    result = CompareResult(total=len(baseline), matched=matched, mismatched=len(baseline) - matched)
    print(
        f"[done] total={result.total} matched={result.matched} mismatched={result.mismatched} "
        f"match_rate={result.matched / max(1, result.total):.3f}"
    )


if __name__ == "__main__":
    main()
