from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def _set_env_sane_defaults() -> None:
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")


def _send_generate(base_url: str, prompt: str, *, max_new_tokens: int, timeout_s: int) -> dict:
    sampling_params: dict = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_new_tokens": int(max_new_tokens),
    }
    resp = requests.post(
        base_url + "/generate",
        json={"text": prompt, "sampling_params": sampling_params},
        timeout=int(timeout_s),
    )
    resp.raise_for_status()
    return resp.json()


def _run_requests(
    base_url: str,
    *,
    prompts: list[str],
    max_new_tokens: int,
    concurrency: int,
    timeout_s: int,
    expect_dflash: bool,
) -> dict:
    start = time.perf_counter()
    total_tokens = 0
    verify_ct = 0
    accept_lengths: list[float] = []

    with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
        futures = {
            pool.submit(
                _send_generate,
                base_url,
                p,
                max_new_tokens=max_new_tokens,
                timeout_s=timeout_s,
            ): i
            for i, p in enumerate(prompts)
        }
        for fut in as_completed(futures):
            out = fut.result()
            meta = out.get("meta_info", {}) or {}
            total_tokens += int(meta.get("completion_tokens", 0))
            verify_ct += int(meta.get("spec_verify_ct", 0))
            if "spec_accept_length" in meta:
                try:
                    accept_lengths.append(float(meta["spec_accept_length"]))
                except (TypeError, ValueError):
                    pass

    latency = time.perf_counter() - start
    toks_per_s = total_tokens / max(latency, 1e-6)
    if expect_dflash and verify_ct <= 0:
        raise RuntimeError("DFLASH sanity check failed: missing spec_verify_ct in responses")

    return {
        "latency_s": float(latency),
        "output_tokens": int(total_tokens),
        "output_toks_per_s": float(toks_per_s),
        "spec_verify_ct_sum": int(verify_ct),
        "spec_accept_length_mean": (sum(accept_lengths) / len(accept_lengths)) if accept_lengths else None,
    }


def _launch_server(cmd: list[str], *, timeout_s: int = 600):
    import subprocess

    proc = subprocess.Popen(cmd)
    # Wait for /get_model_info to be ready.
    base_url = None
    for _ in range(int(timeout_s)):
        time.sleep(1)
        try:
            port = None
            for part in cmd:
                if part.startswith("--port"):
                    # "--port", "30000" or "--port=30000"
                    if part == "--port":
                        continue
                    _, port = part.split("=", 1)
            if port is None:
                # look for separate token
                for i, part in enumerate(cmd):
                    if part == "--port" and i + 1 < len(cmd):
                        port = cmd[i + 1]
                        break
            if port is None:
                raise RuntimeError("missing --port in cmd")
            base_url = f"http://127.0.0.1:{int(port)}"
            r = requests.get(base_url + "/get_model_info", timeout=5)
            if r.status_code == 200:
                return proc, base_url
        except Exception:
            continue
    proc.terminate()
    raise RuntimeError("Server failed to become ready")


def _kill(proc) -> None:
    import signal

    try:
        proc.send_signal(signal.SIGTERM)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def main() -> None:
    _set_env_sane_defaults()

    ap = argparse.ArgumentParser()
    ap.add_argument("--target-model", default="openai/gpt-oss-20b")
    ap.add_argument("--draft-model", required=True, help="SGLang-loadable DFlashDraftModel dir (converted).")
    ap.add_argument("--attention-backend", default="fa3", choices=["fa3", "flashinfer", "trtllm"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--timeout-s", type=int, default=3600)
    ap.add_argument("--port-base", type=int, default=30000)
    args = ap.parse_args()

    prompts = [
        "You are a helpful assistant.\n\nUser: Solve: If 196 = 2^2 * 7^2, how many divisors does it have?\nAssistant:",
        "You are a helpful assistant.\n\nUser: Write a Python function that computes gcd(a,b).\nAssistant:",
        "You are a helpful assistant.\n\nUser: Use a tool to list files in the current directory.\nAssistant:",
        "You are a helpful assistant.\n\nUser: Explain why the sky is blue in two sentences.\nAssistant:",
    ]
    prompts = (prompts * ((int(args.num_prompts) + len(prompts) - 1) // len(prompts)))[: int(args.num_prompts)]

    base_port = int(args.port_base)
    base_cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(args.target_model),
        "--dtype",
        str(args.dtype),
        "--attention-backend",
        str(args.attention_backend),
        "--port",
        str(base_port),
        "--host",
        "127.0.0.1",
        "--max-running-requests",
        str(max(8, int(args.concurrency))),
    ]
    dflash_cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(args.target_model),
        "--dtype",
        str(args.dtype),
        "--attention-backend",
        str(args.attention_backend),
        "--port",
        str(base_port + 1),
        "--host",
        "127.0.0.1",
        "--max-running-requests",
        str(max(8, int(args.concurrency))),
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-draft-model-path",
        str(args.draft_model),
        "--speculative-dflash-block-size",
        str(int(args.block_size)),
    ]

    base_proc, base_url = _launch_server(base_cmd)
    try:
        base_metrics = _run_requests(
            base_url,
            prompts=prompts,
            max_new_tokens=int(args.max_new_tokens),
            concurrency=int(args.concurrency),
            timeout_s=int(args.timeout_s),
            expect_dflash=False,
        )
    finally:
        _kill(base_proc)

    dflash_proc, dflash_url = _launch_server(dflash_cmd)
    try:
        dflash_metrics = _run_requests(
            dflash_url,
            prompts=prompts,
            max_new_tokens=int(args.max_new_tokens),
            concurrency=int(args.concurrency),
            timeout_s=int(args.timeout_s),
            expect_dflash=True,
        )
    finally:
        _kill(dflash_proc)

    print({"baseline": base_metrics, "dflash": dflash_metrics}, flush=True)


if __name__ == "__main__":
    main()

