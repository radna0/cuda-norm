from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def _set_env_sane_defaults() -> None:
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")


def _send_generate(
    base_url: str,
    prompt: str,
    *,
    max_new_tokens: int,
    timeout_s: int,
    temperature: float = 0.0,
) -> dict:
    sampling_params: dict = {
        "temperature": float(temperature),
        "top_p": 1.0,
        "top_k": 1,
        "max_new_tokens": int(max_new_tokens),
        # Make this a decode benchmark, not an EOS/stopping benchmark.
        "min_new_tokens": int(max_new_tokens),
        "ignore_eos": True,
    }
    resp = requests.post(
        base_url + "/generate",
        json={"text": prompt, "sampling_params": sampling_params},
        timeout=int(timeout_s),
    )
    resp.raise_for_status()
    return resp.json()


def _run_batch(
    base_url: str,
    *,
    prompt: str,
    max_new_tokens: int,
    concurrency: int,
    timeout_s: int,
) -> dict:
    start = time.perf_counter()
    total_tokens = 0

    with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
        futures = [
            pool.submit(
                _send_generate,
                base_url,
                prompt,
                max_new_tokens=max_new_tokens,
                timeout_s=timeout_s,
            )
            for _ in range(int(concurrency))
        ]
        for fut in as_completed(futures):
            out = fut.result()
            meta = out.get("meta_info", {}) or {}
            total_tokens += int(meta.get("completion_tokens", 0))

    dt = time.perf_counter() - start
    tok_s_total = float(total_tokens) / max(dt, 1e-9)
    tok_s_per_stream = tok_s_total / max(float(concurrency), 1e-9)
    return {
        "wall_s": float(dt),
        "output_tokens": int(total_tokens),
        "tok_s_total": float(tok_s_total),
        "tok_s_per_stream": float(tok_s_per_stream),
    }


def _launch_server(cmd: list[str], *, timeout_s: int = 600) -> tuple[object, str]:
    import subprocess

    proc = subprocess.Popen(cmd)
    base_url = None
    for _ in range(int(timeout_s)):
        time.sleep(1)
        try:
            port = None
            for i, part in enumerate(cmd):
                if part == "--port" and i + 1 < len(cmd):
                    port = cmd[i + 1]
                    break
                if part.startswith("--port="):
                    port = part.split("=", 1)[1]
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


def _kill(proc: object) -> None:
    import signal

    try:
        proc.send_signal(signal.SIGTERM)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def _parse_int_csv(s: str) -> list[int]:
    out: list[int] = []
    for part in (s or "").split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def main() -> None:
    _set_env_sane_defaults()

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--served-model-name", default="")
    ap.add_argument("--attention-backend", default="fa3", choices=["fa3", "flashinfer", "trtllm"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    ap.add_argument("--timeout-s", type=int, default=3600)
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    prompt = "A" * int(args.prompt_len)
    # Add a small instruction wrapper to avoid tokenizer edge weirdness.
    prompt = f"You are a helpful assistant.\n\nUser: {prompt}\nAssistant:"

    batch_sizes = _parse_int_csv(args.batch_sizes)
    if not batch_sizes:
        raise SystemExit("empty --batch-sizes")

    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(args.model_path),
        "--served-model-name",
        str(args.served_model_name or str(args.model_path)),
        "--dtype",
        str(args.dtype),
        "--attention-backend",
        str(args.attention_backend),
        "--port",
        str(int(args.port)),
        "--host",
        "127.0.0.1",
        "--max-running-requests",
        str(max(64, max(batch_sizes))),
    ]

    proc, base_url = _launch_server(cmd)
    try:
        if bool(args.warmup):
            _run_batch(
                base_url,
                prompt=prompt,
                max_new_tokens=min(64, int(args.max_new_tokens)),
                concurrency=1,
                timeout_s=int(args.timeout_s),
            )

        results: list[dict] = []
        for bs in batch_sizes:
            m = _run_batch(
                base_url,
                prompt=prompt,
                max_new_tokens=int(args.max_new_tokens),
                concurrency=int(bs),
                timeout_s=int(args.timeout_s),
            )
            results.append({"batch_size": int(bs), **m})

        out = {
            "model_path": str(args.model_path),
            "served_model_name": str(args.served_model_name or ""),
            "attention_backend": str(args.attention_backend),
            "dtype": str(args.dtype),
            "prompt_len": int(args.prompt_len),
            "max_new_tokens": int(args.max_new_tokens),
            "batch_sizes": batch_sizes,
            "results": results,
        }
        print(json.dumps(out, indent=2, sort_keys=True), flush=True)

        if args.out_json:
            p = Path(args.out_json)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    finally:
        _kill(proc)


if __name__ == "__main__":
    main()

