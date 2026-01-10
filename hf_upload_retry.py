#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class RetryConfig:
    min_backoff_s: int
    max_backoff_s: int
    backoff_multiplier: float
    jitter_ratio: float
    rate_limit_window_s: int
    rate_limit_threshold: int


def _build_cmd(
    cli: str,
    repo_id: str,
    local_path: str,
    repo_type: str,
    revision: str | None,
    token: str | None,
    num_workers: int,
    no_report: bool,
    no_bars: bool,
) -> list[str]:
    cmd = [
        cli,
        "upload-large-folder",
        repo_id,
        local_path,
        "--repo-type",
        repo_type,
        "--num-workers",
        str(num_workers),
    ]
    if revision:
        cmd.extend(["--revision", revision])
    if token:
        cmd.extend(["--token", token])
    if no_report:
        cmd.append("--no-report")
    if no_bars:
        cmd.append("--no-bars")
    return cmd


def _is_rate_limit_line(line: str) -> bool:
    lower = line.lower()
    return (
        " 429 " in f" {lower} "
        or "429 client error" in lower
        or "too many requests" in lower
        or "rate limit" in lower
        or "rate-lim" in lower
    )


def _sleep_with_jitter(seconds: float, jitter_ratio: float) -> None:
    seconds = max(0.0, seconds)
    if jitter_ratio <= 0:
        time.sleep(seconds)
        return
    jitter = seconds * jitter_ratio
    time.sleep(max(0.0, seconds + random.uniform(-jitter, jitter)))


def _run_with_streaming_output(cmd: list[str], retry: RetryConfig) -> tuple[int, bool, bool]:
    print(f"[{_now()}] starting: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    saw_rate_limit = False
    rate_limit_burst_triggered = False
    rate_limit_times: deque[float] = deque()

    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if _is_rate_limit_line(line):
                saw_rate_limit = True
                now = time.time()
                rate_limit_times.append(now)
                while rate_limit_times and now - rate_limit_times[0] > retry.rate_limit_window_s:
                    rate_limit_times.popleft()
                if len(rate_limit_times) >= retry.rate_limit_threshold:
                    rate_limit_burst_triggered = True
                    print(
                        f"[{_now()}] detected {len(rate_limit_times)} rate-limit messages in "
                        f"{retry.rate_limit_window_s}s; terminating to back off.",
                        flush=True,
                    )
                    proc.terminate()
                    break
    except KeyboardInterrupt:
        print(f"[{_now()}] interrupted; terminating child process...", flush=True)
        proc.terminate()
        raise
    finally:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)

    return proc.returncode, saw_rate_limit, rate_limit_burst_triggered


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Retry/resume Hugging Face `upload-large-folder` with rate-limit aware backoff.\n"
            "Designed to run under `nohup` and keep going until the upload completes."
        )
    )
    parser.add_argument("repo_id", help="Hub repo id, e.g. radna0/harmony-nemotron-cpu-artifacts")
    parser.add_argument("local_path", help="Local folder to upload")
    parser.add_argument("--cli", default="hf", choices=["hf", "huggingface-cli"])
    parser.add_argument("--repo-type", default="dataset", choices=["model", "dataset", "space"])
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None, help="Optional HF token (otherwise uses local login).")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--initial-sleep-s", type=int, default=0)
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--no-bars", action="store_true")

    parser.add_argument("--min-backoff-s", type=int, default=300)
    parser.add_argument("--max-backoff-s", type=int, default=3600)
    parser.add_argument("--backoff-multiplier", type=float, default=2.0)
    parser.add_argument("--jitter-ratio", type=float, default=0.10)
    parser.add_argument("--rate-limit-window-s", type=int, default=60)
    parser.add_argument("--rate-limit-threshold", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=0, help="0 means infinite.")

    args = parser.parse_args()

    if args.num_workers < 1:
        raise SystemExit("--num-workers must be >= 1")

    local_path = os.path.abspath(args.local_path)
    if not os.path.isdir(local_path):
        raise SystemExit(f"local_path is not a directory: {local_path}")

    retry = RetryConfig(
        min_backoff_s=args.min_backoff_s,
        max_backoff_s=args.max_backoff_s,
        backoff_multiplier=args.backoff_multiplier,
        jitter_ratio=args.jitter_ratio,
        rate_limit_window_s=args.rate_limit_window_s,
        rate_limit_threshold=args.rate_limit_threshold,
    )

    if args.initial_sleep_s > 0:
        print(f"[{_now()}] initial sleep: {args.initial_sleep_s}s", flush=True)
        time.sleep(args.initial_sleep_s)

    attempt = 0
    backoff_s = float(retry.min_backoff_s)
    while True:
        attempt += 1
        if args.max_attempts and attempt > args.max_attempts:
            print(f"[{_now()}] reached max attempts ({args.max_attempts}); exiting.", flush=True)
            return 2

        cmd = _build_cmd(
            cli=args.cli,
            repo_id=args.repo_id,
            local_path=local_path,
            repo_type=args.repo_type,
            revision=args.revision,
            token=args.token,
            num_workers=args.num_workers,
            no_report=args.no_report,
            no_bars=args.no_bars,
        )

        print(f"[{_now()}] attempt {attempt} (num_workers={args.num_workers})", flush=True)
        rc, saw_rate_limit, burst = _run_with_streaming_output(cmd, retry)
        if rc == 0:
            print(f"[{_now()}] upload completed successfully.", flush=True)
            return 0

        reason = "rate-limited" if saw_rate_limit else "error"
        if burst:
            reason = "rate-limited (burst)"
        print(f"[{_now()}] upload attempt failed (rc={rc}; {reason}).", flush=True)

        if saw_rate_limit:
            sleep_s = backoff_s
            backoff_s = min(float(retry.max_backoff_s), backoff_s * retry.backoff_multiplier)
        else:
            sleep_s = min(300.0, backoff_s)

        print(f"[{_now()}] sleeping {sleep_s:.0f}s before retry...", flush=True)
        _sleep_with_jitter(sleep_s, retry.jitter_ratio)


if __name__ == "__main__":
    raise SystemExit(main())

