from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def _read_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        env[k] = v
    return env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-id", required=True)
    ap.add_argument("--pid", default="", help="Optional remote PID to probe (from Versa details.pid).")
    ap.add_argument("--env-file", default="harmony/cuda-norm/.env")
    ap.add_argument("--kaggle-url", default="")
    args = ap.parse_args()

    env_file = Path(args.env_file)
    file_env = _read_env_file(env_file)
    kaggle_url = args.kaggle_url.strip() or file_env.get("KAGGLE_URL", "").strip()
    if not kaggle_url:
        raise SystemExit(f"[err] missing KAGGLE_URL (pass --kaggle-url or set it in {env_file})")

    repo_root = Path(__file__).resolve().parents[2]
    versa_py_path = str(repo_root / "third_party" / "Versa")

    pid = args.pid.strip()
    pid_line = ""
    if pid:
        pid_line = f"""
echo "[status] pid={pid}"
if ps -p "{pid}" -o pid,etime,comm,args >/dev/null 2>&1; then
  ps -p "{pid}" -o pid,etime,comm,args || true
else
  echo "[status] pid_not_running"
fi
""".strip()

    remote_script = "\n".join(
        [
            "set -euo pipefail",
            "echo \"[status] nvidia-smi\"",
            "nvidia-smi || true",
            "echo \"[status] compute apps\"",
            "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true",
            "echo \"[status] ps (top mem)\"",
            "ps -eo pid,comm,args,%mem --sort=-%mem | sed -n '1,80p' || true",
            pid_line,
        ]
    ).strip()

    cmd = [
        sys.executable,
        "-m",
        "versa",
        "run",
        "--backend",
        "jupyter",
        "--url",
        kaggle_url,
        "--kernel-id",
        args.kernel_id,
        "--cwd",
        "/kaggle/working",
        "bash",
        "-lc",
        remote_script,
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = versa_py_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd_redacted = [("<kaggle_url_redacted>" if x == kaggle_url else x) for x in cmd]
    print("[*] running:", " ".join(shlex.quote(x) for x in cmd_redacted), flush=True)
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()

