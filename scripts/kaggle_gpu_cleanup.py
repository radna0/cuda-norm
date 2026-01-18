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
    ap.add_argument("--env-file", default="harmony/cuda-norm/.env")
    ap.add_argument("--kaggle-url", default="")
    ap.add_argument(
        "--aggressive",
        action="store_true",
        help="Also pkill common Versa/Modal runner processes (may interrupt other jobs).",
    )
    args = ap.parse_args()

    env_file = Path(args.env_file)
    file_env = _read_env_file(env_file)
    kaggle_url = args.kaggle_url.strip() or file_env.get("KAGGLE_URL", "").strip()
    if not kaggle_url:
        raise SystemExit(f"[err] missing KAGGLE_URL (pass --kaggle-url or set it in {env_file})")

    repo_root = Path(__file__).resolve().parents[2]
    versa_py_path = str(repo_root / "third_party" / "Versa")

    remote_script = r"""
set -euo pipefail
echo "[cleanup] before"
nvidia-smi || true
pulse_ps() {
  echo "[cleanup] ps (python/versa/modal):"
  ps -eo pid,user,comm,args --sort=-%mem | sed -n '1,80p' || true
}
pulse_ps
pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr '\n' ' ' | xargs || true)"
echo "[cleanup] gpu_pids=${pids:-<none>}"
if [[ -n "${pids}" ]]; then
  for p in ${pids}; do
    echo "[cleanup] kill -9 ${p}"
    kill -9 "${p}" 2>/dev/null || true
  done
fi
""".strip()

    if args.aggressive:
        # Avoid pkill/pgrep patterns which can kill the cleanup command itself
        # (because the pattern appears in the current process cmdline). Instead,
        # find target PIDs via `ps` and kill explicitly, excluding our own PID.
        remote_script += r"""
echo "[cleanup] aggressive: killing leaked Versa modal_run processes..."
python - <<'PY'
import os
import signal
import subprocess

me = os.getpid()
txt = subprocess.check_output(["ps", "-eo", "pid,args"], text=True, errors="ignore")
targets = []
for line in txt.splitlines()[1:]:
    line = line.strip()
    if not line:
        continue
    parts = line.split(None, 1)
    if not parts:
        continue
    try:
        pid = int(parts[0])
    except Exception:
        continue
    if pid == me:
        continue
    if ".versa/modal_run.py" in line:
        targets.append(pid)

print("[cleanup] modal_run_pids=", targets)
for pid in targets:
    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:
        pass
PY
"""

    remote_script += r"""
sleep 2
echo "[cleanup] after"
nvidia-smi || true
pulse_ps
""".strip()

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
    print("[*] running:", " ".join(shlex.quote(x) for x in cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
