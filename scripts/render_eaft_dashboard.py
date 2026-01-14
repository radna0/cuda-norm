#!/usr/bin/env python3
"""
CPU-only renderer for the EAFT dashboard.

Reads an existing eaft_data.json and writes:
  - reports/20b_calib_packs_eaft_plots.html
  - reports/20b_calib_packs_eaft_degradation_summary.md
  - reports/20b_calib_packs_eaft_degradation_summary.csv

No Modal runs; this is pure local rendering.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import json
from pathlib import Path


def _load_render_module() -> object:
    repo_root = Path(__file__).resolve().parents[1]
    mod_path = repo_root / "modal" / "eval_calib_packs_eaft_plots.py"
    spec = importlib.util.spec_from_file_location("eaft_plots_mod", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module at {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True, help="Path to eaft_data.json")
    parser.add_argument("--html-out", default="reports/20b_calib_packs_eaft_plots.html")
    parser.add_argument("--summary-out", default="reports/20b_calib_packs_eaft_degradation_summary.md")
    args = parser.parse_args()

    input_path = Path(args.input_json)
    if not input_path.exists():
        raise SystemExit(f"Missing {input_path}")
    res = json.loads(input_path.read_text(encoding="utf-8"))

    mod = _load_render_module()
    html = mod._render_html_dashboard(res)

    html_path = Path(args.html_out)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    mod._write_summary_md(res, summary_path)

    print(f"[+] Wrote {html_path}")
    print(f"[+] Wrote {summary_path}")
    print(f"[+] Wrote {summary_path.parent / '20b_calib_packs_eaft_degradation_summary.csv'}")


if __name__ == "__main__":
    main()
