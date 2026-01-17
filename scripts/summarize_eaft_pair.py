from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _js_divergence_2d(a: Any, b: Any) -> float:
    # Jensen-Shannon divergence between 2D histograms (counts). The upstream JSON
    # stores hist2d as either:
    # - {"counts": [flat counts], "xbins": ..., "ybins": ...}
    # - nested lists (rare)
    # Returns a value in [0, ln(2)] with natural logs; small is "close".
    eps = 1e-12
    def _as_flat(x: Any) -> list[float]:
        if isinstance(x, dict):
            c = x.get("counts")
            if isinstance(c, list):
                return [float(v) for v in c]
            return []
        if isinstance(x, list):
            if not x:
                return []
            # If nested, flatten.
            if isinstance(x[0], list):
                out: list[float] = []
                for row in x:
                    if not isinstance(row, list):
                        continue
                    out.extend(float(v) for v in row)
                return out
            return [float(v) for v in x]
        return []

    flat_a = _as_flat(a)
    flat_b = _as_flat(b)
    # Pad to same length for safety.
    if len(flat_a) != len(flat_b):
        n = max(len(flat_a), len(flat_b))
        flat_a = flat_a + [0.0] * (n - len(flat_a))
        flat_b = flat_b + [0.0] * (n - len(flat_b))

    sa = sum(flat_a) or 0.0
    sb = sum(flat_b) or 0.0
    if sa <= 0 or sb <= 0:
        return 0.0
    pa = [(x / sa) for x in flat_a]
    pb = [(x / sb) for x in flat_b]
    m = [(0.5 * (x + y)) for x, y in zip(pa, pb, strict=True)]

    def _kl(p: list[float], q: list[float]) -> float:
        out = 0.0
        for pi, qi in zip(p, q, strict=True):
            if pi <= 0:
                continue
            out += pi * math.log(pi / max(eps, qi))
        return out

    return 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)


def _index_by_pack(model_json: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    packs = model_json.get("packs") or []
    if not isinstance(packs, list):
        raise TypeError("Expected top-level packs: list")
    for p in packs:
        if not isinstance(p, dict):
            continue
        name = str(p.get("pack") or "").strip()
        if not name:
            continue
        seq = p.get("seq") or {}
        if isinstance(seq, dict):
            out[name] = seq
    return out


def _model_label(model_json: dict[str, Any], fallback: str) -> str:
    meta = model_json.get("meta") or {}
    if isinstance(meta, dict):
        mid = str(meta.get("model_id") or "").strip()
        if mid:
            return mid
    return fallback


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left-json", required=True)
    ap.add_argument("--right-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--gates-json", default="harmony/cuda-norm/pruning/near_lossless_gates.json")
    args = ap.parse_args()

    left_path = Path(args.left_json)
    right_path = Path(args.right_json)
    gates_path = Path(args.gates_json)

    left = _load_json(left_path)
    right = _load_json(right_path)
    gates = _load_json(gates_path) if gates_path.exists() else {}

    left_label = _model_label(left, left_path.stem)
    right_label = _model_label(right, right_path.stem)

    left_packs = _index_by_pack(left)
    right_packs = _index_by_pack(right)
    common_packs = sorted(set(left_packs.keys()) & set(right_packs.keys()))

    thr = (gates.get("thresholds") if isinstance(gates, dict) else {}) or {}
    primary_pack = str((gates.get("primary_pack") if isinstance(gates, dict) else "") or "UNION")
    primary_seq_lens = (gates.get("primary_seq_lens") if isinstance(gates, dict) else None) or ["1024", "2048"]

    rows: list[dict[str, Any]] = []
    for pack in common_packs:
        lseq = left_packs[pack]
        rseq = right_packs[pack]
        for seq_len in sorted(set(lseq.keys()) & set(rseq.keys()), key=lambda s: int(s)):
            l = lseq[seq_len].get("model") if isinstance(lseq.get(seq_len), dict) else None
            r = rseq[seq_len].get("model") if isinstance(rseq.get(seq_len), dict) else None
            if not isinstance(l, dict) or not isinstance(r, dict):
                continue
            ppl_l = float(l.get("ppl") or 0.0)
            ppl_r = float(r.get("ppl") or 0.0)
            cc_l = float(l.get("cc_rate") or 0.0)
            cc_r = float(r.get("cc_rate") or 0.0)
            mp_l = float(l.get("mean_prob") or 0.0)
            mp_r = float(r.get("mean_prob") or 0.0)
            js2d = _js_divergence_2d(
                l.get("hist2d_x_H") or [],
                r.get("hist2d_x_H") or [],
            )
            rows.append(
                {
                    "pack": pack,
                    "seq_len": str(seq_len),
                    "ppl_left": ppl_l,
                    "ppl_right": ppl_r,
                    "delta_ppl": ppl_r - ppl_l,
                    "cc_left": cc_l,
                    "cc_right": cc_r,
                    "delta_cc": cc_r - cc_l,
                    "mean_p_left": mp_l,
                    "mean_p_right": mp_r,
                    "delta_mean_p": mp_r - mp_l,
                    "js2d": float(js2d),
                }
            )

    def _fmt(x: float, n: int = 4) -> str:
        return f"{x:.{n}f}"

    def _fmt_signed(x: float, n: int = 4) -> str:
        return ("+" if x >= 0 else "") + f"{x:.{n}f}"

    md: list[str] = []
    md += [
        "# EAFT parity summary (pair)",
        "",
        f"- Left: `{left_label}`",
        f"- Right: `{right_label}`",
        f"- Gates: `{gates.get('name','')}` (`{args.gates_json}`)",
        "",
    ]

    # Hero for UNION 1024/2048.
    hero = [r for r in rows if r["pack"] == primary_pack and r["seq_len"] in set(str(x) for x in primary_seq_lens)]
    if hero:
        md += ["## Hero (UNION)", ""]
        for r in sorted(hero, key=lambda x: int(x["seq_len"])):
            md.append(
                f"- seq={r['seq_len']}: "
                f"ΔPPL={_fmt_signed(r['delta_ppl'],3)} "
                f"| ΔCC={_fmt_signed(r['delta_cc']*100.0,3)}pp "
                f"| Δmean_p={_fmt_signed(r['delta_mean_p'],4)} "
                f"| JS2D={_fmt(r['js2d'],4)}"
            )
        md.append("")

        thr_abs = float(thr.get("max_abs_delta_ppl") or 0.0)
        thr_rel = float(thr.get("max_rel_delta_ppl") or 0.0)
        thr_cc = float(thr.get("max_abs_delta_cc_rate") or 0.0)
        thr_mp = float(thr.get("max_abs_delta_mean_prob") or 0.0)
        thr_js = float(thr.get("max_js2d") or 0.0)

        # Gate evaluation: all hero rows must pass.
        gate_ok = True
        reasons: list[str] = []
        for r in hero:
            ppl_ok = (abs(r["delta_ppl"]) <= thr_abs) and (
                abs(r["delta_ppl"]) <= thr_rel * max(1e-9, r["ppl_left"])
            )
            cc_ok = abs(r["delta_cc"]) <= thr_cc
            mp_ok = abs(r["delta_mean_p"]) <= thr_mp
            js_ok = r["js2d"] <= thr_js
            if not (ppl_ok and cc_ok and mp_ok and js_ok):
                gate_ok = False
                reasons.append(
                    f"seq={r['seq_len']} ppl_ok={ppl_ok} cc_ok={cc_ok} mean_p_ok={mp_ok} js_ok={js_ok}"
                )
        md += [
            f"## Gate Result: {'PASS' if gate_ok else 'FAIL'}",
            "",
            f"- Rule: |ΔPPL|<=min(abs={thr_abs}, rel={thr_rel}) and |ΔCC|<={thr_cc} and |Δmean_p|<={thr_mp} and JS2D<={thr_js}",
        ]
        if reasons:
            md += ["- Failures:"] + [f"  - {x}" for x in reasons]
        md.append("")

    md += [
        "## Full Table (Right vs Left)",
        "",
        "| pack | seq | ppl_L | ppl_R | Δppl | cc_L | cc_R | Δcc (pp) | mean_p_L | mean_p_R | Δmean_p | JS2D |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: (x["pack"], int(x["seq_len"]))):
        md.append(
            "| "
            + " | ".join(
                [
                    str(r["pack"]),
                    str(r["seq_len"]),
                    _fmt(r["ppl_left"], 3),
                    _fmt(r["ppl_right"], 3),
                    _fmt_signed(r["delta_ppl"], 3),
                    _fmt(r["cc_left"] * 100.0, 3),
                    _fmt(r["cc_right"] * 100.0, 3),
                    _fmt_signed((r["delta_cc"] * 100.0), 3),
                    _fmt(r["mean_p_left"], 4),
                    _fmt(r["mean_p_right"], 4),
                    _fmt_signed(r["delta_mean_p"], 4),
                    _fmt(r["js2d"], 4),
                ]
            )
            + " |"
        )

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[+] Wrote {args.out_md}")


if __name__ == "__main__":
    main()
