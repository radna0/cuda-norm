#!/usr/bin/env python3
"""
Manim animation for EAFT diagnostics (base vs pruned).

Requires:
  pip install manim numpy
  plus system deps (cairo, pango) depending on platform.

Example:
  manim -pql scripts/manim_eaft_animation.py EAFTComparison \\
    --input-json artifacts/eaft_plots/20260113_235934/eaft_data.json \\
    --pack reasoning_style_10k_v2 --seq-len 2048
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0


def _color_ramp(t: np.ndarray) -> np.ndarray:
    # Same palette as HTML (dark-mode optimized).
    stops = [
        (0.00, _hex_to_rgb("#0c1224")),
        (0.25, _hex_to_rgb("#1948a0")),
        (0.45, _hex_to_rgb("#25a89a")),
        (0.65, _hex_to_rgb("#eacc78")),
        (0.85, _hex_to_rgb("#f5995e")),
        (1.00, _hex_to_rgb("#ef5350")),
    ]
    out = np.zeros((t.shape[0], t.shape[1], 3), dtype=np.float32)
    for i in range(len(stops) - 1):
        a, ca = stops[i]
        b, cb = stops[i + 1]
        mask = (t >= a) & (t <= b)
        if not np.any(mask):
            continue
        u = (t - a) / max(1e-12, (b - a))
        u = np.clip(u, 0.0, 1.0)[..., None]
        out[mask] = (ca + (cb - ca) * u)[mask]
    return out


def _hist_to_image(hist: dict, *, out_w: int = 80, out_h: int = 60) -> np.ndarray:
    xbins = int(hist["xbins"])
    ybins = int(hist["ybins"])
    counts = np.array(hist["counts"], dtype=np.float32).reshape(xbins, ybins)
    # Normalize to log density.
    dens = np.log1p(counts)
    dens = dens / max(1e-12, dens.max())
    # Downsample by simple pooling.
    sx = max(1, xbins // out_w)
    sy = max(1, ybins // out_h)
    pooled = dens[: out_w * sx, : out_h * sy].reshape(out_w, sx, out_h, sy).mean(axis=(1, 3))
    pooled = pooled.T  # to (H, W)
    rgb = _color_ramp(pooled)
    return rgb


def _delta_to_image(hist_a: dict, hist_b: dict, *, out_w: int = 80, out_h: int = 60) -> np.ndarray:
    xbins = int(hist_a["xbins"])
    ybins = int(hist_a["ybins"])
    a = np.array(hist_a["counts"], dtype=np.float32).reshape(xbins, ybins)
    b = np.array(hist_b["counts"], dtype=np.float32).reshape(xbins, ybins)
    diff = np.log1p(b) - np.log1p(a)
    diff = diff / max(1e-9, np.abs(diff).max())
    sx = max(1, xbins // out_w)
    sy = max(1, ybins // out_h)
    pooled = diff[: out_w * sx, : out_h * sy].reshape(out_w, sx, out_h, sy).mean(axis=(1, 3))
    pooled = pooled.T  # (H, W)
    # Diverging: blue-white-red
    rgb = np.zeros((pooled.shape[0], pooled.shape[1], 3), dtype=np.float32)
    pos = pooled > 0
    neg = pooled < 0
    rgb[pos] = np.stack([0.86 * pooled[pos] + 0.14, 0.20, 0.20], axis=-1)
    rgb[neg] = np.stack([0.20, 0.20, 0.86 * (-pooled[neg]) + 0.14], axis=-1)
    rgb[~(pos | neg)] = 0.9
    return np.clip(rgb, 0.0, 1.0)


def _load_pack(res: dict, pack: str, seq_len: int) -> dict:
    for p in res["packs"]:
        if p["pack"] == pack:
            return p["seq"][str(seq_len)]
    raise KeyError(f"Pack {pack} not found")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--pack", default="UNION")
    parser.add_argument("--seq-len", type=int, default=2048)
    return parser.parse_args()


def _build_scene_data(res: dict, pack: str, seq_len: int) -> dict:
    s = _load_pack(res, pack, seq_len)
    return {
        "base": s["base"],
        "pruned": s["pruned"],
        "meta": res["meta"],
        "pack": pack,
        "seq_len": seq_len,
    }


def _render_scene(data: dict) -> None:
    from manim import ImageMobject, Scene, Text, VGroup, DOWN, RIGHT, UP

    class EAFTComparison(Scene):
        def construct(self):
            base = data["base"]
            pruned = data["pruned"]

            base_img = ImageMobject(_hist_to_image(base["hist2d_x_H"]))
            pruned_img = ImageMobject(_hist_to_image(pruned["hist2d_x_H"]))
            delta_img = ImageMobject(_delta_to_image(base["hist2d_x_H"], pruned["hist2d_x_H"]))

            title = Text(f"EAFT: {data['pack']} | seq={data['seq_len']}").scale(0.6).to_edge(UP)
            row = VGroup(base_img, pruned_img, delta_img).arrange(RIGHT, buff=0.5).scale(1.2).next_to(title, DOWN)

            labels = VGroup(
                Text("Base").scale(0.5).next_to(base_img, DOWN),
                Text("Pruned").scale(0.5).next_to(pruned_img, DOWN),
                Text("Δ Density").scale(0.5).next_to(delta_img, DOWN),
            )

            stats = Text(
                f"ΔPPL={pruned['ppl']-base['ppl']:+.3f} | CC Δ={pruned['cc_rate_base_thr']-base['cc_rate']:+.4f}",
                font_size=24,
            ).next_to(row, DOWN)

            self.add(title, row, labels, stats)
            self.wait(2)

    EAFTComparison().render()


def main() -> None:
    args = _parse_args()
    res = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    data = _build_scene_data(res, args.pack, args.seq_len)
    _render_scene(data)


if __name__ == "__main__":
    main()
