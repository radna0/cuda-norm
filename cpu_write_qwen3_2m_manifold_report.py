#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(x: float | None) -> str:
    if x is None or math.isnan(x):
        return "n/a"
    return f"{100.0 * x:.2f}%"


def _f(x: float | None, nd: int = 4) -> str:
    if x is None or math.isnan(x):
        return "n/a"
    return f"{x:.{nd}f}"


def _render_section(label: str, m: dict[str, Any]) -> str:
    pca = m.get("pca") or {}
    dens = m.get("density_xyz") or {}
    knn = m.get("knn") or {}
    umap = m.get("umap") or {}

    ev1 = pca.get("explained_var_pc1")
    ev3 = pca.get("explained_var_pc1_3")

    dens_totals = dens.get("totals") or {}
    mean_occ = dens_totals.get("mean_occupancy")

    nn1 = (knn.get("nn1_cos") or {})
    dup = (nn1.get("dup_rates") or {})

    # UMAP k=10 summary (if present)
    umap10 = umap.get("10") or {}

    lines: list[str] = []
    lines.append(f"## {label}")
    lines.append("")
    lines.append(f"- embedding_dir: `{m.get('embedding_dir')}`")
    lines.append(
        f"- PCA variance: pc1={_pct(ev1)} pc1-3={_pct(ev3)} (pc1 dominance = collapse indicator)"
    )
    lines.append(
        f"- PCA voxel occupancy (xyz, mean over dataset×mix_group): {_pct(mean_occ)}"
    )
    if knn:
        lines.append(
            f"- kNN redundancy (sample n={knn.get('n')}): nn1>=0.99={_pct(dup.get('0.99'))}, nn1>=0.999={_pct(dup.get('0.999'))}"
        )
    if umap10:
        lines.append(
            f"- Neighborhood purity (k=10, embedding space): dataset={_pct(umap10.get('purity_dataset'))}, mix_group={_pct(umap10.get('purity_mix_group'))}, len_bucket={_pct(umap10.get('purity_len_bucket'))}"
        )
        lines.append(f"- UMAP overlap@10 (local structure preserved): {_f(umap10.get('overlap'))}")
    lines.append("")

    lines.append("**Top density groups (xyz, by points)**")
    lines.append("")
    lines.append("| dataset | mix_group | points | voxels | avg pts/voxel |")
    lines.append("|---|---:|---:|---:|---:|")
    for g in (dens.get("groups") or [])[:10]:
        lines.append(
            f"| {g['dataset']} | {g['mix_group']} | {g['points']} | {g['voxels']} | {g['avg_pts_per_voxel']:.2f} |"
        )
    lines.append("")

    return "\n".join(lines)


def _summarize_findings(label: str, m: dict[str, Any]) -> list[str]:
    pca = m.get("pca") or {}
    dens = m.get("density_xyz") or {}
    knn = m.get("knn") or {}
    km = m.get("kmeans") or {}

    ev1 = float(pca.get("explained_var_pc1") or float("nan"))
    occ = (dens.get("totals") or {}).get("mean_occupancy")
    dup = ((knn.get("nn1_cos") or {}).get("dup_rates") or {})
    nn99 = dup.get("0.99")
    nn999 = dup.get("0.999")

    # Cluster KPIs (if present).
    k1000 = km.get("1000") or {}
    eff = k1000.get("effective_clusters_exp_entropy")
    g = k1000.get("gini")

    lines: list[str] = []
    lines.append(f"- {label}: pc1={_pct(ev1)} occupancy={_pct(occ)} nn1>=0.99={_pct(nn99)} nn1>=0.999={_pct(nn999)}")
    if eff is not None and g is not None:
        lines.append(f"  - kmeans(k=1000): effective_clusters≈{_f(float(eff), 2)} gini≈{_f(float(g), 4)}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a combined prompt+behavior manifold analysis report from diagnostics.json.")
    ap.add_argument("--prompt_json", type=str, required=True)
    ap.add_argument("--behavior_json", type=str, required=True)
    ap.add_argument("--out_md", type=str, required=True)
    args = ap.parse_args()

    prompt = _read_json(Path(args.prompt_json))
    behavior = _read_json(Path(args.behavior_json))

    lines: list[str] = []
    lines.append("# Qwen3 2M embedding manifold analysis")
    lines.append("")
    lines.append(f"- generated_at: {_now()}")
    lines.append("")
    lines.append("## What to open (full 2M, interactive 3D)")
    lines.append("")
    lines.append("- Prompt view pointcloud: `harmony/cuda-norm/artifacts/map_view/qwen3_prompt_full_pca_2m_20260112/pointcloud_2m_3d.html`")
    lines.append("- Behavior view pointcloud: `harmony/cuda-norm/artifacts/map_view/qwen3_behavior_full_pca_2m_20260112/pointcloud_2m_3d.html`")
    lines.append("- Prompt density: `harmony/cuda-norm/artifacts/map_view/qwen3_prompt_full_pca_2m_20260112/density_view.html`")
    lines.append("- Behavior density: `harmony/cuda-norm/artifacts/map_view/qwen3_behavior_full_pca_2m_20260112/density_view.html`")
    lines.append("")

    lines.append("## Key findings (from quantitative diagnostics)")
    lines.append("")
    lines.extend(_summarize_findings("Prompt view", prompt))
    lines.extend(_summarize_findings("Behavior view", behavior))
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- “Two blobs” in PCA/UMAP is not inherently bad; it often reflects major modes (dataset/style). The real question is redundancy *within* blobs and whether each mode has sufficient internal structure.")
    lines.append("- If nn1>=0.99 is high, compress via medoids/LSH/k-means; this preserves coverage and saves compute.")
    lines.append("")

    lines.append(_render_section("Prompt view", prompt))
    lines.append(_render_section("Behavior view", behavior))

    lines.append("## Recommended next steps (quality-first)")
    lines.append("")
    lines.append("1) Use the prompt view for coverage-driven packs: compress with LSH/k-means into a 200k cover, then derive 1k/10k/100k packs and SWA/full variants.")
    lines.append("2) For agentic/tool behavior, select from the behavior view with explicit per-tool-sequence quotas (coverage over tool policies, not just dataset name).")
    lines.append("3) For deep reasoning diversity, build a dedicated reasoning-excerpt view on a 50k–300k subset (1024–2048 tokens), embed, then select a reasoning_style_10k pack by clustering.")
    lines.append("4) Track KPIs (dup rates, voxel occupancy, k-means gini/effective_clusters) for each new view so decisions are backed by numbers, not just plots.")
    lines.append("")

    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
