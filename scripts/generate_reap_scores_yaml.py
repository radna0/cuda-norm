#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def _coerce_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def build_scores(
    *,
    kind: str,
    method: str,
    score_key: str,
    saliency_parquet: Path,
    ranking_by_layer_json: Path,
    profile_meta_json: Path | None,
    sample_json: Path | None,
) -> dict[str, Any]:
    df = pd.read_parquet(saliency_parquet)
    required = {"layer", "expert", score_key}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in {saliency_parquet}: {sorted(missing)}")

    ranking = _read_json(ranking_by_layer_json)
    if not isinstance(ranking, list) or not ranking:
        raise SystemExit(f"ranking_by_layer_json must be a non-empty list: {ranking_by_layer_json}")

    meta = _read_json(profile_meta_json) if profile_meta_json else {}
    sample = _read_json(sample_json) if sample_json else {}

    num_layers = int(meta.get("num_layers") or (max(_coerce_int(x) for x in df["layer"].unique()) + 1))
    num_experts = int(meta.get("num_experts") or (max(_coerce_int(x) for x in df["expert"].unique()) + 1))

    # Build per-layer sorted expert stats (for later "try other keep_frac without re-profiling").
    layer_stats: list[dict[str, Any]] = []
    for li in range(num_layers):
        sdf = df[df["layer"] == li].copy()
        if sdf.empty:
            continue
        # Sort descending by total score mass.
        sdf = sdf.sort_values(score_key, ascending=False)
        experts: list[dict[str, Any]] = []
        for _, r in sdf.iterrows():
            experts.append(
                {
                    "expert": _coerce_int(r.get("expert")),
                    "score": _coerce_float(r.get(score_key)),
                    "count": _coerce_int(r.get("count")),
                    "gate_mean": _coerce_float(r.get("gate_mean")),
                    "norm_mean": _coerce_float(r.get("norm_mean")),
                    "saliency_mean": _coerce_float(r.get("saliency_mean")),
                    "eaft_saliency_mean": _coerce_float(r.get("eaft_saliency_mean")),
                    "pos_count": _coerce_int(r.get("pos_count")),
                    "neg_count": _coerce_int(r.get("neg_count")),
                }
            )
        layer_stats.append({"layer": li, "experts": experts})

    out: dict[str, Any] = {
        "schema_version": 1,
        "kind": str(kind),
        "method": str(method),
        "score_key": str(score_key),
        "score_definition": (
            "Total expert contribution mass accumulated over selected tokens. "
            "For EAFT-REAP this is w_t * gate_j(x) * ||f_j(x)||_2 summed over tokens where expert j was selected."
        ),
        "inputs": {
            "saliency_parquet": str(saliency_parquet),
            "ranking_by_layer_json": str(ranking_by_layer_json),
            "profile_meta_json": str(profile_meta_json) if profile_meta_json else None,
            "sample_json": str(sample_json) if sample_json else None,
        },
        "profile_meta": meta,
        "sample": sample,
        "summary": {
            "num_layers": int(num_layers),
            "num_experts": int(num_experts),
        },
        "ranking_by_layer": ranking,
        "layer_scores": layer_stats,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", default="reap_scores")
    ap.add_argument("--method", default="eaftreap")
    ap.add_argument("--score-key", default="eaft_gate_norm_sum")
    ap.add_argument("--saliency-parquet", required=True)
    ap.add_argument("--ranking-by-layer-json", required=True)
    ap.add_argument("--profile-meta-json", default="")
    ap.add_argument("--sample-json", default="")
    args = ap.parse_args()

    out_path = Path(args.out)
    out = build_scores(
        kind=str(args.kind),
        method=str(args.method),
        score_key=str(args.score_key),
        saliency_parquet=Path(args.saliency_parquet),
        ranking_by_layer_json=Path(args.ranking_by_layer_json),
        profile_meta_json=(Path(args.profile_meta_json) if str(args.profile_meta_json).strip() else None),
        sample_json=(Path(args.sample_json) if str(args.sample_json).strip() else None),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    print(f"[+] Wrote {out_path}")


if __name__ == "__main__":
    main()

