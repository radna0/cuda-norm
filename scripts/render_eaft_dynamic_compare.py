#!/usr/bin/env python3
"""
Render a dynamic EAFT comparison dashboard from multiple single-model JSON runs.

Input: one or more JSON files produced by modal/collect_calib_packs_eaft_single.py
Output: reports/eaft_dynamic_compare.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_one(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data.get("meta", {})
    model_id = meta.get("model_id") or path.stem
    run_id = meta.get("run_id") or meta.get("run_name") or meta.get("timestamp") or path.stem
    packs = {}
    for p in data.get("packs", []):
        packs[p["pack"]] = p["seq"]
    return {"model_id": str(model_id), "run_id": str(run_id), "meta": meta, "packs": packs}


def _load_gates() -> dict:
    gates_path = Path(__file__).resolve().parents[1] / "pruning" / "near_lossless_gates.json"
    if not gates_path.exists():
        return {}
    try:
        return json.loads(gates_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _merge_runs(paths: list[Path]) -> dict:
    models = {}
    meta_ref = None
    for p in paths:
        d = _load_one(p)
        key = d["model_id"]
        if key in models:
            key = f"{key} @{d['run_id']}"
        models[key] = {"meta": d["meta"], "packs": d["packs"]}
        if meta_ref is None:
            meta_ref = d["meta"]
    meta_ref = meta_ref or {}
    gates = _load_gates()
    # union packs + seq_lens across models
    pack_names = sorted({pack for m in models.values() for pack in m["packs"].keys()})
    seq_lens = sorted({seq for m in models.values() for pack in m["packs"].values() for seq in pack.keys()}, key=lambda x: int(x))
    return {
        "meta": {
            "dataset_repo": meta_ref.get("dataset_repo", ""),
            "pack_files": meta_ref.get("pack_files", []),
            "top_k": meta_ref.get("top_k", meta_ref.get("topK", "")),
            "axes": meta_ref.get("axes", {"x": "p_t", "y": "H_topK/ln(K)"}),
            "prob_scale": meta_ref.get("prob_scale", "linear"),
            "x_min": meta_ref.get("x_min", 0.0),
            "x_max": meta_ref.get("x_max", 1.0),
            "entropy_topk": meta_ref.get("entropy_topk", 20),
            "cc_quantile": meta_ref.get("cc_quantile", 0.15),
            "seq_lens": seq_lens,
            "pack_names": pack_names,
            "gates": gates,
        },
        "models": models,
    }


def _load_plotly_js() -> str:
    """
    Load Plotly JS from our repo so the generated HTML is self-contained and
    works offline.

    Downloaded once into `third_party/plotly-2.30.0.min.js`.
    """
    plotly_path = Path(__file__).resolve().parents[1] / "third_party" / "plotly-2.30.0.min.js"
    if not plotly_path.exists():
        return ""
    return plotly_path.read_text(encoding="utf-8", errors="ignore")


def _render_html_plotly(payload: dict) -> str:
    data_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    plotly_js = _load_plotly_js()
    if not plotly_js:
        raise RuntimeError("Missing Plotly JS bundle at `third_party/plotly-2.30.0.min.js`.")

    # Build as a plain string template (NOT an f-string) to avoid conflicts with
    # JS template literals / object braces.
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EAFT diagnostics — dynamic compare</title>
  <style>
    :root {{
      --bg: #070b12;
      --panel: #0b1220;
      --grid: #1f2937;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --good: #34d399;
      --bad: #fb7185;
      --warn: #fbbf24;
      --accent: #60a5fa;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 20px;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      background: var(--bg);
      color: var(--text);
    }}
    h1 {{ margin: 0; font-size: 20px; letter-spacing: .02em; }}
    h2 {{ margin: 0; font-size: 14px; color: var(--muted); font-weight: 800; }}
    .panel {{
      background: linear-gradient(180deg, var(--panel), #070b12);
      border: 1px solid var(--grid);
      border-radius: 14px;
      padding: 14px;
    }}
    .controls {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: end; }}
    label {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }}
    select, button {{
      background: #060a12;
      color: var(--text);
      border: 1px solid var(--grid);
      border-radius: 10px;
      padding: 8px 10px;
    }}
    button {{ cursor: pointer; }}
    .hero {{
      display: grid;
      grid-template-columns: 1.2fr 2fr;
      gap: 12px;
      align-items: stretch;
    }}
    .hero-left {{
      background: radial-gradient(900px 400px at 30% 0%, rgba(96,165,250,0.16), rgba(0,0,0,0.0));
      border: 1px solid var(--grid);
      border-radius: 14px;
      padding: 14px;
    }}
    .hero-right {{
      display: grid;
      grid-template-columns: repeat(4, minmax(160px, 1fr));
      gap: 10px;
    }}
    .stat {{
      background: rgba(15, 23, 42, 0.55);
      border: 1px solid var(--grid);
      border-radius: 14px;
      padding: 12px;
      min-width: 0;
    }}
    .stat .k {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }}
    .stat .v {{ font-size: 26px; font-weight: 900; margin-top: 6px; }}
    .stat .s {{ font-size: 12px; color: var(--muted); margin-top: 4px; line-height: 1.25; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\"; }}
    .good {{ color: var(--good); }}
    .bad {{ color: var(--bad); }}
    .warn {{ color: var(--warn); }}
    .grid2 {{
      display: flex;
      gap: 12px;
      overflow-x: auto; /* keep side-by-side always */
      padding-bottom: 4px;
    }}
    .col {{
      flex: 0 0 min(980px, 98vw);
      min-width: 720px;
    }}
    .plot {{
      height: 620px;
      width: 100%;
      border: 1px solid var(--grid);
      border-radius: 14px;
      overflow: hidden;
      background: #05070e;
    }}
    .plot.small {{ height: 420px; }}
    .plot.hist {{ height: 380px; }}
    .subgrid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ border: 1px solid var(--grid); padding: 8px 10px; }}
    th {{
      color: var(--muted);
      font-weight: 900;
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: .08em;
      background: #060a12;
    }}
    td {{ text-align: right; }}
    td.left {{ text-align: left; }}
    .footer {{ color: var(--muted); font-size: 12px; line-height: 1.35; }}
    @media (max-width: 980px) {{
      .hero {{ grid-template-columns: 1fr; }}
      .hero-right {{ grid-template-columns: repeat(2, minmax(160px, 1fr)); }}
    }}
  </style>
</head>
<body>
  <div class=\"panel\">
    <div class=\"controls\" style=\"justify-content: space-between; align-items: center;\">
      <div>
        <h1>EAFT diagnostics — dynamic compare</h1>
        <div class=\"footer\" style=\"margin-top: 6px;\">
          Dataset: <span class=\"mono\" id=\"datasetRepo\"></span><br/>
          Packs: <span class=\"mono\" id=\"packsInfo\"></span><br/>
          Axes: <span class=\"mono\" id=\"axesInfo\"></span>
        </div>
      </div>
      <div class=\"controls\">
        <div><label>Left</label><br/><select id=\"leftModel\"></select></div>
        <div><label>Right</label><br/><select id=\"rightModel\"></select></div>
        <div><label>Pack</label><br/><select id=\"packSel\"></select></div>
        <div><label>Seq Len</label><br/><select id=\"seqSel\"></select></div>
        <div><label>Actions</label><br/><button id=\"swapBtn\">Swap</button></div>
      </div>
    </div>
  </div>

  <div class=\"panel\" style=\"margin-top: 12px;\">
    <div class=\"hero\">
      <div class=\"hero-left\">
        <h2 id=\"verdictTitle\">Verdict</h2>
        <div class=\"mono\" id=\"verdictBig\" style=\"margin-top:10px;font-size:34px;font-weight:900;line-height:1.05;\"></div>
        <div class=\"footer\" id=\"verdictExplain\" style=\"margin-top:10px;\"></div>
      </div>
      <div class=\"hero-right\" id=\"heroStats\"></div>
    </div>
  </div>

  <div class=\"panel\" style=\"margin-top: 12px;\">
    <div class=\"grid2\">
      <div class=\"col\">
        <h2 class=\"mono\" id=\"leftTitle\">Left</h2>
        <div id=\"leftHeat\" class=\"plot\"></div>
        <div class=\"subgrid\" style=\"margin-top: 10px;\">
          <div id=\"leftHistP\" class=\"plot hist\"></div>
          <div id=\"leftHistH\" class=\"plot hist\"></div>
        </div>
        <div id=\"leftScatter\" class=\"plot small\" style=\"margin-top:10px;\"></div>
      </div>
      <div class=\"col\">
        <h2 class=\"mono\" id=\"rightTitle\">Right</h2>
        <div id=\"rightHeat\" class=\"plot\"></div>
        <div class=\"subgrid\" style=\"margin-top: 10px;\">
          <div id=\"rightHistP\" class=\"plot hist\"></div>
          <div id=\"rightHistH\" class=\"plot hist\"></div>
        </div>
        <div id=\"rightScatter\" class=\"plot small\" style=\"margin-top:10px;\"></div>
      </div>
    </div>
  </div>

  <div class=\"panel\" style=\"margin-top: 12px;\">
    <h2>Δ density (Right − Left)</h2>
    <div id=\"deltaHeat\" class=\"plot\" style=\"height:680px;\"></div>
  </div>

  <div class=\"panel\" style=\"margin-top: 12px;\">
      <div class=\"controls\" style=\"justify-content: space-between;\">
      <div>
        <h2>Leaderboard (aggregate, all packs)</h2>
        <div class=\"footer\" id=\"leaderExplain\" style=\"margin-top:6px;\"></div>
      </div>
      <div class=\"controls\">
        <div>
          <label>Heatmap Scale</label><br/>
          <select id=\"heatScale\">
            <option value=\"log1p\">log1p(count)</option>
            <option value=\"count\">count</option>
          </select>
        </div>
        <div><label>Seq Len</label><br/><select id=\"leaderSeq\"></select></div>
        <div><label>Actions</label><br/><button id=\"downloadCsvBtn\">Download CSV</button></div>
      </div>
    </div>
    <div id=\"leaderTable\" style=\"margin-top: 10px;\"></div>
  </div>

  <div class=\"panel\" style=\"margin-top: 12px;\">
    <h2>Metric Breakdown (this pack)</h2>
    <div id=\"compareTable\" style=\"margin-top: 10px;\"></div>
    <div class=\"footer\" style=\"margin-top: 10px;\">
      Interpretation: Degradation Score is a z-score of (ΔPPL, CC Δ under left thresholds, JS2D, Δ mean p).<br/>
      Rule of thumb: z≈2.58 ≈ 1% two-tailed (strong evidence), z≈3.3 ≈ 0.1% two-tailed (very strong).
    </div>
  </div>

  <script id="DATA" type="application/json">__EAFT_DATA_JSON__</script>
  <script>__PLOTLY_MIN_JS__</script>
  <script>
    const DATA = JSON.parse(document.getElementById(\"DATA\").textContent);
    const MODELS = Object.keys(DATA.models || {{}});
    const EPS = {{ ppl: 0.05, js2d: 0.005, cc_pp: 0.05 }};
    const UI = {{ heatScale: \"log1p\" }};

    function showError(msg) {{
      const id = \"__eaft_err\";
      let el = document.getElementById(id);
      if (!el) {{
        el = document.createElement(\"div\");
        el.id = id;
        el.style.cssText = \"position:fixed;left:16px;right:16px;bottom:16px;z-index:999999;padding:12px 14px;border-radius:12px;border:1px solid #7f1d1d;background:#110a0a;color:#fecaca;font-family:ui-monospace,Menlo,Consolas,monospace;white-space:pre-wrap;max-height:42vh;overflow:auto;\";
        document.body.appendChild(el);
      }}
      el.textContent = String(msg || \"Unknown error\");
    }}
    window.addEventListener(\"error\", (e) => {{
      try {{
        showError(`JS error: ${e.message}\\n@ ${e.filename}:${e.lineno}:${e.colno}`);
      }} catch (_) {{}}
    }});
    window.addEventListener(\"unhandledrejection\", (e) => {{
      try {{ showError(`Unhandled rejection: ${e.reason}`); }} catch (_) {{}}
    }});

    function fmt(x, d=4) {{
      if (x === null || x === undefined || Number.isNaN(x)) return \"—\";
      return Number(x).toFixed(d);
    }}
    function fmtSigned(x, d=4) {{
      if (x === null || x === undefined || Number.isNaN(x)) return \"—\";
      const v = Number(x);
      return (v >= 0 ? \"+\" : \"\") + v.toFixed(d);
    }}
    function zScores(values) {{
      const n = values.length;
      if (!n) return values.map(() => 0);
      const mean = values.reduce((a, b) => a + b, 0) / n;
      const variance = values.reduce((a, b) => a + (b - mean) * (b - mean), 0) / n;
      const std = Math.sqrt(variance) || 1e-9;
      return values.map(v => (v - mean) / std);
    }}
    function pFromZ(z) {{
      const x = Math.abs(z) / Math.SQRT2;
      const t = 1 / (1 + 0.3275911 * x);
      const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429;
      const erf = 1 - (((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t) * Math.exp(-x*x);
      return 1 - erf;
    }}
    function deltaClass(v) {{
      if (v === null || v === undefined || Number.isNaN(v)) return \"\";
      if (v > 0) return \"bad\";
      if (v < 0) return \"good\";
      return \"\";
    }}
    function reshape2D(flat, xbins, ybins) {{
      // Python collector stores hist2d as flatten of shape (xbins, ybins)
      // with indexing hist2d[ix, iy]. We render as rows=ybins, cols=xbins.
      const z = [];
      for (let y = 0; y < ybins; y++) {{
        const row = [];
        for (let x = 0; x < xbins; x++) {{
          const idx = x * ybins + y;
          row.push(flat[idx] || 0);
        }}
        z.push(row);
      }}
      return z;
    }}
    function sum2D(z) {{
      let s = 0;
      for (const row of z) for (const v of row) s += v;
      return s;
    }}
    function jsDiv2D(a, b) {{
      const eps = 1e-12;
      const sa = sum2D(a) + eps;
      const sb = sum2D(b) + eps;
      let js = 0;
      for (let y = 0; y < a.length; y++) {{
        for (let x = 0; x < a[0].length; x++) {{
          const pa = (a[y][x] + eps) / sa;
          const pb = (b[y][x] + eps) / sb;
          const m = 0.5 * (pa + pb);
          js += 0.5 * (pa * Math.log(pa / m) + pb * Math.log(pb / m));
        }}
      }}
      return js;
    }}
    function ccRateFromThresholds(hist2d, xEdges, yEdges, xThr, hThr) {{
      const xbins = xEdges.length - 1;
      const ybins = yEdges.length - 1;
      const z = reshape2D(hist2d, xbins, ybins);
      let total = 0;
      let cc = 0;
      for (let y = 0; y < ybins; y++) {{
        const y0 = yEdges[y];
        const y1 = yEdges[y + 1];
        const yOk = (y1 <= hThr) || (y0 <= hThr && y1 >= hThr);
        for (let x = 0; x < xbins; x++) {{
          const v = z[y][x];
          total += v;
          const x0 = xEdges[x];
          const x1 = xEdges[x + 1];
          const xOk = (x1 <= xThr) || (x0 <= xThr && x1 >= xThr);
          if (xOk && yOk) cc += v;
        }}
      }}
      return total ? (cc / total) : 0;
    }}
    function getSeq(modelId, packName, seqLen) {{
      const m = DATA.models[modelId];
      if (!m) return null;
      const p = m.packs[packName];
      if (!p) return null;
      return p[String(seqLen)] || null;
    }}
    function computePair(leftId, rightId, packName, seqLen) {{
      const leftSeq = getSeq(leftId, packName, seqLen);
      const rightSeq = getSeq(rightId, packName, seqLen);
      if (!leftSeq || !rightSeq) return null;
      const L = leftSeq.model;
      const R = rightSeq.model;
      if (!L.hist2d_x_H || !R.hist2d_x_H) return null;
      const xEdges = L.hist2d_x_H.x_edges;
      const yEdges = L.hist2d_x_H.y_edges;
      const xbins = L.hist2d_x_H.xbins;
      const ybins = L.hist2d_x_H.ybins;
      const zL = reshape2D(L.hist2d_x_H.counts, xbins, ybins);
      const zR = reshape2D(R.hist2d_x_H.counts, xbins, ybins);
      const js2d = jsDiv2D(zL, zR);
      const ccL = ccRateFromThresholds(L.hist2d_x_H.counts, xEdges, yEdges, L.x_thr, L.h_thr);
      const ccR = ccRateFromThresholds(R.hist2d_x_H.counts, xEdges, yEdges, L.x_thr, L.h_thr);
      const ccDelta = ccR - ccL;
      const deltaPpl = (R.ppl - L.ppl);
      const meanPDropPP = (L.mean_prob - R.mean_prob) * 100.0;
      return {{ leftSeq, rightSeq, L, R, xEdges, yEdges, zL, zR, js2d, ccL, ccR, ccDelta, deltaPpl, meanPDropPP }};
    }}

    const PLOT_THEME = {{
      paper_bgcolor: \"#05070e\",
      plot_bgcolor: \"#05070e\",
      font: {{ color: \"#e5e7eb\", family: \"ui-sans-serif, system-ui\" }},
      xaxis: {{ gridcolor: \"#1f2937\", zerolinecolor: \"#1f2937\", color: \"#cbd5e1\" }},
      yaxis: {{ gridcolor: \"#1f2937\", zerolinecolor: \"#1f2937\", color: \"#cbd5e1\" }},
      margin: {{ l: 66, r: 20, t: 34, b: 54 }},
    }};

    function heatmap(divId, title, zCounts, xEdges, yEdges, xThr, hThr, isDelta=false) {{
      const xCenters = [];
      for (let i = 0; i < xEdges.length - 1; i++) xCenters.push(0.5*(xEdges[i] + xEdges[i+1]));
      const yCenters = [];
      for (let j = 0; j < yEdges.length - 1; j++) yCenters.push(0.5*(yEdges[j] + yEdges[j+1]));
      let z = zCounts;
      let colorscale = [
        [0.0, \"#0c1224\"],
        [0.2, \"#1b3a8a\"],
        [0.4, \"#1f9e89\"],
        [0.6, \"#e9c46a\"],
        [0.8, \"#f4a261\"],
        [1.0, \"#e76f51\"],
      ];
      let zmin = null, zmax = null;
      if (!isDelta) {{
        if (UI.heatScale === \"count\") {{
          z = zCounts;
        }} else {{
          z = zCounts.map(row => row.map(v => Math.log1p(v)));
        }}
      }} else {{
        let absMax = 0;
        for (const row of zCounts) for (const v of row) absMax = Math.max(absMax, Math.abs(v));
        zmin = -absMax; zmax = absMax;
        colorscale = [
          [0.0, \"#1d4ed8\"],
          [0.5, \"#111827\"],
          [1.0, \"#fb7185\"],
        ];
      }}
      const trace = {{
        type: \"heatmap\",
        x: xCenters,
        y: yCenters,
        z,
        zmin,
        zmax,
        colorscale,
        hovertemplate: \"x=%{x:.4f}<br>H=%{y:.4f}<br>value=%{z:.4f}<extra></extra>\",
      }};
      const layout = {{
        ...PLOT_THEME,
        title: {{ text: title, font: {{ color: \"#e5e7eb\", size: 14 }} }},
        xaxis: {{ ...PLOT_THEME.xaxis, title: DATA.meta.axes?.x || \"p_t\" }},
        yaxis: {{ ...PLOT_THEME.yaxis, title: DATA.meta.axes?.y || \"H\" }},
        shapes: [
          {{ type: \"rect\", xref: \"x\", yref: \"y\", x0: xEdges[0], x1: xThr, y0: yEdges[0], y1: hThr, line: {{ width: 1, color: \"#94a3b8\" }}, fillcolor: \"rgba(148,163,184,0.08)\" }},
          {{ type: \"line\", xref: \"x\", yref: \"y\", x0: xThr, x1: xThr, y0: yEdges[0], y1: yEdges[yEdges.length-1], line: {{ color: \"#94a3b8\", width: 1, dash: \"dot\" }} }},
          {{ type: \"line\", xref: \"x\", yref: \"y\", x0: xEdges[0], x1: xEdges[xEdges.length-1], y0: hThr, y1: hThr, line: {{ color: \"#94a3b8\", width: 1, dash: \"dot\" }} }},
        ],
      }};
      Plotly.react(divId, [trace], layout, {{displayModeBar: true, responsive: true}});
    }}

    function hist(divId, title, histObj, xTitle, color) {{
      const edges = histObj.edges || [];
      const counts = histObj.counts || [];
      const xs = [];
      for (let i = 0; i < edges.length - 1; i++) xs.push(0.5*(edges[i] + edges[i+1]));
      const trace = {{
        type: \"bar\",
        x: xs,
        y: counts,
        marker: {{ color }},
        hovertemplate: \"x=%{x:.4f}<br>count=%{y}<extra></extra>\",
      }};
      const layout = {{
        ...PLOT_THEME,
        title: {{ text: title, font: {{ color: \"#e5e7eb\", size: 13 }} }},
        xaxis: {{ ...PLOT_THEME.xaxis, title: xTitle }},
        yaxis: {{ ...PLOT_THEME.yaxis, title: \"count\" }},
      }};
      Plotly.react(divId, [trace], layout, {{displayModeBar: true, responsive: true}});
    }}

    function scatter(divId, title, samples, color) {{
      const p = samples?.p || [];
      const h = samples?.H || [];
      const n = Math.min(p.length, h.length, 25000);
      if (!n) {{
        Plotly.purge(divId);
        return;
      }}
      const trace = {{
        type: \"scattergl\",
        mode: \"markers\",
        x: p.slice(0, n),
        y: h.slice(0, n),
        marker: {{ size: 2, opacity: 0.35, color }},
        hovertemplate: \"p=%{x:.4f}<br>H=%{y:.4f}<extra></extra>\",
      }};
      const layout = {{
        ...PLOT_THEME,
        title: {{ text: title, font: {{ color: \"#e5e7eb\", size: 13 }} }},
        xaxis: {{ ...PLOT_THEME.xaxis, title: DATA.meta.axes?.x || \"p_t\" }},
        yaxis: {{ ...PLOT_THEME.yaxis, title: DATA.meta.axes?.y || \"H\" }},
      }};
      Plotly.react(divId, [trace], layout, {{displayModeBar: true, responsive: true}});
    }}

    function buildPackRows(seqLen, leftId, rightId) {{
      const rows = [];
      for (const pack of (DATA.meta.pack_names || [])) {{
        const pair = computePair(leftId, rightId, pack, seqLen);
        if (!pair) continue;
        rows.push({{
          pack,
          deltaPpl: pair.deltaPpl,
          ccDeltaPP: pair.ccDelta * 100.0,
          js2d: pair.js2d,
          meanPDropPP: pair.meanPDropPP,
        }});
      }}
      const zDelta = zScores(rows.map(r => r.deltaPpl));
      const zCC = zScores(rows.map(r => r.ccDeltaPP));
      const zJS = zScores(rows.map(r => r.js2d));
      const zMeanP = zScores(rows.map(r => r.meanPDropPP));
      rows.forEach((r, i) => {{ r.z = zDelta[i] + zCC[i] + zJS[i] + zMeanP[i]; }});
      const zAll = zScores(rows.map(r => r.z));
      rows.forEach((r, i) => {{ r.z_norm = zAll[i]; }});
      return rows;
    }}

    function buildLeaderboard(seqLen) {{
      const packs = DATA.meta.pack_names || [];
      const rows = [];
      for (const modelId of MODELS) {{
        let ok = 0;
        let pplSum = 0;
        let ccSum = 0;
        let meanPSum = 0;
        for (const pack of packs) {{
          const s = getSeq(modelId, pack, seqLen);
          if (!s) continue;
          ok += 1;
          const m = s.model || {{}};
          pplSum += (m.ppl || 0);
          ccSum += (m.cc_rate || 0);
          meanPSum += (m.mean_prob || 0);
        }}
        if (!ok) continue;
        rows.push({{
          model: modelId,
          ppl: pplSum / ok,
          cc: ccSum / ok,
          meanP: meanPSum / ok,
          packs_ok: ok,
        }});
      }}

      const zppl = zScores(rows.map(r => r.ppl));
      const zcc = zScores(rows.map(r => r.cc));
      const zmeanP = zScores(rows.map(r => r.meanP));
      rows.forEach((r, i) => {{
        r.quality_z = (-zppl[i]) + (-zcc[i]) + (zmeanP[i]);
      }});
      rows.sort((a, b) => b.quality_z - a.quality_z);
      return rows;
    }}

    function renderLeaderboard(seqLen) {{
      const rows = buildLeaderboard(seqLen);
      if (!rows.length) {{
        document.getElementById(\"leaderTable\").innerHTML = \"<div class='footer'>No data for this seq_len.</div>\";
        return rows;
      }}
      let html = \"<table><thead><tr>\" +
        \"<th class='left'>Rank</th>\" +
        \"<th class='left'>Model</th>\" +
        \"<th>PPL</th><th>CC rate</th><th>Mean p_t</th><th>Packs</th><th>Quality z</th>\" +
        \"</tr></thead><tbody>\";
      rows.forEach((r, i) => {{
        const z = r.quality_z;
        const klass = z >= 1.0 ? \"good\" : (z <= -1.0 ? \"bad\" : \"warn\");
        html += `<tr>
          <td>${i+1}</td>
          <td class='left mono'>${r.model}</td>
          <td>${fmt(r.ppl, 4)}</td>
          <td>${fmt(r.cc*100.0, 3)}%</td>
          <td>${fmt(r.meanP, 4)}</td>
          <td>${r.packs_ok}</td>
          <td class='${klass}'>${fmt(z, 3)}</td>
        </tr>`;
      }});
      html += \"</tbody></table>\";
      document.getElementById(\"leaderTable\").innerHTML = html;
      document.getElementById(\"leaderExplain\").textContent =
        \"Quality z is computed per seq_len across models: higher is better (lower PPL + lower Confident-Conflict rate + higher mean p_t). Heatmap log1p(count)=log(1+count) reveals sparse + dense regions.\";
      return rows;
    }}

    function downloadCSV(filename, rows) {{
      if (!rows || !rows.length) return;
      const header = Object.keys(rows[0]).join(\",\");
      const lines = rows.map(r => Object.keys(r).map(k => {{
        const v = r[k];
        if (v === null || v === undefined) return \"\";
        if (typeof v === \"number\") return String(v);
        return `\"${String(v).replaceAll('\"','\"\"')}\"`;
      }}).join(\",\"));
      const csv = [header, ...lines].join(\"\\n\");
      const blob = new Blob([csv], {{type: \"text/csv;charset=utf-8\"}});
      const url = URL.createObjectURL(blob);
      const a = document.createElement(\"a\");
      a.href = url;
      a.download = filename;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }}

    function renderHero(pair, packRows, selectedPack) {{
      const row = packRows.find(r => r.pack === selectedPack);
      const z = row ? row.z_norm : 0;
      const pval = pFromZ(z);
      const tail = pval / 2.0;
      const percentile = (1.0 - tail) * 100.0;

      const worse = (pair.deltaPpl > EPS.ppl) && (pair.ccDelta*100.0 > EPS.cc_pp) && (pair.js2d > EPS.js2d) && (z >= 2.0) && (pval < 0.05);
      const better = (pair.deltaPpl < -EPS.ppl) && (pair.ccDelta*100.0 < -EPS.cc_pp) && (z <= -2.0) && (pval < 0.05);
      let verdict = \"MIXED / INCONCLUSIVE\";
      let klass = \"warn\";
      if (worse) {{ verdict = \"DEFINITIVELY WORSE\"; klass = \"bad\"; }}
      if (better) {{ verdict = \"DEFINITIVELY BETTER\"; klass = \"good\"; }}

      document.getElementById(\"verdictTitle\").textContent = `Verdict for pack: ${selectedPack}`;
      document.getElementById(\"verdictBig\").innerHTML = `<span class='${klass}'>${verdict}</span>`;
      const ordering = worse ? \"Ordering: Left > Right\" : (better ? \"Ordering: Right > Left\" : \"Ordering: unclear (mixed signals)\");
      document.getElementById(\"verdictExplain\").innerHTML =
        `Degradation Score z=${fmt(z,2)} (two-tailed p≈${fmt(pval,4)}, percentile≈${fmt(percentile,1)}%).<br/>` +
        `${ordering}<br/>` +
        `Definition: z-score over ΔPPL, CC Δ, JS2D, and Δ mean p (Right vs Left).`;

      const stats = [
        [\"ΔPPL\", fmtSigned(pair.deltaPpl, 3), \"Right − Left (completion-only)\"],
        [\"CC Δ (pp)\", fmtSigned(pair.ccDelta*100.0, 2), \"Under Left CC thresholds\"],
        [\"JS2D\", fmt(pair.js2d, 4), \"Landscape divergence\"],
        [\"Δ mean p (pp)\", fmtSigned(pair.meanPDropPP, 2), \"Drop in mean p_t\"],
        [\"Left PPL\", fmt(pair.L.ppl, 3), pair.leftSeq ? `rows_seen=${pair.leftSeq.rows_seen}` : \"\"],
        [\"Right PPL\", fmt(pair.R.ppl, 3), pair.rightSeq ? `rows_seen=${pair.rightSeq.rows_seen}` : \"\"],
        [\"Left tok/s\", fmt(pair.L.tok_s_pred, 1), \"Score throughput proxy\"],
        [\"Right tok/s\", fmt(pair.R.tok_s_pred, 1), \"Score throughput proxy\"],
      ];
      const hero = document.getElementById(\"heroStats\");
      hero.innerHTML = \"\";
      for (const [k, v, s] of stats) {{
        const el = document.createElement(\"div\");
        el.className = \"stat\";
        el.innerHTML = `<div class='k'>${k}</div><div class='v mono'>${v}</div><div class='s'>${s}</div>`;
        hero.appendChild(el);
      }}
    }}

    function renderTable(pair) {{
      const rows = [
        [\"PPL\", pair.L.ppl, pair.R.ppl, pair.R.ppl - pair.L.ppl],
        [\"mean p_t\", pair.L.mean_prob, pair.R.mean_prob, pair.R.mean_prob - pair.L.mean_prob],
        [\"mean H\", pair.L.mean_entropy, pair.R.mean_entropy, pair.R.mean_entropy - pair.L.mean_entropy],
        [\"CC rate (left thr)\", pair.ccL, pair.ccR, pair.ccDelta],
        [\"kept tokens\", pair.L.kept_tokens, pair.R.kept_tokens, pair.R.kept_tokens - pair.L.kept_tokens],
      ];
      let html = \"<table><tr><th class='left'>metric</th><th>left</th><th>right</th><th>Δ (right-left)</th></tr>\";
      for (const r of rows) {{
        const dv = Number(r[3]);
        html += `<tr><td class='left'>${r[0]}</td><td class='mono'>${fmt(r[1])}</td><td class='mono'>${fmt(r[2])}</td><td class='mono ${deltaClass(dv)}'>${fmtSigned(dv)}</td></tr>`;
      }}
      html += \"</table>\";
      document.getElementById(\"compareTable\").innerHTML = html;
    }}

    function populate() {{
      document.getElementById(\"datasetRepo\").textContent = DATA.meta.dataset_repo || \"\";
      document.getElementById(\"packsInfo\").textContent = (DATA.meta.pack_names || []).join(\", \");
      document.getElementById(\"axesInfo\").textContent = `${DATA.meta.axes?.x || \"p_t\"} vs ${DATA.meta.axes?.y || \"H\"}`;

      const leftSel = document.getElementById(\"leftModel\");
      const rightSel = document.getElementById(\"rightModel\");
      const packSel = document.getElementById(\"packSel\");
      const seqSel = document.getElementById(\"seqSel\");
      const leaderSeq = document.getElementById(\"leaderSeq\");
      leftSel.innerHTML = \"\"; rightSel.innerHTML = \"\"; packSel.innerHTML = \"\"; seqSel.innerHTML = \"\"; leaderSeq.innerHTML = \"\";

      MODELS.forEach((m) => {{
        const optL = document.createElement(\"option\"); optL.value = m; optL.textContent = m; leftSel.appendChild(optL);
        const optR = document.createElement(\"option\"); optR.value = m; optR.textContent = m; rightSel.appendChild(optR);
      }});
      for (const p of (DATA.meta.pack_names || [])) {{
        const opt = document.createElement(\"option\"); opt.value = p; opt.textContent = p; packSel.appendChild(opt);
      }}
      for (const s of (DATA.meta.seq_lens || [])) {{
        const opt = document.createElement(\"option\"); opt.value = String(s); opt.textContent = String(s); seqSel.appendChild(opt);
      }}
      for (const s of (DATA.meta.seq_lens || [])) {{
        const opt = document.createElement(\"option\"); opt.value = String(s); opt.textContent = String(s); leaderSeq.appendChild(opt);
      }}

      if (MODELS.length >= 2) {{ leftSel.value = MODELS[0]; rightSel.value = MODELS[1]; }}
      if ((DATA.meta.pack_names || []).includes(\"UNION\")) packSel.value = \"UNION\";
      if ((DATA.meta.seq_lens || []).includes(\"131072\")) seqSel.value = \"131072\";
      leaderSeq.value = seqSel.value;

      document.getElementById(\"swapBtn\").onclick = () => {{
        const a = leftSel.value; leftSel.value = rightSel.value; rightSel.value = a;
        render();
      }};
      leftSel.onchange = render;
      rightSel.onchange = render;
      packSel.onchange = render;
      seqSel.onchange = () => {{
        leaderSeq.value = seqSel.value;
        render();
      }};
      leaderSeq.onchange = render;

      document.getElementById(\"downloadCsvBtn\").onclick = () => {{
        const seq = leaderSeq.value;
        const rows = buildLeaderboard(seq);
        downloadCSV(`eaft_leaderboard_seq${seq}.csv`, rows);
      }};

      const heatSel = document.getElementById(\"heatScale\");
      if (heatSel) {{
        heatSel.value = UI.heatScale;
        heatSel.onchange = () => {{
          UI.heatScale = heatSel.value || \"log1p\";
          render();
        }};
      }}
    }}

    function render() {{
      try {{
        if (typeof Plotly === \"undefined\") {{
          showError(\"Plotly is undefined (failed to load). If you opened this via file:// and your browser blocks large inline scripts, try serving the `reports/` folder via a local webserver.\");\n
          return;
        }}

        const leftId = document.getElementById(\"leftModel\").value;
        const rightId = document.getElementById(\"rightModel\").value;
        const packName = document.getElementById(\"packSel\").value;
        const seqLen = document.getElementById(\"seqSel\").value;
        const pair = computePair(leftId, rightId, packName, seqLen);
        if (!pair) {{
          showError(\"No data for selected (model, pack, seq_len). Pick another pack/seq_len or verify the JSON contains it.\");\n
          return;
        }}

        document.getElementById(\"leftTitle\").textContent = `Left: ${leftId}`;
        document.getElementById(\"rightTitle\").textContent = `Right: ${rightId}`;

        const packRows = buildPackRows(seqLen, leftId, rightId);
        renderHero(pair, packRows, packName);
        renderTable(pair);

        const scaleLabel = (UI.heatScale === \"count\") ? \"count\" : \"log1p(count)\";
        heatmap(\"leftHeat\", `Left density (${scaleLabel})`, pair.zL, pair.xEdges, pair.yEdges, pair.L.x_thr, pair.L.h_thr, false);
        heatmap(\"rightHeat\", `Right density (${scaleLabel})`, pair.zR, pair.xEdges, pair.yEdges, pair.L.x_thr, pair.L.h_thr, false);
        const delta = pair.zR.map((row, y) => row.map((v, x) => v - pair.zL[y][x]));
        heatmap(\"deltaHeat\", \"Δ density (Right − Left)\", delta, pair.xEdges, pair.yEdges, pair.L.x_thr, pair.L.h_thr, true);

        hist(\"leftHistP\", \"Left: p_t histogram\", pair.L.hist1d_x, DATA.meta.axes?.x || \"p_t\", \"#60a5fa\");
        hist(\"leftHistH\", \"Left: H histogram\", pair.L.hist1d_H, DATA.meta.axes?.y || \"H\", \"#34d399\");
        hist(\"rightHistP\", \"Right: p_t histogram\", pair.R.hist1d_x, DATA.meta.axes?.x || \"p_t\", \"#fb7185\");
        hist(\"rightHistH\", \"Right: H histogram\", pair.R.hist1d_H, DATA.meta.axes?.y || \"H\", \"#fbbf24\");

        scatter(\"leftScatter\", \"Left: sample points (hover/zoom)\", pair.L.samples, \"#60a5fa\");
        scatter(\"rightScatter\", \"Right: sample points (hover/zoom)\", pair.R.samples, \"#fb7185\");

        // Leaderboard
        renderLeaderboard(document.getElementById(\"leaderSeq\").value || seqLen);
      }} catch (e) {{
        showError(e && e.stack ? e.stack : String(e));
      }}
    }}

    populate();
    render();
  </script>
</body>
</html>"""
    # NOTE: This template was originally written with doubled braces (`{{` / `}}`)
    # (common in Python `.format` templates). Since we are NOT using `.format`,
    # those braces must be normalized back to single braces, otherwise the
    # embedded JS becomes invalid (e.g. `Object.keys(x || {{}})`).
    html = html.replace("{{", "{").replace("}}", "}")
    return html.replace("__EAFT_DATA_JSON__", data_json).replace("__PLOTLY_MIN_JS__", plotly_js)


def _render_html(payload: dict) -> str:
    # Legacy canvas-based renderer (kept for reference). Avoid f-strings here
    # because the embedded JS uses `${...}` template literals and object braces.
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EAFT model comparison — dynamic</title>
  <style>
    :root {{
      --bg: #0b0f14;
      --panel: #0f172a;
      --panel-2: #111827;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --grid: #1f2937;
      --accent: #f97316;
      --good: #34d399;
      --bad: #f87171;
      --plot-bg: #0b0f14;
      --plot-axis: #cbd5e1;
    }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: "IBM Plex Sans","Space Grotesk",ui-sans-serif,system-ui; margin: 24px; color: var(--text); background: var(--bg); }}
    h1 {{ margin: 0 0 8px 0; font-size: 22px; letter-spacing: 0.4px; }}
    h2 {{ margin: 0 0 12px 0; font-size: 16px; }}
    .meta {{ color: var(--muted); font-size: 13px; line-height: 1.4; }}
    .row {{ display: grid; gap: 16px; align-items: stretch; }}
    .grid-2 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .panel {{ border: 1px solid var(--grid); border-radius: 12px; padding: 14px; background: var(--panel); min-width: 0; }}
    .panel.hero {{ min-width: 100%; background: linear-gradient(135deg, #0f172a, #0b1220); }}
    .controls {{ display: flex; gap: 16px; align-items: center; flex-wrap: wrap; margin: 12px 0 16px 0; min-width: 100%; }}
    select, button {{ padding: 6px 10px; background: #0b1220; color: var(--text); border: 1px solid var(--grid); border-radius: 8px; }}
    button {{ cursor: pointer; }}
    canvas {{ border: 1px solid var(--grid); border-radius: 8px; background: var(--plot-bg); width: 100%; height: auto; }}
    canvas.heatmap {{ height: clamp(360px, 55vh, 720px); }}
    canvas.hist {{ height: clamp(240px, 30vh, 360px); }}
    .small {{ font-size: 12px; color: var(--muted); }}
    table {{ border-collapse: collapse; font-size: 12px; width: 100%; }}
    td, th {{ border: 1px solid var(--grid); padding: 8px 10px; text-align: right; }}
    th {{ background: #0b1220; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; font-size: 11px; }}
    .left {{ text-align: left; }}
    .legend {{ display: flex; gap: 12px; align-items: center; }}
    .swatch {{ width: 240px; height: 12px; border-radius: 999px; background: linear-gradient(90deg, #1b1f3a, #234f9a, #25a18e, #e9c46a, #f4a261, #e76f51); border: 1px solid var(--grid); }}
    .hint {{ color: var(--muted); font-size: 12px; }}
    .hero-grid {{ display: grid; grid-template-columns: repeat(5, minmax(160px, 1fr)); gap: 12px; }}
    .stat {{ background: var(--panel-2); border: 1px solid var(--grid); border-radius: 12px; padding: 12px; }}
    .stat .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .stat .value {{ font-size: 26px; font-weight: 700; margin-top: 6px; }}
    .stat .sub {{ font-size: 11px; color: var(--muted); margin-top: 4px; }}
    .bad {{ color: var(--bad); }}
    .good {{ color: var(--good); }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }}
  </style>
</head>
<body>
  <h1>EAFT diagnostics — dynamic 2‑model comparison</h1>
  <div class="meta" id="meta"></div>

  <div class="panel hero">
    <h2>Quality Verdict (selected pack)</h2>
    <div class="hero-grid" id="heroStats"></div>
    <div class="small" id="heroNote"></div>
  </div>

  <div class="panel" style="margin-top: 16px; min-width:100%;">
    <h2>Leaderboard (global averages)</h2>
    <div class="small">Averages across all packs and seq_len values present in each model file.</div>
    <div id="leaderboardTable"></div>
  </div>

  <div class="panel" style="margin-top: 16px; min-width:100%;">
    <h2>Overall Ordering (pairwise wins)</h2>
    <div class="small">Pairwise wins are computed across shared packs/seq_lens using thresholds for “definitive” deltas.</div>
    <div id="orderingTable"></div>
  </div>

  <div class="controls panel">
    <div><span class="small">Left model</span><br/><select id="leftModel"></select></div>
    <div><span class="small">Right model</span><br/><select id="rightModel"></select></div>
    <div><span class="small">Pack</span><br/><select id="packSel"></select></div>
    <div><span class="small">seq_len</span><br/><select id="seqSel"></select></div>
    <div><span class="small">Actions</span><br/><button id="swapBtn">Swap</button></div>
    <div class="legend">
      <div class="swatch"></div>
      <div class="hint">Heatmap shows log(count+1) density. CC thresholds from left model.</div>
    </div>
  </div>

  <div class="row grid-2">
    <div class="panel">
      <h3 id="leftTitle">Left</h3>
      <canvas id="cLeft" class="heatmap"></canvas>
      <div class="small" id="leftStats"></div>
      <div class="small mono" id="leftHover"></div>
    </div>
    <div class="panel">
      <h3 id="rightTitle">Right</h3>
      <canvas id="cRight" class="heatmap"></canvas>
      <div class="small" id="rightStats"></div>
      <div class="small mono" id="rightHover"></div>
    </div>
  </div>

  <div class="panel" style="margin-top: 16px;">
    <h3>Δ Density (right - left)</h3>
    <canvas id="cDelta" class="heatmap"></canvas>
    <div class="small" id="deltaStats"></div>
    <div class="small mono" id="deltaHover"></div>
  </div>

  <div class="row grid-2" style="margin-top: 16px;">
    <div class="panel">
      <h3>Histograms (Left)</h3>
      <canvas id="hLeftX" class="hist"></canvas>
      <div class="small mono" id="hLeftXHover"></div>
      <canvas id="hLeftH" class="hist" style="margin-top:10px;"></canvas>
      <div class="small mono" id="hLeftHHover"></div>
    </div>
    <div class="panel">
      <h3>Histograms (Right)</h3>
      <canvas id="hRightX" class="hist"></canvas>
      <div class="small mono" id="hRightXHover"></div>
      <canvas id="hRightH" class="hist" style="margin-top:10px;"></canvas>
      <div class="small mono" id="hRightHHover"></div>
    </div>
  </div>

  <div class="panel" style="margin-top: 16px;">
    <h2>Left vs Right Metrics</h2>
    <div id="compareTable"></div>
  </div>

  <script id="DATA" type="application/json">{data}</script>
  <script>
    const DATA = JSON.parse(document.getElementById("DATA").textContent);
    const MODELS = Object.keys(DATA.models || {{}});
    const EPS = {{ ppl: 0.10, cc: 0.001, meanp: 0.001 }};

    function fmt(x, d=4) {{
      if (x === null || x === undefined || Number.isNaN(x)) return "—";
      return Number(x).toFixed(d);
    }}
    function fmtSigned(x, d=4) {{
      if (x === null || x === undefined || Number.isNaN(x)) return "—";
      const v = Number(x);
      return (v >= 0 ? "+" : "") + v.toFixed(d);
    }}
    function pFromZ(z) {{
      const x = Math.abs(z) / Math.SQRT2;
      const t = 1 / (1 + 0.3275911 * x);
      const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429;
      const erf = 1 - (((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t) * Math.exp(-x*x);
      return 1 - erf;
    }}
    function deltaClass(v) {{
      if (v === null || v === undefined || Number.isNaN(v)) return "";
      if (v > 0) return "bad";
      if (v < 0) return "good";
      return "";
    }}
    function cssVar(name, fallback) {{
      const val = getComputedStyle(document.body).getPropertyValue(name).trim();
      return val || fallback;
    }}
    const THEME = {{
      bg: cssVar("--plot-bg", "#0b0f14"),
      axis: cssVar("--plot-axis", "#cbd5e1"),
      text: cssVar("--text", "#e5e7eb"),
      grid: cssVar("--grid", "#1f2937"),
    }};
    function prepareCanvas(canvas) {{
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const maxW = 1600, maxH = 1200;
      const w = Math.min(maxW, Math.max(1, rect.width));
      const h = Math.min(maxH, Math.max(1, rect.height));
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {{
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }}
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return {{ ctx, W: w, H: h }};
    }}
    function colorRamp(t) {{
      const stops = [
        [0.00, [12, 18, 36]],
        [0.25, [25, 72, 160]],
        [0.45, [37, 168, 154]],
        [0.65, [234, 204, 120]],
        [0.85, [245, 153, 94]],
        [1.00, [239, 83, 80]],
      ];
      t = Math.min(1, Math.max(0, t));
      for (let i = 0; i < stops.length - 1; i++) {{
        const a = stops[i], b = stops[i+1];
        if (t >= a[0] && t <= b[0]) {{
          const u = (t - a[0]) / (b[0] - a[0] + 1e-12);
          const rgb = [
            Math.round(a[1][0] + u*(b[1][0]-a[1][0])),
            Math.round(a[1][1] + u*(b[1][1]-a[1][1])),
            Math.round(a[1][2] + u*(b[1][2]-a[1][2])),
          ];
          return `rgb(${{rgb[0]}},${{rgb[1]}},${{rgb[2]}})`;
        }}
      }}
      return "rgb(0,0,0)";
    }}
    function drawHeatmap(canvas, hist, thresholds, title, extraThresholds=null, samples=null) {{
      const {{ ctx, W, H }} = prepareCanvas(canvas);
      const padL = 44, padR = 10, padT = 16, padB = 36;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = THEME.bg;
      ctx.fillRect(0, 0, W, H);
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;
      const xbins = hist.xbins, ybins = hist.ybins;
      const counts = hist.counts;
      let maxLog = 0;
      for (let i = 0; i < counts.length; i++) {{
        const v = Math.log1p(counts[i]);
        if (v > maxLog) maxLog = v;
      }}
      maxLog = Math.max(1e-12, maxLog);
      for (let ix = 0; ix < xbins; ix++) {{
        for (let iy = 0; iy < ybins; iy++) {{
          const idx = ix*ybins + iy;
          const v = Math.log1p(counts[idx]) / maxLog;
          ctx.fillStyle = colorRamp(v);
          const x0 = padL + (ix / xbins) * plotW;
          const y0 = padT + ((ybins - 1 - iy) / ybins) * plotH;
          const x1 = padL + ((ix + 1) / xbins) * plotW;
          const y1 = padT + ((ybins - iy) / ybins) * plotH;
          ctx.fillRect(x0, y0, x1 - x0 + 0.5, y1 - y0 + 0.5);
        }}
      }}
      ctx.strokeStyle = THEME.axis;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, padT);
      ctx.lineTo(padL, padT + plotH);
      ctx.lineTo(padL + plotW, padT + plotH);
      ctx.stroke();
      ctx.fillStyle = THEME.text;
      ctx.font = "12px sans-serif";
      const xLabel = DATA.meta.axes.x || "p_t";
      ctx.fillText(xLabel, padL + plotW/2 - 22, H - 10);
      ctx.save();
      ctx.translate(14, padT + plotH/2 + 28);
      ctx.rotate(-Math.PI/2);
      ctx.fillText("H_topK/ln(K)", 0, 0);
      ctx.restore();
      if (thresholds) {{
        const xMin = DATA.meta.x_min, xMax = DATA.meta.x_max;
        const yMin = 0.0, yMax = 1.0;
        const xThr = Math.min(xMax, Math.max(xMin, thresholds.x_thr));
        const yThr = Math.min(yMax, Math.max(yMin, thresholds.h_thr));
        const x = padL + ((xThr - xMin) / (xMax - xMin)) * plotW;
        const y = padT + (1 - (yThr - yMin) / (yMax - yMin)) * plotH;
        ctx.strokeStyle = "rgba(226,232,240,0.9)";
        ctx.setLineDash([6,4]);
        ctx.beginPath();
        ctx.moveTo(x, padT);
        ctx.lineTo(x, padT + plotH);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(padL + plotW, y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "rgba(255,255,255,0.10)";
        ctx.fillRect(padL, y, x - padL, (padT + plotH) - y);
        ctx.strokeStyle = "rgba(255,255,255,0.35)";
        ctx.strokeRect(padL, y, x - padL, (padT + plotH) - y);
      }}
      if (extraThresholds) {{
        const xMin = DATA.meta.x_min, xMax = DATA.meta.x_max;
        const yMin = 0.0, yMax = 1.0;
        const xThr = Math.min(xMax, Math.max(xMin, extraThresholds.x_thr));
        const yThr = Math.min(yMax, Math.max(yMin, extraThresholds.h_thr));
        const x = padL + ((xThr - xMin) / (xMax - xMin)) * plotW;
        const y = padT + (1 - (yThr - yMin) / (yMax - yMin)) * plotH;
        ctx.strokeStyle = "rgba(255,255,255,0.7)";
        ctx.setLineDash([2,4]);
        ctx.beginPath();
        ctx.moveTo(x, padT);
        ctx.lineTo(x, padT + plotH);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(padL + plotW, y);
        ctx.stroke();
        ctx.setLineDash([]);
      }}
      if (title) {{
        ctx.fillStyle = THEME.text;
        ctx.font = "bold 12px sans-serif";
        ctx.fillText(title, padL, 12);
      }}
      if (samples && samples.length) {{
        ctx.fillStyle = "rgba(255,255,255,0.35)";
        const xMin = DATA.meta.x_min, xMax = DATA.meta.x_max;
        const yMin = 0.0, yMax = 1.0;
        for (let i = 0; i < samples.length; i++) {{
          const sx = samples[i][0];
          const sy = samples[i][1];
          const x = padL + ((sx - xMin) / (xMax - xMin)) * plotW;
          const y = padT + (1 - (sy - yMin) / (yMax - yMin)) * plotH;
          ctx.fillRect(x, y, 1.5, 1.5);
        }}
      }}
    }}
    function drawDeltaHeatmap(canvas, histL, histR, thresholds, title) {{
      const {{ ctx, W, H }} = prepareCanvas(canvas);
      const padL = 44, padR = 10, padT = 16, padB = 36;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = THEME.bg;
      ctx.fillRect(0, 0, W, H);
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;
      const xbins = histL.xbins, ybins = histL.ybins;
      const a = histL.counts;
      const b = histR.counts;
      let maxAbs = 0;
      const diff = new Array(a.length);
      for (let i = 0; i < a.length; i++) {{
        const v = Math.log1p(b[i]) - Math.log1p(a[i]);
        diff[i] = v;
        maxAbs = Math.max(maxAbs, Math.abs(v));
      }}
      maxAbs = Math.max(1e-9, maxAbs);
      function diverging(t) {{
        const u = (t + 1) / 2;
        const r = Math.round(220 * u + 20);
        const b = Math.round(220 * (1 - u) + 20);
        const g = Math.round(220 * (1 - Math.abs(t)) + 20);
        return `rgb(${{r}},${{g}},${{b}})`;
      }}
      for (let ix = 0; ix < xbins; ix++) {{
        for (let iy = 0; iy < ybins; iy++) {{
          const idx = ix*ybins + iy;
          const v = diff[idx] / maxAbs;
          ctx.fillStyle = diverging(v);
          const x0 = padL + (ix / xbins) * plotW;
          const y0 = padT + ((ybins - 1 - iy) / ybins) * plotH;
          const x1 = padL + ((ix + 1) / xbins) * plotW;
          const y1 = padT + ((ybins - iy) / ybins) * plotH;
          ctx.fillRect(x0, y0, x1 - x0 + 0.5, y1 - y0 + 0.5);
        }}
      }}
      ctx.strokeStyle = THEME.axis;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, padT);
      ctx.lineTo(padL, padT + plotH);
      ctx.lineTo(padL + plotW, padT + plotH);
      ctx.stroke();
      ctx.fillStyle = THEME.text;
      ctx.font = "12px sans-serif";
      const xLabel = DATA.meta.axes.x || "p_t";
      ctx.fillText(xLabel, padL + plotW/2 - 22, H - 10);
      ctx.save();
      ctx.translate(14, padT + plotH/2 + 28);
      ctx.rotate(-Math.PI/2);
      ctx.fillText("H_topK/ln(K)", 0, 0);
      ctx.restore();
      if (thresholds) {{
        const xMin = DATA.meta.x_min, xMax = DATA.meta.x_max;
        const yMin = 0.0, yMax = 1.0;
        const xThr = Math.min(xMax, Math.max(xMin, thresholds.x_thr));
        const yThr = Math.min(yMax, Math.max(yMin, thresholds.h_thr));
        const x = padL + ((xThr - xMin) / (xMax - xMin)) * plotW;
        const y = padT + (1 - (yThr - yMin) / (yMax - yMin)) * plotH;
        ctx.strokeStyle = "rgba(226,232,240,0.9)";
        ctx.setLineDash([6,4]);
        ctx.beginPath();
        ctx.moveTo(x, padT);
        ctx.lineTo(x, padT + plotH);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(padL + plotW, y);
        ctx.stroke();
        ctx.setLineDash([]);
      }}
      if (title) {{
        ctx.fillStyle = THEME.text;
        ctx.font = "bold 12px sans-serif";
        ctx.fillText(title, padL, 12);
      }}
    }}
    function drawHist(canvas, hist, thresholds, label) {{
      const {{ ctx, W, H }} = prepareCanvas(canvas);
      const padL = 44, padR = 10, padT = 10, padB = 20;
      ctx.clearRect(0,0,W,H);
      ctx.fillStyle = THEME.bg; ctx.fillRect(0,0,W,H);
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;
      const counts = hist.counts;
      let maxC = 1;
      for (const c of counts) maxC = Math.max(maxC, c);
      const n = counts.length;
      for (let i = 0; i < n; i++) {{
        const x0 = padL + (i / n) * plotW;
        const x1 = padL + ((i + 1) / n) * plotW;
        const h = (counts[i] / maxC) * plotH;
        ctx.fillStyle = "rgba(76, 146, 219, 0.85)";
        ctx.fillRect(x0, padT + (plotH - h), Math.max(1, x1 - x0 - 0.5), h);
      }}
      ctx.strokeStyle=THEME.axis; ctx.beginPath();
      ctx.moveTo(padL, padT); ctx.lineTo(padL, padT+plotH); ctx.lineTo(padL+plotW, padT+plotH); ctx.stroke();
      ctx.fillStyle=THEME.text; ctx.font="12px sans-serif";
      ctx.fillText(label, padL, H-4);
      if (thresholds && thresholds.value !== null && thresholds.value !== undefined) {{
        const vmin = hist.edges[0], vmax = hist.edges[hist.edges.length-1];
        const x = padL + ((thresholds.value - vmin) / (vmax - vmin)) * plotW;
        ctx.strokeStyle="rgba(226,232,240,0.9)";
        ctx.setLineDash([6,4]);
        ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT+plotH); ctx.stroke();
        ctx.setLineDash([]);
      }}
    }}
    function attachHistHover(canvas, hist, label, outId) {{
      const out = document.getElementById(outId);
      if (!out) return;
      canvas.onmousemove = (ev) => {{
        const rect = canvas.getBoundingClientRect();
        const x = (ev.clientX - rect.left) / rect.width;
        if (x < 0 || x > 1) {{
          out.textContent = "";
          return;
        }}
        const bins = hist.counts.length;
        const ix = Math.min(bins - 1, Math.max(0, Math.floor(x * bins)));
        const count = hist.counts[ix] || 0;
        const x0 = hist.edges[ix];
        const x1 = hist.edges[ix + 1];
        out.textContent = `${{label}}: bin [${{x0.toFixed(4)}},${{x1.toFixed(4)}}] count=${{count}}`;
      }};
      canvas.onmouseleave = () => {{ out.textContent = ""; }};
    }}
    function attachHeatmapHover(canvas, hist, label, outId, samples=null) {{
      const out = document.getElementById(outId);
      if (!out) return;
      canvas.onmousemove = (ev) => {{
        const rect = canvas.getBoundingClientRect();
        const x = (ev.clientX - rect.left) / rect.width;
        const y = (ev.clientY - rect.top) / rect.height;
        if (x < 0 || x > 1 || y < 0 || y > 1) {{
          out.textContent = "";
          return;
        }}
        const xbins = hist.xbins, ybins = hist.ybins;
        const ix = Math.min(xbins - 1, Math.max(0, Math.floor(x * xbins)));
        const iy = Math.min(ybins - 1, Math.max(0, Math.floor((1 - y) * ybins)));
        const idx = ix * ybins + iy;
        const count = hist.counts[idx] || 0;
        const x0 = hist.x_edges[ix];
        const x1 = hist.x_edges[ix + 1];
        const y0 = hist.y_edges[iy];
        const y1 = hist.y_edges[iy + 1];
        let nearest = "";
        if (samples && samples.length) {{
          const xMin = hist.x_edges[0];
          const xMax = hist.x_edges[hist.x_edges.length - 1];
          const yMin = hist.y_edges[0];
          const yMax = hist.y_edges[hist.y_edges.length - 1];
          let best = null;
          let bestD = 1e9;
          for (let i = 0; i < samples.length; i++) {{
            const sx = (samples[i][0] - xMin) / (xMax - xMin + 1e-12);
            const sy = 1 - (samples[i][1] - yMin) / (yMax - yMin + 1e-12);
            const dx = sx - x;
            const dy = sy - y;
            const d = dx*dx + dy*dy;
            if (d < bestD) {{ bestD = d; best = samples[i]; }}
          }}
          if (best) {{
            nearest = ` | nearest=(${{best[0].toFixed(4)}},${{best[1].toFixed(4)}})`;
          }}
        }}
        out.textContent = `${{label}}: x∈[${{x0.toFixed(3)}},${{x1.toFixed(3)}}] y∈[${{y0.toFixed(3)}},${{y1.toFixed(3)}}] count=${{count}}${{nearest}}`;
      }};
      canvas.onmouseleave = () => {{ out.textContent = ""; }};
    }}

    function jsDivergence(countsA, countsB) {{
      let sumA = 0, sumB = 0;
      for (let i = 0; i < countsA.length; i++) {{ sumA += countsA[i]; sumB += countsB[i]; }}
      if (sumA <= 0 || sumB <= 0) return 0;
      const eps = 1e-12;
      let js = 0;
      for (let i = 0; i < countsA.length; i++) {{
        const p = Math.max(eps, countsA[i] / sumA);
        const q = Math.max(eps, countsB[i] / sumB);
        const m = 0.5 * (p + q);
        js += 0.5 * (p * Math.log(p / m) + q * Math.log(q / m));
      }}
      return js;
    }}

    function ccRateFromHist(hist, thr) {{
      const xEdges = hist.x_edges, yEdges = hist.y_edges;
      const xbins = hist.xbins, ybins = hist.ybins;
      let total = 0, ll = 0;
      for (let ix = 0; ix < xbins; ix++) {{
        const x0 = xEdges[ix+0], x1 = xEdges[ix+1];
        const xMid = (x0 + x1) * 0.5;
        for (let iy = 0; iy < ybins; iy++) {{
          const y0 = yEdges[iy+0], y1 = yEdges[iy+1];
          const yMid = (y0 + y1) * 0.5;
          const idx = ix*ybins + iy;
          const c = hist.counts[idx] || 0;
          total += c;
          if (xMid <= thr.x_thr && yMid <= thr.h_thr) ll += c;
        }}
      }}
      return total > 0 ? (ll / total) : 0;
    }}

    function getModelSeq(modelId, pack, seq) {{
      const m = DATA.models[modelId];
      if (!m || !m.packs[pack] || !m.packs[pack][seq]) return null;
      return m.packs[pack][seq].model;
    }}

    function allModelPoints(modelId) {{
      const m = DATA.models[modelId];
      if (!m) return [];
      const points = [];
      for (const pack of Object.keys(m.packs)) {{
        const seqs = m.packs[pack];
        for (const seq of Object.keys(seqs)) {{
          const model = seqs[seq].model;
          if (model) points.push({{ pack, seq, model }});
        }}
      }}
      return points;
    }}

    function computeLeaderboard() {{
      const rows = [];
      for (const modelId of MODELS) {{
        const pts = allModelPoints(modelId);
        if (!pts.length) continue;
        let sumPpl = 0, sumCC = 0, sumMP = 0, sumH = 0;
        for (const p of pts) {{
          sumPpl += p.model.ppl;
          sumCC += p.model.cc_rate;
          sumMP += p.model.mean_prob;
          sumH += p.model.mean_entropy;
        }}
        const n = pts.length;
        rows.push({{
          modelId,
          n,
          avgPpl: sumPpl / n,
          avgCC: sumCC / n,
          avgMP: sumMP / n,
          avgH: sumH / n,
        }});
      }}
      rows.sort((a, b) => a.avgPpl - b.avgPpl);
      return rows;
    }}

    function sharedPoints(leftId, rightId) {{
      const left = DATA.models[leftId];
      const right = DATA.models[rightId];
      const points = [];
      if (!left || !right) return points;
      for (const pack of Object.keys(left.packs)) {{
        if (!right.packs[pack]) continue;
        for (const seq of Object.keys(left.packs[pack])) {{
          if (!right.packs[pack][seq]) continue;
          points.push({{ pack, seq, left: left.packs[pack][seq].model, right: right.packs[pack][seq].model }});
        }}
      }}
      return points;
    }}

    function compareModels(leftId, rightId) {{
      const pts = sharedPoints(leftId, rightId);
      let winsRight = 0, winsLeft = 0, mixed = 0;
      let sumDeltaPpl = 0, sumDeltaCC = 0, sumDeltaMP = 0;
      for (const p of pts) {{
        const left = p.left;
        const right = p.right;
        const leftThr = {{ x_thr: left.x_thr, h_thr: left.h_thr }};
        const leftCC = ccRateFromHist(left.hist2d_x_H, leftThr);
        const rightCC = ccRateFromHist(right.hist2d_x_H, leftThr);
        const deltaPpl = right.ppl - left.ppl;
        const deltaCC = rightCC - leftCC;
        const deltaMP = right.mean_prob - left.mean_prob;
        sumDeltaPpl += deltaPpl;
        sumDeltaCC += deltaCC;
        sumDeltaMP += deltaMP;
        const rightBetter =
          (deltaPpl < -EPS.ppl) &&
          (deltaCC < -EPS.cc) &&
          (deltaMP > EPS.meanp);
        const leftBetter =
          (deltaPpl > EPS.ppl) &&
          (deltaCC > EPS.cc) &&
          (deltaMP < -EPS.meanp);
        if (rightBetter) winsRight += 1;
        else if (leftBetter) winsLeft += 1;
        else mixed += 1;
      }}
      const n = Math.max(1, pts.length);
      return {{
        leftId,
        rightId,
        n,
        winsLeft,
        winsRight,
        mixed,
        avgDeltaPpl: sumDeltaPpl / n,
        avgDeltaCC: sumDeltaCC / n,
        avgDeltaMP: sumDeltaMP / n,
      }};
    }}

    function renderLeaderboard() {{
      const rows = computeLeaderboard();
      let html = "<table><tr><th class='left'>model</th><th>points</th><th>avg PPL</th><th>avg CC</th><th>avg mean_p</th><th>avg entropy</th></tr>";
      rows.forEach(r => {{
        html += `<tr><td class='left'>${{r.modelId}}</td><td>${{r.n}}</td><td>${{fmt(r.avgPpl,3)}}</td><td>${{fmt(r.avgCC,4)}}</td><td>${{fmt(r.avgMP,6)}}</td><td>${{fmt(r.avgH,4)}}</td></tr>`;
      }});
      html += "</table>";
      document.getElementById("leaderboardTable").innerHTML = html;
    }}

    function renderOrdering() {{
      const rows = [];
      for (let i = 0; i < MODELS.length; i++) {{
        for (let j = 0; j < MODELS.length; j++) {{
          if (i === j) continue;
          rows.push(compareModels(MODELS[i], MODELS[j]));
        }}
      }}
      const score = {{}};
      MODELS.forEach(m => score[m] = 0);
      rows.forEach(r => {{
        score[r.leftId] += r.winsLeft;
        score[r.rightId] += r.winsRight;
      }});
      const ordering = Object.keys(score)
        .map(m => ({{ modelId: m, score: score[m] }}))
        .sort((a, b) => b.score - a.score);
      let html = "<table><tr><th class='left'>model</th><th>pairwise wins</th></tr>";
      ordering.forEach(o => {{
        html += `<tr><td class='left'>${{o.modelId}}</td><td>${{o.score}}</td></tr>`;
      }});
      html += "</table>";
      document.getElementById("orderingTable").innerHTML = html;
    }}

    function populate() {{
      const leftSel = document.getElementById("leftModel");
      const rightSel = document.getElementById("rightModel");
      leftSel.innerHTML = ""; rightSel.innerHTML = "";
      MODELS.forEach((m, i) => {{
        const optL = document.createElement("option");
        optL.value = m; optL.textContent = m; leftSel.appendChild(optL);
        const optR = document.createElement("option");
        optR.value = m; optR.textContent = m; rightSel.appendChild(optR);
        if (i === 0) leftSel.value = m;
        if (i === 1) rightSel.value = m;
      }});
      if (MODELS.length === 1) rightSel.value = MODELS[0];

      const packSel = document.getElementById("packSel");
      packSel.innerHTML = "";
      DATA.meta.pack_names.forEach(p => {{
        const opt = document.createElement("option");
        opt.value = p; opt.textContent = p; packSel.appendChild(opt);
      }});

      const seqSel = document.getElementById("seqSel");
      seqSel.innerHTML = "";
      (DATA.meta.seq_lens || []).forEach(s => {{
        const opt = document.createElement("option");
        opt.value = String(s); opt.textContent = String(s); seqSel.appendChild(opt);
      }});
    }}

    function render() {{
      const leftId = document.getElementById("leftModel").value;
      const rightId = document.getElementById("rightModel").value;
      const pack = document.getElementById("packSel").value;
      const seq = document.getElementById("seqSel").value;
      const left = getModelSeq(leftId, pack, seq);
      const right = getModelSeq(rightId, pack, seq);
      if (!left || !right) return;

      document.getElementById("leftTitle").textContent = `Left: ${leftId}`;
      document.getElementById("rightTitle").textContent = `Right: ${rightId}`;
      document.getElementById("meta").textContent =
        `dataset=${DATA.meta.dataset_repo} | top_k=${DATA.meta.top_k || ""} | entropy_topk=${DATA.meta.entropy_topk} | cc_q=${DATA.meta.cc_quantile} | seq_len=${seq}`;

      const leftThr = {{x_thr: left.x_thr, h_thr: left.h_thr}};
      drawHeatmap(document.getElementById("cLeft"), left.hist2d_x_H, leftThr, `ppl=${fmt(left.ppl,3)} CC=${fmt(left.cc_rate,4)}`, null, left.samples || []);
      drawHeatmap(document.getElementById("cRight"), right.hist2d_x_H, leftThr, `ppl=${fmt(right.ppl,3)} CC=${fmt(right.cc_rate,4)}`, {{x_thr: right.x_thr, h_thr: right.h_thr}}, right.samples || []);
      drawDeltaHeatmap(document.getElementById("cDelta"), left.hist2d_x_H, right.hist2d_x_H, leftThr, `Δ log1p(density)`);
      attachHeatmapHover(document.getElementById("cLeft"), left.hist2d_x_H, "left", "leftHover", left.samples || []);
      attachHeatmapHover(document.getElementById("cRight"), right.hist2d_x_H, "right", "rightHover", right.samples || []);
      attachHeatmapHover(document.getElementById("cDelta"), right.hist2d_x_H, "delta", "deltaHover", right.samples || []);

      const xLabel = (DATA.meta.axes && DATA.meta.axes.x) ? DATA.meta.axes.x : "p_t";
      drawHist(document.getElementById("hLeftX"), left.hist1d_x, {{value: left.x_thr}}, `left ${xLabel}`);
      drawHist(document.getElementById("hLeftH"), left.hist1d_H, {{value: left.h_thr}}, "left H");
      drawHist(document.getElementById("hRightX"), right.hist1d_x, {{value: left.x_thr}}, `right ${xLabel} (left thr)`);
      drawHist(document.getElementById("hRightH"), right.hist1d_H, {{value: left.h_thr}}, "right H (left thr)");
      attachHistHover(document.getElementById("hLeftX"), left.hist1d_x, `left ${xLabel}`, "hLeftXHover");
      attachHistHover(document.getElementById("hLeftH"), left.hist1d_H, "left H", "hLeftHHover");
      attachHistHover(document.getElementById("hRightX"), right.hist1d_x, `right ${xLabel}`, "hRightXHover");
      attachHistHover(document.getElementById("hRightH"), right.hist1d_H, "right H", "hRightHHover");

      const rightCC = ccRateFromHist(right.hist2d_x_H, leftThr);
      const leftCC = ccRateFromHist(left.hist2d_x_H, leftThr);
      const deltaPpl = right.ppl - left.ppl;
      const deltaPplPct = left.ppl > 0 ? (deltaPpl / left.ppl) * 100.0 : 0.0;
      const deltaCC = rightCC - leftCC;
      const deltaMeanP = right.mean_prob - left.mean_prob;
      const js2d = jsDivergence(left.hist2d_x_H.counts, right.hist2d_x_H.counts);

      // Near-lossless gate (strict pass/fail).
      const gates = (DATA.meta && DATA.meta.gates) ? DATA.meta.gates : null;
      const thr = (gates && gates.thresholds) ? gates.thresholds : null;
      let gatePass = null;
      let gateWhy = "";
      if (thr) {{
        const relOk = Math.abs(deltaPpl) <= Math.max(Number(thr.max_abs_delta_ppl || 0), Math.abs(left.ppl) * Number(thr.max_rel_delta_ppl || 0));
        const ccOk = Math.abs(deltaCC) <= Number(thr.max_abs_delta_cc_rate || 0);
        const mpOk = Math.abs(deltaMeanP) <= Number(thr.max_abs_delta_mean_prob || 0);
        const jsOk = js2d <= Number(thr.max_js2d || 0);
        gatePass = relOk && ccOk && mpOk && jsOk;
        gateWhy = `ΔPPL<=max(abs=${thr.max_abs_delta_ppl}, rel=${thr.max_rel_delta_ppl}) & |ΔCC|<=${thr.max_abs_delta_cc_rate} & |Δmean_p|<=${thr.max_abs_delta_mean_prob} & JS2D<=${thr.max_js2d}`;
      }}
      const winner = (deltaPpl < 0 && deltaCC < 0 && js2d < 0.02 && deltaMeanP > 0) ? "RIGHT better" :
                     (deltaPpl > 0 && deltaCC > 0 && js2d > 0.02 && deltaMeanP < 0) ? "LEFT better" : "MIXED";
      const hero = [
        ...(gatePass === null ? [] : [{{ label: "Near‑Lossless Gate", value: gatePass ? "PASS" : "FAIL", sub: gateWhy, cls: gatePass ? "good" : "bad" }}]),
        {{ label: "Winner", value: winner, sub: "based on ΔPPL + ΔCC + JS2D + Δmean_p", cls: winner === "RIGHT better" ? "good" : (winner === "LEFT better" ? "bad" : "") }},
        {{ label: "ΔPPL", value: fmtSigned(deltaPpl,3), sub: `right - left (${fmtSigned(deltaPplPct,1)}%)`, cls: deltaPpl > 0 ? "bad" : "good" }},
        {{ label: "ΔCC (pp)", value: fmtSigned(deltaCC*100,2), sub: "right - left (left thresholds)", cls: deltaCC > 0 ? "bad" : "good" }},
        {{ label: "JS2D", value: fmt(js2d,4), sub: "distribution shift", cls: js2d > 0.02 ? "bad" : "" }},
        {{ label: "Δ mean p", value: fmtSigned(deltaMeanP,4), sub: "right - left", cls: deltaMeanP < 0 ? "bad" : "good" }},
      ];
      let html = "";
      hero.forEach(h => {{
        const extra = h.cls ? (" " + h.cls) : "";
        html += `<div class='stat'><div class='label'>${{h.label}}</div><div class='value${{extra}}'>${{h.value}}</div><div class='sub'>${{h.sub}}</div></div>`;
      }});
      document.getElementById("heroStats").innerHTML = html;
      document.getElementById("heroNote").textContent =
        "Interpretation: lower PPL and CC, lower JS2D, higher mean p is better. Left thresholds define CC region.";

      const rows = [
        ["PPL", fmt(left.ppl,3), fmt(right.ppl,3), fmtSigned(deltaPpl,3)],
        ["mean NLL", fmt(left.mean_nll,6), fmt(right.mean_nll,6), fmtSigned(right.mean_nll - left.mean_nll,6)],
        ["CC_rate (leftThr)", fmt(leftCC,4), fmt(rightCC,4), fmtSigned(deltaCC,4)],
        ["mean_prob", fmt(left.mean_prob,6), fmt(right.mean_prob,6), fmtSigned(deltaMeanP,6)],
        ["mean_entropy", fmt(left.mean_entropy,4), fmt(right.mean_entropy,4), fmtSigned(right.mean_entropy - left.mean_entropy,4)],
        ["JS divergence (2D)", "—", fmt(js2d,4), fmtSigned(js2d,4)],
      ];
      let t = "<table><tr><th class='left'>metric</th><th>left</th><th>right</th><th>Δ (right-left)</th></tr>";
      rows.forEach(r => {{
        const d = (typeof r[3] === "string" && r[3] !== "—") ? Number(r[3]) : null;
        const cls = deltaClass(d);
        t += `<tr><td class='left'>${{r[0]}}</td><td>${{r[1]}}</td><td>${{r[2]}}</td><td class='mono ${{cls}}'>${{r[3]}}</td></tr>`;
      }});
      t += "</table>";
      document.getElementById("compareTable").innerHTML = t;

      document.getElementById("leftStats").textContent =
        `kept_tokens=${left.kept_tokens} | tok/s(pred)=${fmt(left.tok_s_pred,1)} | mean_p=${fmt(left.mean_prob,5)} | mean_H=${fmt(left.mean_entropy,4)}`;
      document.getElementById("rightStats").textContent =
        `kept_tokens=${right.kept_tokens} | tok/s(pred)=${fmt(right.tok_s_pred,1)} | mean_p=${fmt(right.mean_prob,5)} | mean_H=${fmt(right.mean_entropy,4)}`;
      document.getElementById("deltaStats").textContent =
        `ΔPPL=${fmtSigned(deltaPpl,3)} | ΔCC=${fmtSigned(deltaCC*100,2)}pp | JS2D=${fmt(js2d,4)}`;
    }}

    populate();
    renderLeaderboard();
    renderOrdering();
    document.getElementById("leftModel").addEventListener("change", render);
    document.getElementById("rightModel").addEventListener("change", render);
    document.getElementById("packSel").addEventListener("change", render);
    document.getElementById("seqSel").addEventListener("change", render);
    document.getElementById("swapBtn").addEventListener("click", () => {{
      const l = document.getElementById("leftModel");
      const r = document.getElementById("rightModel");
      const tmp = l.value; l.value = r.value; r.value = tmp;
      render();
    }});
    window.addEventListener("resize", render);
    render();
  </script>
</body>
</html>
"""
    return html.replace("{data}", data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsons", required=True, help="Comma-separated list of single-model JSON files.")
    parser.add_argument("--html-out", default="reports/eaft_dynamic_compare.html")
    args = parser.parse_args()

    paths = [Path(p.strip()) for p in args.input_jsons.split(",") if p.strip()]
    if len(paths) < 1:
        raise SystemExit("Need at least one JSON file.")
    for p in paths:
        if not p.exists():
            raise SystemExit(f"Missing {p}")

    payload = _merge_runs(paths)
    html = _render_html_plotly(payload)

    out_path = Path(args.html_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"[+] Wrote {out_path}")


if __name__ == "__main__":
    main()
