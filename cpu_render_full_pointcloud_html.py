#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import html
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def _dict_encode(col: pa.Array, *, unknown: str = "unknown") -> tuple[np.ndarray, list[str]]:
    if col.null_count:
        col = pc.fill_null(col, unknown)
    darr = col.dictionary_encode()
    labels = [str(x) for x in darr.dictionary.to_pylist()]
    codes = darr.indices.to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
    if codes.min(initial=0) < 0:
        raise RuntimeError("unexpected negative dictionary index")
    max_code = int(codes.max(initial=0))
    if max_code <= 255:
        return codes.astype(np.uint8, copy=False), labels
    if max_code <= 65535:
        return codes.astype(np.uint16, copy=False), labels
    return codes.astype(np.uint32, copy=False), labels


def _render_html(*, title: str, meta: dict[str, Any], b64: str) -> str:
    title_html = html.escape(title, quote=True)
    title_json = json.dumps(title, ensure_ascii=False)
    meta_json = json.dumps(meta, ensure_ascii=False, separators=(",", ":"))
    b64_json = json.dumps(b64, ensure_ascii=False)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title_html}</title>
  <style>
    :root {{
      --bg: #ffffff;
      --fg: #0f172a;
      --muted: #334155;
      --border: #e2e8f0;
      --control_bg: #ffffff;
      --chip_bg: rgba(148, 163, 184, 0.12);
      --chip_border: rgba(148, 163, 184, 0.35);
    }}
    :root[data-theme="dark"] {{
      --bg: #0b1020;
      --fg: #e2e8f0;
      --muted: #cbd5e1;
      --border: rgba(148, 163, 184, 0.25);
      --control_bg: rgba(15, 23, 42, 0.85);
      --chip_bg: rgba(148, 163, 184, 0.10);
      --chip_border: rgba(148, 163, 184, 0.22);
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--fg);
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
    }}
    #topbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      background: var(--bg);
      z-index: 10;
    }}
    #topbar .left {{ display: flex; flex-direction: column; gap: 2px; }}
    #topbar .muted {{ color: var(--muted); font-size: 12px; }}
    #controls {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; justify-content: flex-end; }}
    select, input[type="range"] {{
      background: var(--control_bg);
      color: var(--fg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 4px 8px;
    }}
    label {{ user-select: none; }}
    input[type="checkbox"] {{ accent-color: #38bdf8; }}
    #legend {{ max-height: 120px; overflow: auto; font-size: 12px; color: var(--muted); }}
    .chip {{
      display: inline-block;
      padding: 2px 6px;
      margin: 2px 6px 2px 0;
      border-radius: 10px;
      border: 1px solid var(--chip_border);
      background: var(--chip_bg);
    }}
    .swatch {{ display: inline-block; width: 10px; height: 10px; margin-right: 6px; border-radius: 2px; vertical-align: middle; }}
    #canvasWrap {{ width: 100vw; height: calc(100vh - 68px); background: var(--bg); }}
    canvas {{ display: block; width: 100%; height: 100%; }}
  </style>
</head>
<body>
  <div id="topbar">
    <div class="left">
      <div><strong>{title_html}</strong></div>
      <div class="muted">Full 2M pointcloud (PCA-3). Use mouse drag to orbit, right-drag to pan, wheel to zoom.</div>
    </div>
    <div id="controls">
      <div>
        Color by:
        <select id="colorBy">
          <option value="dataset">dataset</option>
          <option value="meta_domain">domain</option>
          <option value="mix_group">mix_group</option>
          <option value="difficulty_bin">difficulty</option>
          <option value="len_bucket">len_bucket</option>
        </select>
      </div>
      <div>
        <label>size <input id="ptSize" type="range" min="1" max="6" step="1" value="2"/></label>
      </div>
      <div>
        <label>opacity <input id="ptOpacity" type="range" min="0.05" max="1.0" step="0.05" value="0.45"/></label>
      </div>
      <div>
        <label><input id="darkMode" type="checkbox" checked /> dark</label>
      </div>
    </div>
  </div>
  <div id="legend" style="padding: 6px 12px;"></div>
  <div id="canvasWrap"></div>

  <script type="module">
    import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";
    import {{ OrbitControls }} from "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js";

    const meta = {meta_json};
    const b64 = {b64_json};

    function setTheme(dark) {{
      document.documentElement.dataset.theme = dark ? "dark" : "light";
      const bg = dark ? 0x0b1020 : 0xffffff;
      renderer.setClearColor(bg, 1.0);
    }}

    function b64ToArrayBufferChunked(base64, chunkChars = 8 * 1024 * 1024) {{
      // chunkChars should be a multiple of 4.
      const chunks = [];
      let totalLen = 0;
      for (let i = 0; i < base64.length; i += chunkChars) {{
        const slice = base64.slice(i, i + chunkChars);
        const bin = atob(slice);
        const bytes = new Uint8Array(bin.length);
        for (let j = 0; j < bin.length; j++) bytes[j] = bin.charCodeAt(j);
        chunks.push(bytes);
        totalLen += bytes.length;
      }}
      const out = new Uint8Array(totalLen);
      let off = 0;
      for (const c of chunks) {{
        out.set(c, off);
        off += c.length;
      }}
      return out.buffer;
    }}

    const buf = b64ToArrayBufferChunked(b64);
    let off = 0;

    const N = meta.n;
    const pos = new Float32Array(buf, off, N * 3);
    off += N * 3 * 4;

    function readAttr(name) {{
      const a = meta.attrs[name];
      const dtype = a.dtype;
      const len = N;
      let out;
      if (dtype === "u8") {{
        out = new Uint8Array(buf, off, len);
        off += len;
      }} else if (dtype === "u16") {{
        out = new Uint16Array(buf, off, len);
        off += len * 2;
      }} else if (dtype === "u32") {{
        out = new Uint32Array(buf, off, len);
        off += len * 4;
      }} else {{
        throw new Error("unknown dtype " + dtype);
      }}
      return out;
    }}

    const codes = {{
      dataset: readAttr("dataset"),
      meta_domain: readAttr("meta_domain"),
      mix_group: readAttr("mix_group"),
      difficulty_bin: readAttr("difficulty_bin"),
      len_bucket: readAttr("len_bucket"),
    }};

    const canvasWrap = document.getElementById("canvasWrap");
    const renderer = new THREE.WebGLRenderer({{ antialias: false, powerPreference: "high-performance" }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    canvasWrap.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 2000);
    camera.position.set(0, 0, 2.5);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    // Center + scale to a sane cube using metadata ranges.
    const cx = (meta.coord_range.x[0] + meta.coord_range.x[1]) / 2.0;
    const cy = (meta.coord_range.y[0] + meta.coord_range.y[1]) / 2.0;
    const cz = (meta.coord_range.z[0] + meta.coord_range.z[1]) / 2.0;
    const sx = (meta.coord_range.x[1] - meta.coord_range.x[0]);
    const sy = (meta.coord_range.y[1] - meta.coord_range.y[0]);
    const sz = (meta.coord_range.z[1] - meta.coord_range.z[0]);
    const s = 1.8 / Math.max(sx, sy, sz);

    for (let i = 0; i < N; i++) {{
      const j = i * 3;
      pos[j + 0] = (pos[j + 0] - cx) * s;
      pos[j + 1] = (pos[j + 1] - cy) * s;
      pos[j + 2] = (pos[j + 2] - cz) * s;
    }}

    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));

    const colors = new Float32Array(N * 3);
    geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geom.computeBoundingSphere();

    const material = new THREE.PointsMaterial({{
      size: 0.006,
      vertexColors: true,
      transparent: true,
      opacity: 0.45,
      depthWrite: false,
      sizeAttenuation: true,
    }});

    const points = new THREE.Points(geom, material);
    points.frustumCulled = false;
    scene.add(points);

    const axes = new THREE.AxesHelper(1.0);
    axes.material.depthTest = false;
    axes.renderOrder = 1;
    scene.add(axes);

    const palette = [
      "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f",
      "#bcbd22","#17becf","#393b79","#637939","#8c6d31","#843c39","#7b4173","#3182bd",
      "#31a354","#756bb1","#636363","#e6550d"
    ];

    function hexToRgb01(hex) {{
      const h = hex.replace("#", "");
      const r = parseInt(h.substring(0, 2), 16) / 255;
      const g = parseInt(h.substring(2, 4), 16) / 255;
      const b = parseInt(h.substring(4, 6), 16) / 255;
      return [r, g, b];
    }}

    function setLegend(attrName) {{
      const el = document.getElementById("legend");
      const labels = meta.attrs[attrName].labels;
      const parts = [];
      for (let i = 0; i < labels.length; i++) {{
        const c = palette[i % palette.length];
        const label = labels[i];
        parts.push(`<span class="chip"><span class="swatch" style="background:${{c}}"></span>${{label}}</span>`);
      }}
      el.innerHTML = parts.join("");
    }}

    function applyColors(attrName) {{
      const labels = meta.attrs[attrName].labels;
      const lut = new Array(labels.length);
      for (let i = 0; i < labels.length; i++) {{
        lut[i] = hexToRgb01(palette[i % palette.length]);
      }}
      const codeArr = codes[attrName];
      for (let i = 0; i < N; i++) {{
        const rgb = lut[codeArr[i] % lut.length];
        const j = i * 3;
        colors[j + 0] = rgb[0];
        colors[j + 1] = rgb[1];
        colors[j + 2] = rgb[2];
      }}
      geom.attributes.color.needsUpdate = true;
      setLegend(attrName);
    }}

    function resize() {{
      const w = canvasWrap.clientWidth;
      const h = canvasWrap.clientHeight;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }}
    window.addEventListener("resize", resize);
    resize();

    // UI hooks
    const colorBy = document.getElementById("colorBy");
    const darkMode = document.getElementById("darkMode");
    const ptSize = document.getElementById("ptSize");
    const ptOpacity = document.getElementById("ptOpacity");

    function syncMaterial() {{
      material.size = 0.0025 * Number(ptSize.value);
      material.opacity = Number(ptOpacity.value);
      material.needsUpdate = true;
    }}
    ptSize.addEventListener("input", syncMaterial);
    ptOpacity.addEventListener("input", syncMaterial);
    colorBy.addEventListener("change", () => applyColors(colorBy.value));
    darkMode.addEventListener("change", () => setTheme(darkMode.checked));

    setTheme(true);
    syncMaterial();
    applyColors("dataset");

    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }}
    animate();
  </script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a full pointcloud HTML from a PCA-3 parquet (2M points).")
    ap.add_argument("--in_pca_parquet", type=str, required=True)
    ap.add_argument("--in_density_manifest", type=str, required=True, help="density_manifest.json for coord_range")
    ap.add_argument("--out_html", type=str, required=True)
    ap.add_argument("--title", type=str, default="Full Pointcloud (2M)")
    ap.add_argument("--x_col", type=str, default="pca_x")
    ap.add_argument("--y_col", type=str, default="pca_y")
    ap.add_argument("--z_col", type=str, default="pca_z")
    args = ap.parse_args()

    pca_path = Path(args.in_pca_parquet)
    dens_manifest = json.loads(Path(args.in_density_manifest).read_text(encoding="utf-8"))
    coord_range = dens_manifest["coord_range"]

    x_col = str(args.x_col)
    y_col = str(args.y_col)
    z_col = str(args.z_col)
    cols = [x_col, y_col, z_col, "dataset", "meta_domain", "mix_group", "difficulty_bin", "len_bucket"]
    tbl = pq.read_table(pca_path, columns=cols)
    n = int(tbl.num_rows)
    if n <= 0:
        raise SystemExit("no rows")

    x = tbl[x_col].to_numpy(zero_copy_only=False).astype(np.float32)
    y = tbl[y_col].to_numpy(zero_copy_only=False).astype(np.float32)
    z = tbl[z_col].to_numpy(zero_copy_only=False).astype(np.float32)
    pos = np.empty((n, 3), dtype=np.float32)
    pos[:, 0] = x
    pos[:, 1] = y
    pos[:, 2] = z

    codes: dict[str, np.ndarray] = {}
    labels: dict[str, list[str]] = {}
    for name in ["dataset", "meta_domain", "mix_group", "difficulty_bin", "len_bucket"]:
        c, lab = _dict_encode(tbl[name].combine_chunks())
        codes[name] = c
        labels[name] = lab

    # Pack binary: positions (f32) + 5 code arrays (u8/u16/u32).
    # Keep everything little-endian.
    chunks: list[bytes] = [pos.reshape(-1).tobytes(order="C")]
    attrs_meta: dict[str, dict[str, Any]] = {}
    for name in ["dataset", "meta_domain", "mix_group", "difficulty_bin", "len_bucket"]:
        arr = codes[name]
        if arr.dtype == np.uint8:
            dtype = "u8"
        elif arr.dtype == np.uint16:
            dtype = "u16"
        elif arr.dtype == np.uint32:
            dtype = "u32"
        else:
            raise RuntimeError(f"unsupported dtype {arr.dtype} for {name}")
        chunks.append(arr.tobytes(order="C"))
        attrs_meta[name] = {"dtype": dtype, "labels": labels[name]}

    blob = b"".join(chunks)
    b64 = base64.b64encode(blob).decode("ascii")

    meta = {
        "generated_at": _now(),
        "source": str(pca_path),
        "n": n,
        "attrs": attrs_meta,
        "coord_range": coord_range,
    }

    out_path = Path(args.out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render_html(title=args.title, meta=meta, b64=b64), encoding="utf-8")
    print(f"[ok] wrote {out_path} ({len(b64) / (1024 * 1024):.1f} MiB base64)")


if __name__ == "__main__":
    main()
