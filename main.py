#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers",
#   "numpy",
#   "scikit-learn",
#   "pot",
#   "umap-learn",
#   "accelerate",
# ]
# ///
"""
Hidden-State Manifold Explorer  ·  v2
=====================================
Kein Plotly, kein Dash. Python rechnet, Canvas rendert.

 Python  ->  Hidden States, PCA/t-SNE/UMAP, Wasserstein, Shape-Metriken
 Browser ->  handgezeichnetes WebGL-freies Canvas-UI (Stil: ascending manifold)

Start:  python3 main.py   ->  http://127.0.0.1:8051
"""

import os
import subprocess
import sys
from pathlib import Path

# ============================================================================
# 0) BOOTSTRAP
# ============================================================================

REQUIRED = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("ot", "pot"),
    ("umap", "umap-learn"),
    ("accelerate", "accelerate"),
]

VENV_DIR = Path(__file__).resolve().parent / ".venv-manifold"
BOOTSTRAP_FLAG = "MANIFOLD_BOOTSTRAPPED"

def _venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

def _have_uv() -> bool:
    from shutil import which
    return which("uv") is not None

def _install_uv():
    print("[bootstrap] 'uv' nicht gefunden -> installiere via pip ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "-q", "uv"])
    import site
    user_bin = Path(site.getuserbase()) / ("Scripts" if os.name == "nt" else "bin")
    os.environ["PATH"] = f"{user_bin}{os.pathsep}{os.environ.get('PATH','')}"

def _missing_modules():
    missing = []
    for mod, pkg in REQUIRED:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    return missing

def bootstrap():
    if os.environ.get(BOOTSTRAP_FLAG) == "1":
        missing = _missing_modules()
        if missing:
            print(f"[bootstrap] FEHLER: nach Neustart fehlen immer noch: {missing}")
            sys.exit(1)
        return

    missing = _missing_modules()
    if not missing:
        os.environ[BOOTSTRAP_FLAG] = "1"
        return

    print(f"[bootstrap] Fehlende Pakete: {missing}")
    if not _have_uv():
        _install_uv()
    if not VENV_DIR.exists():
        print(f"[bootstrap] Erzeuge venv unter {VENV_DIR} ...")
        subprocess.check_call(["uv", "venv", str(VENV_DIR)])

    pkgs = [pkg for _, pkg in REQUIRED]
    print("[bootstrap] Installiere Abhaengigkeiten via uv ...")
    env = dict(os.environ, VIRTUAL_ENV=str(VENV_DIR))
    subprocess.check_call(
        ["uv", "pip", "install", "--python", str(_venv_python()), *pkgs], env=env)

    print("[bootstrap] Neustart im venv ...")
    env[BOOTSTRAP_FLAG] = "1"
    os.execve(str(_venv_python()),
              [str(_venv_python()), os.path.abspath(__file__), *sys.argv[1:]], env)

bootstrap()

# ============================================================================
# 1) Imports
# ============================================================================

import warnings
warnings.filterwarnings("ignore")

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
import ot
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

# ============================================================================
# 2) Modell-Katalog
# ============================================================================

MODELS = {
    "distilbert-base-uncased":                {"size": "66M",  "tier": "CPU",  "note": "6 Layer, schnell"},
    "sentence-transformers/all-MiniLM-L6-v2": {"size": "22M",  "tier": "CPU",  "note": "winzig, 6 Layer"},
    "bert-base-uncased":                      {"size": "110M", "tier": "CPU",  "note": "12 Layer, Klassiker"},
    "roberta-base":                           {"size": "125M", "tier": "CPU",  "note": "12 Layer"},
    "gpt2":                                   {"size": "124M", "tier": "CPU",  "note": "decoder-only"},
    "EleutherAI/pythia-160m":                 {"size": "160M", "tier": "CPU",  "note": "decoder, RoPE"},
    "gpt2-medium":                            {"size": "355M", "tier": "CPU+", "note": "24 Layer"},
    "EleutherAI/pythia-410m":                 {"size": "410M", "tier": "CPU+", "note": "24 Layer"},
    "microsoft/deberta-v3-base":              {"size": "184M", "tier": "CPU+", "note": "disentangled attn"},
    "EleutherAI/pythia-1.4b":                 {"size": "1.4B", "tier": "GPU",  "note": "24 Layer"},
    "Qwen/Qwen2.5-0.5B":                      {"size": "0.5B", "tier": "GPU",  "note": "modern, RoPE"},
    "Qwen/Qwen2.5-1.5B":                      {"size": "1.5B", "tier": "GPU",  "note": "28 Layer"},
    "meta-llama/Llama-3.2-1B":                {"size": "1.2B", "tier": "GPU",  "note": "gated, HF-Login"},
}

_CACHE = {}
_LOCK = threading.Lock()

def load_model(name: str):
    with _LOCK:
        if name in _CACHE:
            return _CACHE[name]
        print(f"[model] Lade {name} auf {DEVICE} ...")
        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        dtype = torch.float32 if DEVICE == "cpu" else torch.float16
        model = AutoModel.from_pretrained(name, torch_dtype=dtype,
                                          output_hidden_states=True)
        model.to(DEVICE).eval()
        _CACHE[name] = (tok, model)
        return tok, model

# ============================================================================
# 3) Hidden States
# ============================================================================

@torch.no_grad()
def hidden_states(text: str, model_name: str):
    tok, model = load_model(model_name)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=192)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    out = model(**enc)
    hs = torch.stack(out.hidden_states)[:, 0].float().cpu().numpy()  # [L+1,T,D]
    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
    clean = []
    for t in tokens:
        t = t.replace("Ġ", "·").replace("▁", "·").replace("##", "")
        clean.append(t)
    return clean, hs

# ============================================================================
# 4) Shape-Metriken
# ============================================================================

def shape_metrics(X: np.ndarray) -> dict:
    Xc = X - X.mean(0, keepdims=True)
    T = Xc.shape[0]
    if T < 2:
        return dict(eff_dim=1.0, anisotropy=0.0, radius=0.0,
                    spectrum=[1.0], var_top2=1.0, curvature=0.0)

    s = np.linalg.svd(Xc, compute_uv=False)
    ev = (s ** 2) / max(T - 1, 1)
    p = ev / (ev.sum() + 1e-12)
    eff_dim = float((ev.sum() ** 2) / ((ev ** 2).sum() + 1e-12))

    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    G = Xn @ Xn.T
    iu = np.triu_indices(T, k=1)
    anisotropy = float(G[iu].mean()) if len(iu[0]) else 0.0
    radius = float(np.sqrt((Xc ** 2).sum(1).mean()))

    # Sequenz-Kruemmung: mittlerer Winkel aufeinanderfolgender Differenzvektoren
    curv = 0.0
    if T >= 3:
        d = np.diff(X, axis=0)
        dn = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
        cos = np.clip((dn[:-1] * dn[1:]).sum(1), -1, 1)
        curv = float(np.arccos(cos).mean())

    return dict(eff_dim=eff_dim, anisotropy=anisotropy, radius=radius,
                spectrum=[float(v) for v in p[:12]],
                var_top2=float(p[:2].sum()), curvature=curv)

# ============================================================================
# 5) Wasserstein
# ============================================================================

def wasserstein(A, B, mode="exact", normalize=True, n_proj=256):
    A = np.ascontiguousarray(A, dtype=np.float64)
    B = np.ascontiguousarray(B, dtype=np.float64)
    if normalize:
        scale = (np.linalg.norm(A - A.mean(0), axis=1).mean() +
                 np.linalg.norm(B - B.mean(0), axis=1).mean()) / 2 + 1e-12
        A, B = A / scale, B / scale
    a = np.ones(len(A)) / len(A)
    b = np.ones(len(B)) / len(B)
    if mode == "sliced":
        return float(ot.sliced_wasserstein_distance(A, B, a, b,
                                                    n_projections=n_proj, seed=0))
    M = ot.dist(A, B, metric="euclidean")
    if mode == "sinkhorn":
        G = ot.sinkhorn(a, b, M, reg=0.05, numItermax=500)
        return float((G * M).sum())
    return float(ot.emd2(a, b, M))

# ============================================================================
# 6) Dimensionsreduktion  (+ Layer-Alignment via Procrustes)
# ============================================================================

def reduce(X, method="pca", dims=3, seed=0):
    n = X.shape[0]
    dims = min(dims, max(1, n - 1)) if n > 1 else 1
    if method == "tsne":
        perp = max(2.0, min(30.0, (n - 1) / 3.0))
        return TSNE(n_components=min(dims, 3), perplexity=perp, init="pca",
                    random_state=seed, max_iter=500).fit_transform(X)
    if method == "umap":
        import umap
        nn = max(2, min(15, n - 1))
        return umap.UMAP(n_components=dims, n_neighbors=nn, min_dist=0.1,
                         random_state=seed).fit_transform(X)
    return PCA(n_components=dims, random_state=seed).fit_transform(X)

def joint_reduce(blocks, method="pca", dims=3):
    sizes = [b.shape[0] for b in blocks]
    Z = reduce(np.vstack(blocks), method=method, dims=dims)
    if Z.shape[1] < dims:                       # auf dims auffuellen
        Z = np.hstack([Z, np.zeros((Z.shape[0], dims - Z.shape[1]))])
    out, i = [], 0
    for s in sizes:
        out.append(Z[i:i + s])
        i += s
    return out

def procrustes_align(P, ref):
    """Rotiert/spiegelt P so, dass es moeglichst auf ref liegt.
    Verhindert das wilde Umklappen der PCA-Achsen zwischen Layern."""
    if ref is None or P.shape != ref.shape:
        return P
    A = P - P.mean(0)
    B = ref - ref.mean(0)
    U, _, Vt = np.linalg.svd(A.T @ B)
    R = U @ Vt
    return A @ R + ref.mean(0)

# ============================================================================
# 7) Analyse-Pipeline
# ============================================================================

def analyse(ta, tb, model_name, method, emd_mode, dims=3):
    tok_a, HA = hidden_states(ta, model_name)
    tok_b, HB = hidden_states(tb, model_name)
    n_layers = HA.shape[0]

    proj_a, proj_b, emds, sh_a, sh_b = [], [], [], [], []
    prev_a = prev_b = None

    for L in range(n_layers):
        A, B = HA[L], HB[L]
        pa, pb = joint_reduce([A, B], method=method, dims=dims)
        pa = procrustes_align(pa, prev_a)
        pb = procrustes_align(pb, prev_b)
        prev_a, prev_b = pa, pb
        proj_a.append(pa)
        proj_b.append(pb)
        emds.append(wasserstein(A, B, mode=emd_mode))
        sh_a.append(shape_metrics(A))
        sh_b.append(shape_metrics(B))

    # Gemeinsame Normierung ueber ALLE Layer -> stabile Kamera
    allp = np.vstack(proj_a + proj_b)
    centre = allp.mean(0)
    scale = float(np.abs(allp - centre).max()) + 1e-9
    proj_a = [((p - centre) / scale).round(4).tolist() for p in proj_a]
    proj_b = [((p - centre) / scale).round(4).tolist() for p in proj_b]

    drift = [0.0] + [wasserstein(HA[L - 1], HA[L], mode=emd_mode)
                     for L in range(1, n_layers)]
    norms_a = [float(np.linalg.norm(HA[L], axis=1).mean()) for L in range(n_layers)]
    norms_b = [float(np.linalg.norm(HB[L], axis=1).mean()) for L in range(n_layers)]

    return dict(tok_a=tok_a, tok_b=tok_b, proj_a=proj_a, proj_b=proj_b,
                emds=emds, drift=drift, shapes_a=sh_a, shapes_b=sh_b,
                norms_a=norms_a, norms_b=norms_b,
                n_layers=n_layers, model=model_name, method=method,
                text_a=ta, text_b=tb, device=DEVICE)

# ============================================================================
# 8) FRONTEND
# ============================================================================

def build_html() -> str:
    opts = "".join(
        f'<option value="{k}">[{v["tier"]}] {k} · {v["size"]} · {v["note"]}</option>'
        for k, v in MODELS.items())

    return r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>hidden-state manifold</title>
<style>
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:#070910;overflow:hidden;
    font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:#79829f}
  canvas{display:block;position:fixed;inset:0}
  ::-webkit-scrollbar{width:6px}
  ::-webkit-scrollbar-thumb{background:rgba(120,140,200,.18);border-radius:3px}

  /* ── linke Konsole ───────────────────── */
  #ui{position:fixed;left:24px;top:24px;z-index:14;width:302px;
    max-height:calc(100vh - 48px);overflow-y:auto;padding-right:4px}
  .card{background:rgba(14,18,30,.52);border:1px solid rgba(120,140,200,.1);
    border-radius:10px;padding:13px 14px;margin-bottom:9px;
    backdrop-filter:blur(16px);transition:.35s cubic-bezier(.22,1,.36,1)}
  .card:hover{border-color:rgba(150,175,235,.2)}
  .lb{display:block;font-size:8.5px;letter-spacing:.3em;opacity:.55;
    margin-bottom:6px;color:#8c97bb}
  textarea,select{width:100%;background:rgba(7,9,16,.7);color:#c3cbe4;
    border:1px solid rgba(120,140,200,.14);border-radius:6px;padding:8px 9px;
    font:11px/1.45 ui-monospace,monospace;outline:none;resize:none;
    transition:.25s}
  textarea:focus,select:focus{border-color:var(--a);color:#e8edfa}
  textarea{height:52px}
  select{height:30px;cursor:pointer;appearance:none;
    background-image:linear-gradient(45deg,transparent 50%,#6b7593 50%),
      linear-gradient(135deg,#6b7593 50%,transparent 50%);
    background-position:calc(100% - 15px) 13px,calc(100% - 10px) 13px;
    background-size:5px 5px;background-repeat:no-repeat}
  select option{background:#0c1018;color:#c3cbe4}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:8px}
  #go{width:100%;margin-top:2px;padding:11px;border-radius:8px;cursor:pointer;
    background:rgba(22,27,42,.8);border:1px solid var(--a);color:#e8edfa;
    font:10px/1 ui-monospace,monospace;letter-spacing:.28em;
    box-shadow:0 0 22px -8px var(--a);transition:.3s}
  #go:hover{background:var(--a);color:#070910}
  #go.busy{opacity:.55;pointer-events:none;letter-spacing:.2em}
  #msg{font-size:9px;letter-spacing:.14em;color:#5a6casd;margin-top:8px;
    min-height:12px;color:#6d7795}
  #msg.err{color:#e0657f}

  /* ── Kennzahlen ──────────────────────── */
  .stat{display:flex;justify-content:space-between;align-items:baseline;
    padding:4px 0;border-bottom:1px solid rgba(120,140,200,.07)}
  .stat:last-child{border:0}
  .stat span{font-size:8.5px;letter-spacing:.2em;color:#5c6party}
  .stat span{color:#5c6688}
  .stat b{font-weight:400;font-size:12px;color:#d2daf0;
    font-variant-numeric:tabular-nums}
  .stat b.a{color:#6fd3f7}
  .stat b.b{color:#f77fb0}

  /* ── Header ──────────────────────────── */
  #h{position:fixed;right:116px;top:26px;text-align:right;z-index:10;
    font-size:9.5px;letter-spacing:.26em;color:#434a65;pointer-events:none}
  #h b{display:block;font-size:40px;letter-spacing:.03em;color:#cfd8ef;
    font-weight:200;margin-top:-2px;font-variant-numeric:tabular-nums}
  #h i{font-style:normal;font-size:8.5px;letter-spacing:.2em;color:#39405c}

  /* ── Slider ──────────────────────────── */
  #rail{position:fixed;right:46px;top:50%;transform:translateY(-50%);
    width:46px;height:58vh;z-index:12;cursor:pointer}
  #line{position:absolute;left:23px;top:0;bottom:0;width:1px;
    background:linear-gradient(180deg,rgba(255,255,255,.03),
      rgba(255,255,255,.12),rgba(255,255,255,.03))}
  #prog{position:absolute;left:22.5px;bottom:0;width:2px;border-radius:1px;
    background:linear-gradient(180deg,var(--a),transparent);opacity:.6}
  .dot{position:absolute;left:19px;width:9px;height:9px;margin-top:-4.5px;
    border-radius:50%;background:#10131f;border:1px solid rgba(150,170,220,.26);
    transition:.3s cubic-bezier(.22,1,.36,1)}
  .dot.pass{border-color:rgba(160,190,245,.5);background:#1b2234}
  .dot.act{background:var(--a);border-color:var(--a);
    box-shadow:0 0 14px var(--a);transform:scale(1.5)}
  .dot label{position:absolute;right:17px;top:-4px;font-size:8px;
    letter-spacing:.18em;color:#39405c;white-space:nowrap;pointer-events:none}
  .dot.act label{color:#cbd5ef}
  #knob{position:absolute;left:14.5px;width:18px;height:18px;margin-top:-9px;
    border-radius:50%;border:1px solid var(--a);background:rgba(10,14,24,.85);
    box-shadow:0 0 20px -2px var(--a);pointer-events:none}
  #knob:after{content:'';position:absolute;inset:5px;border-radius:50%;
    background:var(--a)}

  /* ── Transport ───────────────────────── */
  #tp{position:fixed;right:30px;bottom:26px;display:flex;gap:7px;z-index:12}
  .btn{width:30px;height:30px;border-radius:7px;cursor:pointer;
    background:rgba(16,20,32,.6);border:1px solid rgba(120,140,200,.14);
    color:#8e98ba;font-size:11px;display:flex;align-items:center;
    justify-content:center;backdrop-filter:blur(10px);transition:.25s}
  .btn:hover{border-color:rgba(160,185,240,.34);color:#dfe6fa}
  .btn.on{border-color:var(--a);color:#eaf0ff;box-shadow:0 0 16px -4px var(--a)}

  /* ── Token-Leiste ────────────────────── */
  #toks{position:fixed;left:50%;bottom:22px;transform:translateX(-50%);
    z-index:11;display:flex;flex-direction:column;gap:5px;align-items:center;
    pointer-events:none;max-width:60vw}
  .trow{display:flex;gap:4px;flex-wrap:wrap;justify-content:center}
  .tk{padding:2px 6px;border-radius:4px;font-size:9px;letter-spacing:.08em;
    background:rgba(14,18,30,.55);border:1px solid rgba(120,140,200,.12);
    backdrop-filter:blur(8px);transition:.25s}
  .tk.a{color:#6fd3f7;border-color:rgba(111,211,247,.2)}
  .tk.b{color:#f77fb0;border-color:rgba(247,127,176,.2)}

  /* ── Footer ──────────────────────────── */
  #ft{position:fixed;left:24px;bottom:22px;font-size:9px;letter-spacing:.24em;
    color:#2b3045;z-index:10;pointer-events:none}

  /* ── Sparkline-Panels ────────────────── */
  .spark{width:100%;height:46px;display:block;margin-top:4px}
  .legend{display:flex;gap:10px;margin-top:5px;font-size:8px;letter-spacing:.18em}
  .legend i{font-style:normal;display:flex;align-items:center;gap:4px;color:#5c6688}
  .legend u{text-decoration:none;width:7px;height:2px;border-radius:1px;display:inline-block}

  /* ── Boot-Overlay ────────────────────── */
  #boot{position:fixed;inset:0;z-index:40;background:#070910;
    display:flex;align-items:center;justify-content:center;flex-direction:column;
    gap:14px;transition:opacity .7s;font-size:9px;letter-spacing:.34em;color:#4a5272}
  #boot.gone{opacity:0;pointer-events:none}
  #bootbar{width:180px;height:1px;background:rgba(120,140,200,.14);overflow:hidden}
  #bootbar i{display:block;height:100%;width:34%;background:#6fd3f7;
    animation:sweep 1.25s cubic-bezier(.65,0,.35,1) infinite}
  @keyframes sweep{0%{transform:translateX(-110%)}100%{transform:translateX(330%)}}
</style>
</head>
<body>
<canvas id="c"></canvas>

<div id="ui">
  <div class="card">
    <span class="lb">SATZ A</span>
    <textarea id="ta">The cat sat quietly on the warm windowsill.</textarea>
  </div>
  <div class="card">
    <span class="lb">SATZ B</span>
    <textarea id="tb">Non-abelian cohomology classifies torsors over a site.</textarea>
  </div>
  <div class="card">
    <span class="lb">MODELL</span>
    <select id="model">__OPTS__</select>
    <div class="row" style="margin-top:8px">
      <div><span class="lb">PROJEKTION</span>
        <select id="method">
          <option value="pca">PCA</option>
          <option value="umap">UMAP</option>
          <option value="tsne">T-SNE</option>
        </select></div>
      <div><span class="lb">TRANSPORT</span>
        <select id="emd">
          <option value="exact">EXAKT · EMD</option>
          <option value="sliced">SLICED</option>
          <option value="sinkhorn">SINKHORN</option>
        </select></div>
    </div>
    <div style="height:9px"></div>
    <div id="go">ANALYSIEREN</div>
    <div id="msg"></div>
  </div>

  <div class="card" id="panelStats">
    <span class="lb">LAYER-KENNZAHLEN</span>
    <div class="stat"><span>W₁ (A,B)</span><b id="s-emd">—</b></div>
    <div class="stat"><span>DRIFT L-1→L</span><b id="s-drift">—</b></div>
    <div class="stat"><span>EFF. DIM</span><b><em class="a" style="font-style:normal;color:#6fd3f7" id="s-ea">—</em> <span style="opacity:.3">/</span> <em class="b" style="font-style:normal;color:#f77fb0" id="s-eb">—</em></b></div>
    <div class="stat"><span>ANISOTROPIE</span><b><em style="font-style:normal;color:#6fd3f7" id="s-aa">—</em> <span style="opacity:.3">/</span> <em style="font-style:normal;color:#f77fb0" id="s-ab">—</em></b></div>
    <div class="stat"><span>KRÜMMUNG</span><b><em style="font-style:normal;color:#6fd3f7" id="s-ca">—</em> <span style="opacity:.3">/</span> <em style="font-style:normal;color:#f77fb0" id="s-cb">—</em></b></div>
    <div class="stat"><span>‖h‖ MITTEL</span><b><em style="font-style:normal;color:#6fd3f7" id="s-na">—</em> <span style="opacity:.3">/</span> <em style="font-style:normal;color:#f77fb0" id="s-nb">—</em></b></div>
  </div>

  <div class="card">
    <span class="lb">VERLAUF ÜBER LAYER</span>
    <canvas id="sp1" class="spark"></canvas>
    <div class="legend">
      <i><u style="background:#f77fb0"></u>W₁(A,B)</i>
      <i><u style="background:#7ee7a8"></u>DRIFT</i>
    </div>
    <canvas id="sp2" class="spark" style="margin-top:10px"></canvas>
    <div class="legend">
      <i><u style="background:#6fd3f7"></u>EFF.DIM A</i>
      <i><u style="background:#f77fb0"></u>EFF.DIM B</i>
    </div>
  </div>

  <div class="card">
    <span class="lb">SPEKTRUM @ LAYER</span>
    <canvas id="sp3" class="spark" style="height:60px"></canvas>
  </div>
</div>

<div id="h"><span>LAYER</span><b id="hn">00</b><i id="hs">—</i></div>

<div id="rail"><div id="line"></div><div id="prog"></div><div id="knob"></div></div>

<div id="tp">
  <div class="btn on" id="bPlay">❚❚</div>
  <div class="btn" id="bPrev">◂</div>
  <div class="btn" id="bNext">▸</div>
  <div class="btn" id="bView">◐</div>
</div>

<div id="toks">
  <div class="trow" id="trowA"></div>
  <div class="trow" id="trowB"></div>
</div>

<div id="ft">DRAG ROTATE · SCROLL ZOOM · SLIDER SCRUB · SPACE PLAY</div>

<div id="boot"><div>HIDDEN-STATE MANIFOLD</div><div id="bootbar"><i></i></div>
  <div id="bootmsg">BEREIT · ANALYSE STARTEN</div></div>

<script>
/* ════════════════════════════════════════════════════════════════════
   0 · CANVAS
   ════════════════════════════════════════════════════════════════════ */
const c=document.getElementById('c'),x=c.getContext('2d',{alpha:false});
let W,H,DPR;
function rs(){DPR=Math.min(devicePixelRatio||1,2);W=innerWidth;H=innerHeight;
  c.width=W*DPR;c.height=H*DPR;x.setTransform(DPR,0,0,DPR,0,0);
  x.lineCap='round';x.lineJoin='round'}
addEventListener('resize',rs);rs();

const HUE_A=196, HUE_B=330;
document.documentElement.style.setProperty('--a',`hsl(${HUE_A},72%,60%)`);

/* ════════════════════════════════════════════════════════════════════
   1 · STATE
   ════════════════════════════════════════════════════════════════════ */
let D=null;                 // Analyse-Daten vom Server
let LAYERS=1, RISE=168, GH=3, GAP=0.55;
let pos=0, playing=false, scrub=null, targetPos=null, view=2;
const SPD=0.30;
let yaw=-.62,pit=.24,dist=1180,camY=0,tg={y:-.62,p:.24},dg=null;

/* ════════════════════════════════════════════════════════════════════
   2 · INTERAKTION
   ════════════════════════════════════════════════════════════════════ */
c.onpointerdown=e=>dg={x:e.clientX,y:e.clientY,a:tg.y,b:tg.p};
addEventListener('pointerup',()=>{dg=null;scrub=null});
addEventListener('pointermove',e=>{
  if(scrub!==null){const r=rail.getBoundingClientRect();
    const u=1-Math.max(0,Math.min(1,(e.clientY-r.top)/r.height));
    pos=u*(LAYERS-1); targetPos=null; return}
  if(!dg)return;
  tg.y=dg.a+(e.clientX-dg.x)*.0055;
  tg.p=Math.max(-.1,Math.min(.95,dg.b+(e.clientY-dg.y)*.0042))});
addEventListener('wheel',e=>dist=Math.max(560,Math.min(2400,dist+e.deltaY*.5)),{passive:1});

/* ════════════════════════════════════════════════════════════════════
   3 · PROJEKTION  (3D -> Bildschirm, identische Kamera wie Referenz)
   ════════════════════════════════════════════════════════════════════ */
const SCALE=300;
function project(p,lf,out){
  // p = [x,y,z] in [-1,1], lf = fraktionaler Layer
  const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pit),sp=Math.sin(pit);
  const ax=p[0]*SCALE, az=(p[2]!==undefined?p[2]:0)*SCALE*.72;
  const ay=-(p[1]*SCALE*.55 + lf*RISE);
  const X=ax*cy-az*sy, Z=ax*sy+az*cy, Yc=ay-camY;
  const zc=Yc*sp+Z*cp+dist, f=860/(zc>1?zc:1);
  out.x=W*.5+X*f; out.y=H*.5+(Yc*cp-Z*sp)*f+40; out.z=zc; out.f=f;
  return out}

/* interpoliert Layer-Punktwolke bei fraktionalem lf */
function cloudAt(key,lf){
  const arr=D[key], n=arr.length;
  const i0=Math.max(0,Math.min(n-1,Math.floor(lf)));
  const i1=Math.max(0,Math.min(n-1,i0+1));
  const k=lf-i0, ke=k*k*(3-2*k);
  const A=arr[i0],B=arr[i1],T=A.length,out=new Array(T);
  for(let t=0;t<T;t++){
    const a=A[t],b=B[t];
    out[t]=[a[0]+(b[0]-a[0])*ke, a[1]+(b[1]-a[1])*ke,
            (a[2]||0)+((b[2]||0)-(a[2]||0))*ke]}
  return out}

/* ════════════════════════════════════════════════════════════════════
   4 · RENDER EINER WOLKE
   ════════════════════════════════════════════════════════════════════ */
const _p={};
function drawCloud(key,lf,hue,op,front,tokens){
  const P=cloudAt(key,lf), T=P.length;
  const t=LAYERS>1?lf/(LAYERS-1):0;
  const S=new Array(T);
  for(let i=0;i<T;i++){const o={};project(P[i],lf,o);S[i]=o}

  /* ── Delaunay-freies Nachbarschafts-Netz (Rips-Komplex) ── */
  const R=172*(1-t*.18);
  x.lineWidth=.6;
  for(let i=0;i<T;i++)for(let j=i+1;j<T;j++){
    const ux=S[i].x-S[j].x, uy=S[i].y-S[j].y;
    const d2=ux*ux+uy*uy, r=R*(.45+.55*S[i].f);
    if(d2>r*r)continue;
    const dn=Math.sqrt(d2)/r;
    const al=(.06+.30*(1-dn))*op*(.42+.58*t);
    if(al<.015)continue;
    x.globalAlpha=al>.42?.42:al;
    x.strokeStyle=`hsl(${hue+26},${48+t*16}%,${46+(1-dn)*20}%)`;
    x.beginPath();x.moveTo(S[i].x,S[i].y);x.lineTo(S[j].x,S[j].y);x.stroke()}

  /* ── Sequenz-Pfad: Token-Reihenfolge ── */
  if(T>1){
    x.globalAlpha=op*(.20+.26*t);
    x.strokeStyle=`hsl(${hue},62%,62%)`;
    x.lineWidth=front?1.25:.7;
    x.beginPath();x.moveTo(S[0].x,S[0].y);
    for(let i=1;i<T;i++)x.lineTo(S[i].x,S[i].y);
    x.stroke()}

  /* ── Knoten ── */
  const order=S.map((s,i)=>i).sort((a,b)=>S[b].z-S[a].z);
  for(const i of order){
    const s=S[i], r=(1.3+s.f*2.6);
    const al=(.34+t*.3)*op;
    x.globalAlpha=al>1?1:al;
    x.fillStyle=`hsl(${hue+i*1.5},${52+t*18}%,${58+t*10}%)`;
    x.beginPath();x.arc(s.x,s.y,r,0,6.283);x.fill();
    if(front&&op>.9){
      const rr=3.5+s.f*7;
      const g=x.createRadialGradient(s.x,s.y,0,s.x,s.y,rr);
      g.addColorStop(0,`hsla(${hue+40},90%,74%,${.17*op})`);
      g.addColorStop(1,`hsla(${hue+40},90%,66%,0)`);
      x.globalCompositeOperation='lighter';
      x.fillStyle=g;x.beginPath();x.arc(s.x,s.y,rr,0,6.283);x.fill();
      x.globalCompositeOperation='source-over'}}

  /* ── Token-Labels nur im Frontlayer, nur nah ── */
  if(front&&tokens&&op>.9){
    x.font='9px ui-monospace,monospace';
    x.textAlign='center';
    for(let i=0;i<T;i++){
      const s=S[i]; if(s.f<.62)continue;
      x.globalAlpha=Math.min(.78,(s.f-.55)*2.0);
      x.fillStyle=`hsl(${hue+18},46%,76%)`;
      x.fillText(tokens[i]||'',s.x,s.y-7)}}
  x.globalAlpha=1}

/* ════════════════════════════════════════════════════════════════════
   5 · SLIDER
   ════════════════════════════════════════════════════════════════════ */
const rail=document.getElementById('rail'),knob=document.getElementById('knob'),
      prog=document.getElementById('prog');
let DOTS=[];
function buildRail(){
  [...rail.querySelectorAll('.dot')].forEach(d=>d.remove());
  DOTS=[];
  for(let i=0;i<LAYERS;i++){
    const d=document.createElement('div'); d.className='dot';
    d.style.bottom=(LAYERS>1?i/(LAYERS-1)*100:0)+'%';
    const step=Math.max(1,Math.ceil(LAYERS/18));
    const lbl=(i%step===0||i===LAYERS-1)?(i?String(i).padStart(2,'0'):'EMB'):'';
    d.innerHTML=`<label>${lbl}</label>`;
    d.onpointerdown=e=>{e.stopPropagation();targetPos=i;playing=false;sync()};
    rail.appendChild(d);DOTS.push(d)}}
rail.onpointerdown=e=>{scrub=1;const r=rail.getBoundingClientRect();
  pos=(1-Math.max(0,Math.min(1,(e.clientY-r.top)/r.height)))*(LAYERS-1);
  targetPos=null;playing=false;sync()};

function updateRail(){
  const u=LAYERS>1?pos/(LAYERS-1):0;
  knob.style.bottom=(u*100)+'%';
  prog.style.height=(u*100)+'%';
  for(let i=0;i<LAYERS;i++){
    const d=DOTS[i]; if(!d)continue;
    const act=Math.abs(pos-i)<.34;
    if(d._a!==act){d.classList.toggle('act',act);d._a=act}
    const pass=i<=pos+.05;
    if(d._p!==pass){d.classList.toggle('pass',pass);d._p=pass}}}

/* ════════════════════════════════════════════════════════════════════
   6 · TRANSPORT
   ════════════════════════════════════════════════════════════════════ */
const bPlay=document.getElementById('bPlay'),bView=document.getElementById('bView');
function sync(){bPlay.textContent=playing?'❚❚':'▶';bPlay.classList.toggle('on',playing)}
bPlay.onclick=()=>{if(!D)return;playing=!playing;targetPos=null;sync()};
document.getElementById('bPrev').onclick=()=>{playing=false;
  targetPos=Math.max(0,Math.round(pos)-1);sync()};
document.getElementById('bNext').onclick=()=>{playing=false;
  targetPos=Math.min(LAYERS-1,Math.round(pos)+1);sync()};
bView.onclick=()=>{view=(view+1)%3;
  bView.textContent=['◐','◑','◓'][view];
  bView.classList.toggle('on',view!==2)};
addEventListener('keydown',e=>{
  if(e.target.tagName==='TEXTAREA'||e.target.tagName==='SELECT')return;
  if(e.code==='Space'){e.preventDefault();if(D){playing=!playing;targetPos=null;sync()}return}
  if(e.code==='ArrowUp'||e.code==='ArrowRight'){e.preventDefault();playing=false;
    targetPos=Math.min(LAYERS-1,Math.round(pos)+1);sync();return}
  if(e.code==='ArrowDown'||e.code==='ArrowLeft'){e.preventDefault();playing=false;
    targetPos=Math.max(0,Math.round(pos)-1);sync();return}
  if(e.code==='Home'){e.preventDefault();playing=false;targetPos=0;sync();return}
  if(e.code==='End'){e.preventDefault();playing=false;targetPos=LAYERS-1;sync();return}
  if(e.code==='KeyV'){bView.onclick()}
  if(e.code==='Enter'&&(e.metaKey||e.ctrlKey))run()});

/* ════════════════════════════════════════════════════════════════════
   7 · SPARKLINES
   ════════════════════════════════════════════════════════════════════ */
function sparkSetup(cv){
  const r=cv.getBoundingClientRect();
  const dpr=Math.min(devicePixelRatio||1,2);
  cv.width=Math.max(1,r.width*dpr); cv.height=Math.max(1,r.height*dpr);
  const g=cv.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0);
  g.clearRect(0,0,r.width,r.height); return [g,r.width,r.height]}

function sparkLines(cv,series,markIdx){
  const [g,w,h]=sparkSetup(cv);
  let lo=Infinity,hi=-Infinity;
  for(const s of series)for(const v of s.v){if(v<lo)lo=v;if(v>hi)hi=v}
  if(!isFinite(lo)){return}
  if(hi-lo<1e-9){hi=lo+1}
  const pad=4, n=series[0].v.length;
  const X=i=>pad+(n>1?i/(n-1):0)*(w-2*pad);
  const Y=v=>h-pad-((v-lo)/(hi-lo))*(h-2*pad);
  // Layer-Marker
  if(markIdx!==undefined){
    g.strokeStyle='rgba(150,175,235,.22)';g.lineWidth=1;
    g.beginPath();g.moveTo(X(markIdx),0);g.lineTo(X(markIdx),h);g.stroke()}
  for(const s of series){
    g.beginPath();
    for(let i=0;i<s.v.length;i++){const px=X(i),py=Y(s.v[i]);
      i?g.lineTo(px,py):g.moveTo(px,py)}
    g.strokeStyle=s.c;g.lineWidth=s.w||1.4;g.lineJoin='round';g.stroke();
    if(markIdx!==undefined&&s.v[markIdx]!==undefined){
      g.fillStyle=s.c;g.beginPath();
      g.arc(X(markIdx),Y(s.v[markIdx]),2.1,0,6.283);g.fill()}}}

function sparkBars(cv,a,b){
  const [g,w,h]=sparkSetup(cv);
  const n=Math.max(a.length,b.length), pad=3;
  const bw=(w-2*pad)/n;
  const mx=Math.max(...a,...b,1e-9);
  for(let i=0;i<n;i++){
    const ha=((a[i]||0)/mx)*(h-2*pad), hb=((b[i]||0)/mx)*(h-2*pad);
    const px=pad+i*bw;
    g.fillStyle='rgba(111,211,247,.62)';
    g.fillRect(px+bw*.12,h-pad-ha,bw*.36,ha);
    g.fillStyle='rgba(247,127,176,.62)';
    g.fillRect(px+bw*.52,h-pad-hb,bw*.36,hb)}}

/* ════════════════════════════════════════════════════════════════════
   8 · HUD-UPDATE
   ════════════════════════════════════════════════════════════════════ */
const $=id=>document.getElementById(id);
const f3=v=>(v===undefined||v===null)?'—':(Math.abs(v)<10?v.toFixed(3):v.toFixed(1));
let lastHudLayer=-1;

function updateHUD(){
  if(!D)return;
  const L=Math.round(Math.max(0,Math.min(LAYERS-1,pos)));
  if(L===lastHudLayer)return;
  lastHudLayer=L;

  const sa=D.shapes_a[L], sb=D.shapes_b[L];
  $('s-emd').textContent   = f3(D.emds[L]);
  $('s-drift').textContent = f3(D.drift[L]);
  $('s-ea').textContent    = f3(sa.eff_dim);
  $('s-eb').textContent    = f3(sb.eff_dim);
  $('s-aa').textContent    = f3(sa.anisotropy);
  $('s-ab').textContent    = f3(sb.anisotropy);
  $('s-ca').textContent    = f3(sa.curvature);
  $('s-cb').textContent    = f3(sb.curvature);
  $('s-na').textContent    = f3(D.norms_a[L]);
  $('s-nb').textContent    = f3(D.norms_b[L]);

  $('hs').textContent = (L===0?'EMBEDDING':'BLOCK '+String(L).padStart(2,'0'))
                        +' · W₁ '+f3(D.emds[L]);

  sparkLines($('sp1'),[
    {v:D.emds, c:'#f77fb0', w:1.5},
    {v:D.drift,c:'#7ee7a8', w:1.2}],L);
  sparkLines($('sp2'),[
    {v:D.shapes_a.map(s=>s.eff_dim), c:'#6fd3f7', w:1.5},
    {v:D.shapes_b.map(s=>s.eff_dim), c:'#f77fb0', w:1.5}],L);
  sparkBars($('sp3'), sa.spectrum, sb.spectrum);
}

/* ════════════════════════════════════════════════════════════════════
   9 · TOKEN-LEISTE
   ════════════════════════════════════════════════════════════════════ */
function buildTokens(){
  const mk=(arr,cls,host)=>{
    host.innerHTML='';
    arr.slice(0,38).forEach(t=>{
      const e=document.createElement('div');
      e.className='tk '+cls; e.textContent=t; host.appendChild(e)})};
  mk(D.tok_a,'a',$('trowA'));
  mk(D.tok_b,'b',$('trowB'));
}

/* ════════════════════════════════════════════════════════════════════
   10 · HINTERGRUND
   ════════════════════════════════════════════════════════════════════ */
function background(){
  x.fillStyle='#070910'; x.fillRect(0,0,W,H);
  const vg=x.createRadialGradient(W*.5,H*.52,0,W*.5,H*.52,Math.max(W,H)*.7);
  vg.addColorStop(0,'rgba(20,27,46,.55)');
  vg.addColorStop(1,'rgba(4,5,11,0)');
  x.fillStyle=vg; x.fillRect(0,0,W,H);

  /* Layer-Horizonte: dünne Ellipsen als räumliche Referenz */
  if(!D)return;
  const o={};
  for(let L=0;L<LAYERS;L++){
    const d=Math.abs(L-pos);
    if(d>4.2)continue;
    const al=Math.max(0,(1-d/4.2))*.11;
    if(al<.008)continue;
    x.globalAlpha=al; x.strokeStyle='hsl(218,34%,58%)'; x.lineWidth=.6;
    x.beginPath();
    for(let k=0;k<=46;k++){
      const a=k/46*6.2832;
      project([Math.cos(a)*1.18,0,Math.sin(a)*1.18],L,o);
      k?x.lineTo(o.x,o.y):x.moveTo(o.x,o.y)}
    x.closePath(); x.stroke()}
  x.globalAlpha=1;
}

/* ════════════════════════════════════════════════════════════════════
   11 · LOOP
   ════════════════════════════════════════════════════════════════════ */
const hn=$('hn');
let last=performance.now();
function loop(now){
  const dt=Math.min(.05,(now-last)/1000); last=now;
  const ease=(cur,t_,sp)=>cur+(t_-cur)*(1-Math.exp(-sp*dt));

  yaw=ease(yaw,tg.y,6); pit=ease(pit,tg.p,6);
  if(!dg&&scrub===null)tg.y+=.030*dt;

  if(D){
    if(targetPos!==null){
      pos=ease(pos,targetPos,7);
      if(Math.abs(pos-targetPos)<.004){pos=targetPos;targetPos=null}
    }else if(playing&&scrub===null){
      pos+=SPD*dt;
      if(pos>LAYERS-1){pos=0;camY=0}
    }
    camY=ease(camY,-pos*RISE,3.4);
  }

  background();

  if(D){
    /* Geister-Layer dahinter */
    for(let g=GH;g>=1;g--){
      const lf=pos-g*GAP; if(lf<-.05)continue;
      const op=Math.pow(.34,g)*.78;
      if(view!==1)drawCloud('proj_a',Math.max(0,lf),HUE_A-g*3,op,false,null);
      if(view!==0)drawCloud('proj_b',Math.max(0,lf),HUE_B-g*3,op,false,null)}
    /* Frontlayer */
    if(view!==1)drawCloud('proj_a',pos,HUE_A,1,true,D.tok_a);
    if(view!==0)drawCloud('proj_b',pos,HUE_B,1,true,D.tok_b);

    hn.textContent=String(Math.round(pos)).padStart(2,'0');
    updateRail(); updateHUD();
  }
  requestAnimationFrame(loop)}

/* ════════════════════════════════════════════════════════════════════
   12 · SERVER-CALL
   ════════════════════════════════════════════════════════════════════ */
const msg=$('msg'), go=$('go'), boot=$('boot'), bootmsg=$('bootmsg');
let busy=false;

async function run(){
  if(busy)return; busy=true;
  go.classList.add('busy'); go.textContent='RECHNET …';
  msg.className=''; msg.textContent='Modell laden & Hidden States extrahieren …';
  boot.classList.remove('gone');
  bootmsg.textContent='FORWARD-PASS · PROJEKTION · OPTIMAL TRANSPORT';

  const body={
    text_a:$('ta').value, text_b:$('tb').value,
    model:$('model').value, method:$('method').value, emd:$('emd').value};

  try{
    const r=await fetch('/api/analyse',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j=await r.json();
    if(!r.ok||j.error)throw new Error(j.error||('HTTP '+r.status));

    D=j; LAYERS=D.n_layers;
    RISE = LAYERS>20 ? 120 : LAYERS>12 ? 150 : 185;
    pos=0; camY=0; targetPos=null; playing=true; lastHudLayer=-1;
    buildRail(); buildTokens(); sync();
    msg.textContent=`${D.model.split('/').pop()} · ${LAYERS} Layer · `
      +`${D.tok_a.length}/${D.tok_b.length} Tokens · ${D.device.toUpperCase()}`;
    boot.classList.add('gone');
  }catch(e){
    msg.className='err'; msg.textContent='FEHLER: '+e.message;
    bootmsg.textContent='FEHLER · '+e.message;
    setTimeout(()=>boot.classList.add('gone'),900);
  }finally{
    busy=false; go.classList.remove('busy'); go.textContent='ANALYSIEREN';
  }
}
go.onclick=run;

/* Start */
sync();
requestAnimationFrame(loop);
setTimeout(()=>boot.classList.add('gone'),600);
</script>
</body>
</html>
""".replace("__OPTS__", opts)

# ============================================================================
# 9) HTTP-SERVER
# ============================================================================

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        if "/api/" in str(args):
            print(f"[http] {args[0]}")

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, build_html().encode("utf-8"),
                       "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/analyse":
            self._send(404, b'{"error":"unknown endpoint"}', "application/json")
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            ta = (req.get("text_a") or "").strip() or "Hello world."
            tb = (req.get("text_b") or "").strip() or "Goodbye world."
            model = req.get("model") or "distilbert-base-uncased"
            method = req.get("method") or "pca"
            emd = req.get("emd") or "exact"

            print(f"[api] {model} | {method} | {emd}")
            data = analyse(ta, tb, model, method, emd, dims=3)
            self._send(200, json.dumps(data).encode("utf-8"),
                       "application/json; charset=utf-8")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send(500, json.dumps({"error": str(e)}).encode("utf-8"),
                       "application/json")

# ============================================================================
# 10) ENTRY POINT
# ============================================================================

def main():
    port = int(os.environ.get("MANIFOLD_PORT", "8051"))
    url = f"http://127.0.0.1:{port}"
    print("=" * 72)
    print("  HIDDEN-STATE MANIFOLD EXPLORER  ·  v2")
    print("=" * 72)
    print(f"  Device   : {DEVICE.upper()}")
    print(f"  CUDA     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU      : {torch.cuda.get_device_name(0)}")
    cpu_ok = sum(1 for v in MODELS.values() if v["tier"].startswith("CPU"))
    print(f"  Modelle  : {len(MODELS)}  ({cpu_ok} CPU-tauglich)")
    print(f"  URL      : {url}")
    print("=" * 72)
    print("  Hinweis  : erster Lauf laedt das Modell -> dauert etwas.")
    print()

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[exit] Server gestoppt.")
        srv.server_close()

if __name__ == "__main__":
    main()
