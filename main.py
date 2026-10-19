#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch", "transformers", "numpy", "scikit-learn",
#   "pot", "umap-learn", "accelerate",
# ]
# ///
"""
Hidden-State Manifold Explorer  ·  v3
=====================================
Echte Vietoris-Rips-Flaeche · freie Reduktions-Pipeline · manuelles Layer-Stepping

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
import itertools
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
import ot
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.random_projection import GaussianRandomProjection
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
def hidden_states(text: str, model_name: str, max_len: int = 192):
    tok, model = load_model(model_name)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    out = model(**enc)
    hs = torch.stack(out.hidden_states)[:, 0].float().cpu().numpy()  # [L+1,T,D]
    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
    clean = [t.replace("Ġ", "·").replace("▁", "·").replace("##", "") for t in tokens]
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
# 6) REDUKTIONS-PIPELINE  (frei konfigurierbare Kette)
# ============================================================================
#
#  Eine Stufe ist {"method": "tsne", "dims": 3}.
#  Die Kette wird der Reihe nach angewendet:
#      X [T, D]  ->  Stufe1  ->  Stufe2  ->  ...  ->  Z [T, 3]
#
#  Beispiel: [{"umap":10}, {"pca":3}]  bzw.  [{"pca":50}, {"tsne":3}]
#
#  Gueltige Methoden siehe REDUCERS.
# ============================================================================

REDUCERS = {
    "pca":    {"label": "PCA",            "max_out": None, "note": "linear, varianzmaximal"},
    "svd":    {"label": "TRUNCATED SVD",  "max_out": None, "note": "linear, ohne Zentrierung"},
    "rand":   {"label": "RANDOM PROJ.",   "max_out": None, "note": "Johnson-Lindenstrauss"},
    "tsne":   {"label": "T-SNE",          "max_out": 3,    "note": "lokal, nicht-linear"},
    "umap":   {"label": "UMAP",           "max_out": None, "note": "lokal+global, schnell"},
    "isomap": {"label": "ISOMAP",         "max_out": None, "note": "geodaetisch"},
    "mds":    {"label": "MDS",            "max_out": None, "note": "distanz-erhaltend"},
    "none":   {"label": "— KEINE —",      "max_out": None, "note": "Stufe ueberspringen"},
}

def _one_stage(X: np.ndarray, method: str, dims: int, seed: int = 0) -> np.ndarray:
    """Eine einzelne Reduktionsstufe. Robust gegen zu kleine n/d."""
    n, d = X.shape
    if method == "none" or dims <= 0:
        return X

    # Zieldimension hart begrenzen
    cap = REDUCERS.get(method, {}).get("max_out")
    if cap:
        dims = min(dims, cap)
    dims = max(1, min(dims, d))
    if n > 1:
        dims = min(dims, n - 1) if method in ("pca", "tsne", "isomap") else dims
    dims = max(1, dims)

    if dims >= d and method in ("pca", "svd", "rand"):
        return X  # nichts zu tun

    try:
        if method == "pca":
            return PCA(n_components=dims, random_state=seed).fit_transform(X)

        if method == "svd":
            return TruncatedSVD(n_components=min(dims, d - 1),
                                random_state=seed).fit_transform(X)

        if method == "rand":
            return GaussianRandomProjection(n_components=dims,
                                            random_state=seed).fit_transform(X)

        if method == "tsne":
            perp = max(2.0, min(30.0, (n - 1) / 3.0))
            return TSNE(n_components=min(dims, 3), perplexity=perp,
                        init="pca", random_state=seed,
                        max_iter=500).fit_transform(X)

        if method == "umap":
            import umap
            nn = max(2, min(15, n - 1))
            return umap.UMAP(n_components=dims, n_neighbors=nn, min_dist=0.08,
                             random_state=seed).fit_transform(X)

        if method == "isomap":
            nn = max(2, min(10, n - 1))
            return Isomap(n_components=dims, n_neighbors=nn).fit_transform(X)

        if method == "mds":
            return MDS(n_components=dims, random_state=seed,
                       normalized_stress="auto").fit_transform(X)

    except Exception as e:                       # Fallback: PCA
        print(f"[reduce] '{method}' fehlgeschlagen ({e}) -> PCA-Fallback")
        return PCA(n_components=max(1, min(dims, d, n - 1)),
                   random_state=seed).fit_transform(X)

    return X

def run_pipeline(X: np.ndarray, pipeline: list, seed: int = 0) -> np.ndarray:
    """Wendet die komplette Kette an und garantiert am Ende genau 3 Spalten."""
    Z = np.asarray(X, dtype=np.float64)
    for stage in pipeline:
        m = stage.get("method", "pca")
        d = int(stage.get("dims", 3))
        if m == "none":
            continue
        Z = _one_stage(Z, m, d, seed=seed)

    # auf exakt 3D bringen
    if Z.shape[1] > 3:
        Z = PCA(n_components=3, random_state=seed).fit_transform(Z)
    elif Z.shape[1] < 3:
        Z = np.hstack([Z, np.zeros((Z.shape[0], 3 - Z.shape[1]))])
    return Z

def joint_pipeline(blocks, pipeline, seed=0):
    """Gemeinsame Reduktion beider Saetze -> vergleichbares Koordinatensystem."""
    sizes = [b.shape[0] for b in blocks]
    Z = run_pipeline(np.vstack(blocks), pipeline, seed=seed)
    out, i = [], 0
    for s in sizes:
        out.append(Z[i:i + s])
        i += s
    return out

def procrustes_align(P, ref):
    """Rotiert/spiegelt P moeglichst auf ref -> kein Achsen-Umklappen pro Layer."""
    if ref is None or P.shape != ref.shape:
        return P
    A = P - P.mean(0)
    B = ref - ref.mean(0)
    U, _, Vt = np.linalg.svd(A.T @ B)
    R = U @ Vt
    return A @ R + ref.mean(0)

# ============================================================================
# 7) VIETORIS-RIPS  —  echter Simplizialkomplex im Originalraum
# ============================================================================
#
#  Idee (nach Latschev / Adams):  Fuer eine Punktwolke, die nahe an einer
#  Mannigfaltigkeit liegt, rekonstruiert VR(X, eps) die Topologie korrekt,
#  wenn eps im "Goldlöckchen-Fenster" liegt:
#     zu klein -> Staub,  zu gross -> alles kollabiert.
#  Wir waehlen eps datengetrieben aus den kNN-Distanzen.
# ============================================================================

def _pairwise(X):
    G = X @ X.T
    sq = np.diag(G)
    D2 = sq[:, None] + sq[None, :] - 2 * G
    np.maximum(D2, 0, out=D2)
    return np.sqrt(D2, out=D2)

def rips_epsilon(D: np.ndarray, k: int = 4, density: float = 1.0,
                 pct: float = 72.0) -> float:
    """eps aus der k-ten Nachbardistanz: 'gerade so gross, dass es passt'."""
    n = D.shape[0]
    if n < 3:
        return float(D.max()) if n > 1 else 1.0
    kk = int(max(1, min(k, n - 1)))
    knn = np.partition(D + np.eye(n) * 1e9, kk, axis=1)[:, kk]
    base = float(np.percentile(knn, pct))
    return max(base * float(density), 1e-9)

def rips_complex(X: np.ndarray, density: float = 1.0, k: int = 4,
                 max_tris: int = 2600):
    """
    Liefert (edges, triangles, eps, stats) des Vietoris-Rips-Komplexes.
    edges     : [[i,j], ...]            1-Simplizes
    triangles : [[i,j,k], ...]          2-Simplizes  -> die FLAECHE
    """
    n = X.shape[0]
    if n < 3:
        return [], [], 0.0, dict(n_edges=0, n_tris=0, eps=0.0, euler=n)

    D = _pairwise(np.asarray(X, dtype=np.float64))
    eps = rips_epsilon(D, k=k, density=density)

    adj = (D <= eps) & ~np.eye(n, dtype=bool)

    # --- Kanten ---
    ei, ej = np.where(np.triu(adj, 1))
    edges = np.stack([ei, ej], 1)

    # --- Dreiecke: alle Tripel mit paarweisem Abstand <= eps ---
    nbrs = [set(np.where(adj[i])[0].tolist()) for i in range(n)]
    tris = []
    for i, j in zip(ei, ej):
        if j <= i:
            continue
        common = nbrs[i] & nbrs[j]
        for kk in common:
            if kk > j:
                tris.append((int(i), int(j), int(kk)))
        if len(tris) > max_tris * 3:
            break

    # Bei Überfüllung: kleinste Dreiecke (kompakteste) bevorzugen
    if len(tris) > max_tris:
        diam = np.array([max(D[a, b], D[b, c], D[a, c]) for a, b, c in tris])
        keep = np.argsort(diam)[:max_tris]
        tris = [tris[q] for q in keep]

    V, E, F = n, len(edges), len(tris)
    return (edges.tolist(), [list(t) for t in tris], float(eps),
            dict(n_edges=E, n_tris=F, eps=float(eps), euler=int(V - E + F)))

# ============================================================================
# 8) Analyse-Pipeline
# ============================================================================

def analyse(ta, tb, model_name, pipeline, emd_mode,
            density=1.0, rips_k=4):
    tok_a, HA = hidden_states(ta, model_name)
    tok_b, HB = hidden_states(tb, model_name)
    n_layers = HA.shape[0]

    proj_a, proj_b = [], []
    rips_a, rips_b = [], []
    emds, sh_a, sh_b = [], [], []
    prev_a = prev_b = None

    for L in range(n_layers):
        A, B = HA[L], HB[L]

        # --- Vietoris-Rips IM ORIGINALRAUM, nicht im Bildraum ---------------
        ea, ta_, epsA, statA = rips_complex(A, density=density, k=rips_k)
        eb, tb_, epsB, statB = rips_complex(B, density=density, k=rips_k)
        rips_a.append(dict(edges=ea, tris=ta_, **statA))
        rips_b.append(dict(edges=eb, tris=tb_, **statB))

        # --- gemeinsame Reduktionskette -> vergleichbares Koordinatensystem --
        pa, pb = joint_pipeline([A, B], pipeline, seed=0)
        pa = procrustes_align(pa, prev_a)
        pb = procrustes_align(pb, prev_b)
        prev_a, prev_b = pa, pb
        proj_a.append(pa)
        proj_b.append(pb)

        emds.append(wasserstein(A, B, mode=emd_mode))
        sh_a.append(shape_metrics(A))
        sh_b.append(shape_metrics(B))

    # --- gemeinsame Normierung ueber ALLE Layer -> stabile Kamera ----------
    allp = np.vstack(proj_a + proj_b)
    centre = allp.mean(0)
    scale = float(np.abs(allp - centre).max()) + 1e-9
    proj_a = [((p - centre) / scale).round(4).tolist() for p in proj_a]
    proj_b = [((p - centre) / scale).round(4).tolist() for p in proj_b]

    drift = [0.0] + [wasserstein(HA[L - 1], HA[L], mode=emd_mode)
                     for L in range(1, n_layers)]
    norms_a = [float(np.linalg.norm(HA[L], axis=1).mean()) for L in range(n_layers)]
    norms_b = [float(np.linalg.norm(HB[L], axis=1).mean()) for L in range(n_layers)]

    pipe_label = " → ".join(
        f'{REDUCERS.get(s["method"], {}).get("label", s["method"])}·{s["dims"]}D'
        for s in pipeline if s.get("method") != "none") or "PCA·3D"

    return dict(tok_a=tok_a, tok_b=tok_b,
                proj_a=proj_a, proj_b=proj_b,
                rips_a=rips_a, rips_b=rips_b,
                emds=emds, drift=drift,
                shapes_a=sh_a, shapes_b=sh_b,
                norms_a=norms_a, norms_b=norms_b,
                n_layers=n_layers, model=model_name,
                pipeline=pipeline, pipe_label=pipe_label,
                density=density, rips_k=rips_k,
                text_a=ta, text_b=tb, device=DEVICE)

# ============================================================================
# 9) FRONTEND
# ============================================================================

def build_html() -> str:
    opts = "".join(
        f'<option value="{k}">[{v["tier"]}] {k} · {v["size"]} · {v["note"]}</option>'
        for k, v in MODELS.items())

    red_opts = "".join(
        f'<option value="{k}">{v["label"]}</option>' for k, v in REDUCERS.items())

    return r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>hidden-state manifold · v3</title>
<style>
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:#05070d;overflow:hidden;
    font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:#79829f}
  canvas{display:block;position:fixed;inset:0}
  ::-webkit-scrollbar{width:5px}
  ::-webkit-scrollbar-track{background:transparent}
  ::-webkit-scrollbar-thumb{background:rgba(120,140,200,.16);border-radius:3px}
  ::-webkit-scrollbar-thumb:hover{background:rgba(140,165,225,.3)}

  /* ── Konsole links ───────────────────── */
  #ui{position:fixed;left:22px;top:22px;z-index:14;width:316px;
    max-height:calc(100vh - 44px);overflow-y:auto;padding-right:6px}
  .card{background:linear-gradient(165deg,rgba(17,22,36,.62),rgba(10,13,22,.5));
    border:1px solid rgba(120,140,200,.09);border-radius:12px;
    padding:14px 15px;margin-bottom:9px;backdrop-filter:blur(18px) saturate(1.3);
    box-shadow:0 8px 32px -16px rgba(0,0,0,.9);
    transition:.4s cubic-bezier(.22,1,.36,1)}
  .card:hover{border-color:rgba(150,175,235,.19)}
  .card.accent{border-color:rgba(111,211,247,.16)}
  .lb{display:block;font-size:8px;letter-spacing:.32em;opacity:.5;
    margin-bottom:7px;color:#939ec2;text-transform:uppercase}
  .lb em{font-style:normal;float:right;opacity:.7;letter-spacing:.12em;
    color:var(--a)}
  textarea,select,input[type=text]{width:100%;background:rgba(5,7,13,.72);
    color:#c3cbe4;border:1px solid rgba(120,140,200,.13);border-radius:7px;
    padding:8px 10px;font:11px/1.45 ui-monospace,monospace;outline:none;
    resize:none;transition:.28s}
  textarea:focus,select:focus{border-color:var(--a);color:#e8edfa;
    box-shadow:0 0 0 3px rgba(111,211,247,.06)}
  textarea{height:50px}
  select{height:29px;cursor:pointer;appearance:none;
    background-image:linear-gradient(45deg,transparent 50%,#6b7593 50%),
      linear-gradient(135deg,#6b7593 50%,transparent 50%);
    background-position:calc(100% - 14px) 13px,calc(100% - 9px) 13px;
    background-size:5px 5px;background-repeat:no-repeat}
  select option{background:#0a0d16;color:#c3cbe4}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:8px}
  .row3{display:grid;grid-template-columns:1fr 54px 26px;gap:6px;
    align-items:center;margin-bottom:6px}
  .row3 .ix{font-size:8px;letter-spacing:.2em;color:#3f4765;text-align:center}

  /* ── Pipeline-Stufen ─────────────────── */
  #stages{margin-bottom:4px}
  .stage{position:relative;padding-left:14px}
  .stage:before{content:'';position:absolute;left:4px;top:-4px;bottom:50%;
    width:1px;background:rgba(120,140,200,.16)}
  .stage:after{content:'';position:absolute;left:4px;top:50%;width:7px;
    height:1px;background:rgba(120,140,200,.16)}
  .stage:first-child:before{top:50%}
  .stage:last-child:before{bottom:50%}
  .mini{width:100%;padding:6px;border-radius:6px;cursor:pointer;
    background:rgba(16,21,34,.6);border:1px dashed rgba(120,140,200,.18);
    color:#6c7695;font:9px/1 ui-monospace,monospace;letter-spacing:.2em;
    transition:.25s;text-align:center}
  .mini:hover{border-color:var(--a);color:#cdd6ee;border-style:solid}
  .mini.rm{border-color:rgba(224,101,127,.2)}
  .mini.rm:hover{border-color:#e0657f;color:#f2b6c3}
  #pipeHint{font-size:8px;letter-spacing:.14em;color:#3e4563;margin-top:7px;
    line-height:1.6}

  #go{width:100%;margin-top:4px;padding:12px;border-radius:9px;cursor:pointer;
    background:linear-gradient(145deg,rgba(26,33,52,.9),rgba(16,21,34,.9));
    border:1px solid var(--a);color:#e8edfa;
    font:10px/1 ui-monospace,monospace;letter-spacing:.3em;text-align:center;
    box-shadow:0 0 26px -10px var(--a);transition:.3s}
  #go:hover{background:var(--a);color:#05070d;letter-spacing:.36em;
    box-shadow:0 0 34px -6px var(--a)}
  #go.busy{opacity:.5;pointer-events:none;letter-spacing:.2em}
  #msg{font-size:8.5px;letter-spacing:.14em;margin-top:9px;min-height:12px;
    color:#6d7795;line-height:1.7}
  #msg.err{color:#e0657f}

  /* ── Kennzahlen ──────────────────────── */
  .stat{display:flex;justify-content:space-between;align-items:baseline;
    padding:4.5px 0;border-bottom:1px solid rgba(120,140,200,.06)}
  .stat:last-child{border:0;padding-bottom:0}
  .stat>span{font-size:8px;letter-spacing:.2em;color:#5c6688}
  .stat>b{font-weight:400;font-size:12px;color:#d2daf0;
    font-variant-numeric:tabular-nums;letter-spacing:.02em}
  .va{color:#6fd3f7}.vb{color:#f77fb0}
  .sep{opacity:.25;padding:0 3px}

  /* ── Header ──────────────────────────── */
  #h{position:fixed;right:112px;top:24px;text-align:right;z-index:10;
    font-size:9px;letter-spacing:.3em;color:#3f4765;pointer-events:none}
  #h b{display:block;font-size:46px;letter-spacing:.02em;color:#d6deef;
    font-weight:200;margin:-4px 0 0;font-variant-numeric:tabular-nums;
    text-shadow:0 0 40px rgba(111,211,247,.14)}
  #h i{font-style:normal;font-size:8px;letter-spacing:.22em;color:#353c57;
    display:block;margin-top:1px}
  #h u{text-decoration:none;font-size:8px;letter-spacing:.2em;color:#2e3449;
    display:block;margin-top:5px}

  /* ── Rail ────────────────────────────── */
  #rail{position:fixed;right:44px;top:50%;transform:translateY(-50%);
    width:46px;height:60vh;z-index:12;cursor:pointer}
  #line{position:absolute;left:23px;top:0;bottom:0;width:1px;
    background:linear-gradient(180deg,rgba(255,255,255,.02),
      rgba(255,255,255,.13),rgba(255,255,255,.02))}
  #prog{position:absolute;left:22px;bottom:0;width:3px;border-radius:2px;
    background:linear-gradient(180deg,var(--a),rgba(111,211,247,0));opacity:.65}
  .dot{position:absolute;left:19px;width:9px;height:9px;margin-top:-4.5px;
    border-radius:50%;background:#0c0f1a;border:1px solid rgba(150,170,220,.24);
    transition:.35s cubic-bezier(.22,1,.36,1)}
  .dot.pass{border-color:rgba(160,190,245,.46);background:#19202f}
  .dot.act{background:var(--a);border-color:var(--a);
    box-shadow:0 0 16px var(--a);transform:scale(1.55)}
  .dot label{position:absolute;right:17px;top:-4px;font-size:7.5px;
    letter-spacing:.2em;color:#333a54;white-space:nowrap;pointer-events:none;
    transition:.3s}
  .dot.act label{color:#cfd8ef;letter-spacing:.26em}
  #knob{position:absolute;left:14px;width:19px;height:19px;margin-top:-9.5px;
    border-radius:50%;border:1px solid var(--a);background:rgba(8,11,20,.88);
    box-shadow:0 0 24px -3px var(--a);pointer-events:none;
    transition:box-shadow .3s}
  #knob:after{content:'';position:absolute;inset:5px;border-radius:50%;
    background:var(--a)}

  /* ── Transport ───────────────────────── */
  #tp{position:fixed;right:28px;bottom:24px;display:flex;gap:6px;z-index:12}
  .btn{width:31px;height:31px;border-radius:8px;cursor:pointer;
    background:rgba(14,18,30,.62);border:1px solid rgba(120,140,200,.13);
    color:#8e98ba;font-size:11px;display:flex;align-items:center;
    justify-content:center;backdrop-filter:blur(12px);transition:.25s;
    user-select:none}
  .btn:hover{border-color:rgba(160,185,240,.34);color:#dfe6fa;
    transform:translateY(-1px)}
  .btn.on{border-color:var(--a);color:#eaf0ff;box-shadow:0 0 18px -5px var(--a)}

  /* ── Tokens ──────────────────────────── */
  #toks{position:fixed;left:50%;bottom:20px;transform:translateX(-50%);
    z-index:11;display:flex;flex-direction:column;gap:5px;align-items:center;
    pointer-events:none;max-width:54vw}
  .trow{display:flex;gap:3px;flex-wrap:wrap;justify-content:center}
  .tk{padding:2px 6px;border-radius:4px;font-size:8.5px;letter-spacing:.06em;
    background:rgba(12,16,26,.6);border:1px solid rgba(120,140,200,.1);
    backdrop-filter:blur(8px)}
  .tk.a{color:#6fd3f7;border-color:rgba(111,211,247,.17)}
  .tk.b{color:#f77fb0;border-color:rgba(247,127,176,.17)}

  #ft{position:fixed;left:22px;bottom:20px;font-size:8px;letter-spacing:.26em;
    color:#262b40;z-index:10;pointer-events:none}

  .spark{width:100%;height:44px;display:block;margin-top:4px}
  .legend{display:flex;gap:10px;margin-top:5px;font-size:7.5px;
    letter-spacing:.18em}
  .legend i{font-style:normal;display:flex;align-items:center;gap:4px;
    color:#545e7e}
  .legend u{text-decoration:none;width:8px;height:2px;border-radius:1px;
    display:inline-block}

  #boot{position:fixed;inset:0;z-index:40;background:#05070d;
    display:flex;align-items:center;justify-content:center;flex-direction:column;
    gap:16px;transition:opacity .8s;font-size:9px;letter-spacing:.36em;
    color:#4a5272}
  #boot.gone{opacity:0;pointer-events:none}
  #bootbar{width:200px;height:1px;background:rgba(120,140,200,.12);
    overflow:hidden}
  #bootbar i{display:block;height:100%;width:32%;background:#6fd3f7;
    box-shadow:0 0 12px #6fd3f7;
    animation:sweep 1.3s cubic-bezier(.65,0,.35,1) infinite}
  @keyframes sweep{0%{transform:translateX(-110%)}100%{transform:translateX(340%)}}
</style>
</head>
<body>
<canvas id="c"></canvas>

<div id="ui">
  <div class="card">
    <span class="lb">Satz A <em>cyan</em></span>
    <textarea id="ta">The cat sat quietly on the warm windowsill.</textarea>
  </div>
  <div class="card">
    <span class="lb">Satz B <em style="color:#f77fb0">magenta</em></span>
    <textarea id="tb">Non-abelian cohomology classifies torsors over a site.</textarea>
  </div>

  <div class="card accent">
    <span class="lb">Modell</span>
    <select id="model">__OPTS__</select>

    <div style="height:12px"></div>
    <span class="lb">Reduktions-Pipeline</span>
    <div id="stages"></div>
    <div class="row" style="margin-top:4px">
      <div class="mini" id="addStage">+ STUFE</div>
      <div class="mini rm" id="delStage">− STUFE</div>
    </div>
    <div id="pipeHint">Kette wird von oben nach unten angewandt.<br>
      Letzte Ausgabe wird auf 3D projiziert.</div>

    <div style="height:12px"></div>
    <div class="row">
      <div><span class="lb">Transport</span>
        <select id="emd">
          <option value="exact">EXAKT · EMD</option>
          <option value="sliced">SLICED</option>
          <option value="sinkhorn">SINKHORN</option>
        </select></div>
      <div><span class="lb">RIPS ε · DICHTE</span>
        <select id="density">
          <option value="0.7">ENG · 0.7</option>
          <option value="1.0" selected>PASSEND · 1.0</option>
          <option value="1.3">WEIT · 1.3</option>
          <option value="1.7">SEHR WEIT · 1.7</option>
          <option value="2.4">HÜLLE · 2.4</option>
        </select></div>
    </div>

    <div style="height:11px"></div>
    <div id="go">ANALYSIEREN</div>
    <div id="msg"></div>
  </div>

  <div class="card" id="panelStats">
    <span class="lb">Layer-Kennzahlen</span>
    <div class="stat"><span>W₁ (A,B)</span><b id="s-emd">—</b></div>
    <div class="stat"><span>DRIFT L-1→L</span><b id="s-drift">—</b></div>
    <div class="stat"><span>RIPS ε</span>
      <b><em class="va" id="s-epa">—</em><span class="sep">/</span><em class="vb" id="s-epb">—</em></b></div>
    <div class="stat"><span>2-SIMPLIZES</span>
      <b><em class="va" id="s-tra">—</em><span class="sep">/</span><em class="vb" id="s-trb">—</em></b></div>
    <div class="stat"><span>EULER χ</span>
      <b><em class="va" id="s-eua">—</em><span class="sep">/</span><em class="vb" id="s-eub">—</em></b></div>
    <div class="stat"><span>EFF. DIM</span>
      <b><em class="va" id="s-ea">—</em><span class="sep">/</span><em class="vb" id="s-eb">—</em></b></div>
    <div class="stat"><span>ANISOTROPIE</span>
      <b><em class="va" id="s-aa">—</em><span class="sep">/</span><em class="vb" id="s-ab">—</em></b></div>
    <div class="stat"><span>KRÜMMUNG</span>
      <b><em class="va" id="s-ca">—</em><span class="sep">/</span><em class="vb" id="s-cb">—</em></b></div>
    <div class="stat"><span>‖h‖ MITTEL</span>
      <b><em class="va" id="s-na">—</em><span class="sep">/</span><em class="vb" id="s-nb">—</em></b></div>
  </div>

  <div class="card">
    <span class="lb">Verlauf über Layer</span>
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
    <span class="lb">Spektrum @ Layer</span>
    <canvas id="sp3" class="spark" style="height:58px"></canvas>
  </div>
</div>

<div id="h"><span>LAYER</span><b id="hn">00</b><i id="hs">—</i><u id="hp">—</u></div>

<div id="rail"><div id="line"></div><div id="prog"></div><div id="knob"></div></div>

<div id="tp">
  <div class="btn" id="bPlay">▶</div>
  <div class="btn" id="bPrev">◂</div>
  <div class="btn" id="bNext">▸</div>
  <div class="btn on" id="bSurf">◆</div>
  <div class="btn" id="bView">◐</div>
</div>

<div id="toks"><div class="trow" id="trowA"></div><div class="trow" id="trowB"></div></div>

<div id="ft">DRAG ROTATE · SCROLL ZOOM · ◂ ▸ LAYER · ◆ FLÄCHE · V ANSICHT</div>

<div id="boot"><div>HIDDEN-STATE MANIFOLD · V3</div>
  <div id="bootbar"><i></i></div>
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

const HUE_A=193, HUE_B=330;
document.documentElement.style.setProperty('--a',`hsl(${HUE_A},74%,62%)`);

/* ════════════════════════════════════════════════════════════════════
   1 · STATE  —  pos bewegt sich NUR auf Befehl
   ════════════════════════════════════════════════════════════════════ */
let D=null;
let LAYERS=1, RISE=168, GH=2, GAP=0.62;
let pos=0, playing=false, scrub=null, targetPos=null, view=2, surf=true;
const STEP_SPD=5.2;          // Easing-Rate beim Durchschreiten der Layer
const AUTO_SPD=0.34;         // nur aktiv, wenn ▶ gedrueckt wurde
let yaw=-.68,pit=.26,dist=1180,camY=0,tg={y:-.68,p:.26},dg=null;
let breathe=0;               // minimales Atmen, kein Formwechsel

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
  tg.p=Math.max(-.12,Math.min(1.02,dg.b+(e.clientY-dg.y)*.0042))});
addEventListener('wheel',e=>dist=Math.max(540,Math.min(2600,dist+e.deltaY*.5)),
  {passive:1});

/* ════════════════════════════════════════════════════════════════════
   3 · PROJEKTION
   ════════════════════════════════════════════════════════════════════ */
const SCALE=312;
function project(p,lf,out){
  const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pit),sp=Math.sin(pit);
  const ax=p[0]*SCALE, az=(p[2]!==undefined?p[2]:0)*SCALE*.78;
  const ay=-(p[1]*SCALE*.62 + lf*RISE);
  const X=ax*cy-az*sy, Z=ax*sy+az*cy, Yc=ay-camY;
  const zc=Yc*sp+Z*cp+dist, f=880/(zc>1?zc:1);
  out.x=W*.5+X*f; out.y=H*.5+(Yc*cp-Z*sp)*f+38; out.z=zc; out.f=f;
  return out}

/* Punktwolke bei (fraktionalem) Layer — Form bleibt stabil */
function cloudAt(key,lf){
  const arr=D[key], n=arr.length;
  const i0=Math.max(0,Math.min(n-1,Math.floor(lf)));
  const i1=Math.max(0,Math.min(n-1,i0+1));
  const k=lf-i0, ke=k*k*(3-2*k);
  const A=arr[i0],B=arr[i1],T=A.length,out=new Array(T);
  const br=Math.sin(breathe)*0.006;          // 0.6 % — nur ein Hauch
  for(let t=0;t<T;t++){
    const a=A[t],b=B[t],s=1+br;
    out[t]=[(a[0]+(b[0]-a[0])*ke)*s,
            (a[1]+(b[1]-a[1])*ke)*s,
            ((a[2]||0)+((b[2]||0)-(a[2]||0))*ke)*s]}
  return out}

/* Rips-Komplex des naechstgelegenen ganzzahligen Layers */
function ripsAt(key,lf){
  const arr=D[key];
  const i=Math.max(0,Math.min(arr.length-1,Math.round(lf)));
  return arr[i]}

/* ════════════════════════════════════════════════════════════════════
   4 · FLÄCHEN-RENDERER  —  2-Simplizes, tiefensortiert, deckend
   ════════════════════════════════════════════════════════════════════ */
const _tmp={};
function drawManifold(keyP,keyR,lf,hue,op,front,tokens){
  const P=cloudAt(keyP,lf), T=P.length;
  const RC=ripsAt(keyR,lf);
  const t=LAYERS>1?lf/(LAYERS-1):0;

  /* --- projizieren --- */
  const S=new Array(T);
  let zmin=1e9,zmax=-1e9;
  for(let i=0;i<T;i++){
    const o={};project(P[i],lf,o);S[i]=o;
    if(o.z<zmin)zmin=o.z; if(o.z>zmax)zmax=o.z}
  const zr=1/Math.max(1,zmax-zmin);

  /* ════ 2-SIMPLIZES: die eigentliche Oberfläche ════ */
  if(surf&&RC&&RC.tris&&RC.tris.length){
    const tris=RC.tris, M=tris.length;
    const key=new Float32Array(M), idx=new Int32Array(M);
    const lit=new Float32Array(M);

    for(let m=0;m<M;m++){
      const T3=tris[m],a=S[T3[0]],b=S[T3[1]],cc=S[T3[2]];
      if(!a||!b||!cc){key[m]=1e9;idx[m]=m;continue}
      key[m]=(a.z+b.z+cc.z)/3; idx[m]=m;
      /* Bildschirm-Normale -> Pseudo-Beleuchtung */
      const nz=(b.x-a.x)*(cc.y-a.y)-(b.y-a.y)*(cc.x-a.x);
      let L=nz/5200+.5; lit[m]=L<0?0:L>1?1:L}

    const ord=Array.prototype.slice.call(idx).sort((m,n)=>key[n]-key[m]);

    for(const m of ord){
      const T3=tris[m],a=S[T3[0]],b=S[T3[1]],cc=S[T3[2]];
      if(!a||!b||!cc)continue;
      const dep=1-(key[m]-zmin)*zr, L=lit[m];
      x.beginPath();
      x.moveTo(a.x,a.y);x.lineTo(b.x,b.y);x.lineTo(cc.x,cc.y);x.closePath();
      /* deckende Füllung: Flächen verdecken einander korrekt */
      x.globalAlpha=op*(.72+dep*.28);
      x.fillStyle=`hsl(${hue-8+t*22},${28+t*16}%,${4.5+L*13+t*4+dep*3}%)`;
      x.fill();
      /* Kantenlicht */
      const al=(.07+L*.26+dep*.14)*(.45+.55*t)*op;
      if(al>.02){
        x.globalAlpha=al>.52?.52:al;
        x.strokeStyle=`hsl(${hue+L*26+t*16},${54+t*18}%,${48+L*26}%)`;
        x.lineWidth=(.45+dep*.5)*(front?1:.55);
        x.stroke()}}
    x.globalAlpha=1}

  /* ════ 1-SIMPLIZES: nur wo keine Fläche liegt (Rand/Filamente) ════ */
  if(RC&&RC.edges&&(!surf||!RC.tris||RC.tris.length<4)){
    x.lineWidth=front?.85:.5;
    for(const [i,j] of RC.edges){
      const a=S[i],b=S[j]; if(!a||!b)continue;
      const dep=1-((a.z+b.z)*.5-zmin)*zr;
      const al=(.10+dep*.26)*op*(.4+.6*t);
      if(al<.018)continue;
      x.globalAlpha=al>.46?.46:al;
      x.strokeStyle=`hsl(${hue+22},${50+t*16}%,${46+dep*22}%)`;
      x.beginPath();x.moveTo(a.x,a.y);x.lineTo(b.x,b.y);x.stroke()}
    x.globalAlpha=1}

  /* ════ Sequenzpfad: die Token-Reihenfolge ════ */
  if(T>1&&front){
    x.globalAlpha=op*(.22+.3*t);
    x.strokeStyle=`hsl(${hue},66%,66%)`;
    x.lineWidth=1.35;
    x.beginPath();x.moveTo(S[0].x,S[0].y);
    for(let i=1;i<T;i++)x.lineTo(S[i].x,S[i].y);
    x.stroke();x.globalAlpha=1}

  /* ════ Knoten — Glow-Budget ════ */
  const ord2=S.map((s,i)=>i).sort((a,b)=>S[b].z-S[a].z);
  for(const i of ord2){
    const s=S[i], dep=1-(s.z-zmin)*zr, r=(1.1+s.f*2.3)*(front?1:.6);
    const al=(.26+t*.26+dep*.2)*op;
    x.globalAlpha=al>1?1:al;
    x.fillStyle=`hsl(${hue+i*1.2},${50+t*20}%,${56+dep*16}%)`;
    x.beginPath();x.arc(s.x,s.y,r,0,6.283);x.fill()}
  x.globalAlpha=1;

  if(front&&op>.9){
    const glow=ord2.slice(-6);                 // die 6 vordersten
    for(const i of glow){
      const s=S[i],rr=3.4+s.f*7.5;
      const g=x.createRadialGradient(s.x,s.y,0,s.x,s.y,rr);
      g.addColorStop(0,`hsla(${hue+44},92%,76%,${.18*op})`);
      g.addColorStop(1,`hsla(${hue+44},92%,66%,0)`);
      x.globalCompositeOperation='lighter';
      x.fillStyle=g;x.beginPath();x.arc(s.x,s.y,rr,0,6.283);x.fill();
      x.globalCompositeOperation='source-over'}
    x.globalAlpha=1}

  /* ════ Token-Labels ════ */
  if(front&&tokens&&op>.9){
    x.font='9px ui-monospace,monospace'; x.textAlign='center';
    for(let i=0;i<T;i++){
      const s=S[i]; if(s.f<.66)continue;
      x.globalAlpha=Math.min(.74,(s.f-.6)*2.1);
      x.fillStyle=`hsl(${hue+16},44%,78%)`;
      x.fillText(tokens[i]||'',s.x,s.y-8)}}
  x.globalAlpha=1}

/* ════════════════════════════════════════════════════════════════════
   5 · PIPELINE-EDITOR
   ════════════════════════════════════════════════════════════════════ */
const RED_OPTS=`__REDOPTS__`;
const stagesHost=document.getElementById('stages');
let STAGES=[{m:'umap',d:12},{m:'pca',d:3}];      // sinnvolle Vorgabe

function renderStages(){
  stagesHost.innerHTML='';
  STAGES.forEach((st,i)=>{
    const row=document.createElement('div');
    row.className='row3 stage';
    row.innerHTML=
      `<select data-k="m" data-i="${i}">${RED_OPTS}</select>`+
      `<input type="text" data-k="d" data-i="${i}" value="${st.d}">`+
      `<div class="ix">${i===STAGES.length-1?'→3D':'↓'}</div>`;
    stagesHost.appendChild(row);
    row.querySelector('[data-k="m"]').value=st.m;
  });
  stagesHost.querySelectorAll('select,input').forEach(el=>{
    el.onchange=el.oninput=()=>{
      const i=+el.dataset.i, k=el.dataset.k;
      if(k==='m')STAGES[i].m=el.value;
      else{const v=parseInt(el.value,10); STAGES[i].d=isNaN(v)?3:Math.max(1,Math.min(512,v))}
    }})}

document.getElementById('addStage').onclick=()=>{
  if(STAGES.length>=4)return;
  STAGES.splice(STAGES.length-1,0,{m:'pca',d:50}); renderStages()};
document.getElementById('delStage').onclick=()=>{
  if(STAGES.length<=1)return; STAGES.splice(STAGES.length-2,1); renderStages()};
renderStages();

/* ════════════════════════════════════════════════════════════════════
   6 · SLIDER
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
    const step=Math.max(1,Math.ceil(LAYERS/16));
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
   7 · TRANSPORT  —  Standard: STILLSTAND
   ════════════════════════════════════════════════════════════════════ */
const bPlay=document.getElementById('bPlay'),
      bView=document.getElementById('bView'),
      bSurf=document.getElementById('bSurf');
function sync(){
  bPlay.textContent=playing?'❚❚':'▶';
  bPlay.classList.toggle('on',playing)}
bPlay.onclick=()=>{if(!D)return;playing=!playing;targetPos=null;sync()};
document.getElementById('bPrev').onclick=()=>{playing=false;
  targetPos=Math.max(0,Math.round(pos)-1);sync()};
document.getElementById('bNext').onclick=()=>{playing=false;
  targetPos=Math.min(LAYERS-1,Math.round(pos)+1);sync()};
bSurf.onclick=()=>{surf=!surf;bSurf.classList.toggle('on',surf);
  bSurf.textContent=surf?'◆':'◇'};
bView.onclick=()=>{view=(view+1)%3;
  bView.textContent=['◐','◑','◓'][view];
  bView.classList.toggle('on',view!==2)};
addEventListener('keydown',e=>{
  if(e.target.tagName==='TEXTAREA'||e.target.tagName==='SELECT'
     ||e.target.tagName==='INPUT')return;
  if(e.code==='Space'){e.preventDefault();if(D){playing=!playing;targetPos=null;sync()}return}
  if(e.code==='ArrowUp'||e.code==='ArrowRight'){e.preventDefault();playing=false;
    targetPos=Math.min(LAYERS-1,Math.round(pos)+1);sync();return}
  if(e.code==='ArrowDown'||e.code==='ArrowLeft'){e.preventDefault();playing=false;
    targetPos=Math.max(0,Math.round(pos)-1);sync();return}
  if(e.code==='Home'){e.preventDefault();playing=false;targetPos=0;sync();return}
  if(e.code==='End'){e.preventDefault();playing=false;targetPos=LAYERS-1;sync();return}
  if(e.code==='KeyV'){bView.onclick()}
  if(e.code==='KeyS'){bSurf.onclick()}
  if(e.code==='Enter'&&(e.metaKey||e.ctrlKey))run()});

/* ════════════════════════════════════════════════════════════════════
   8 · SPARKLINES
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
  if(!isFinite(lo))return;
  if(hi-lo<1e-9)hi=lo+1;
  const pad=4,n=series[0].v.length;
  const X=i=>pad+(n>1?i/(n-1):0)*(w-2*pad);
  const Y=v=>h-pad-((v-lo)/(hi-lo))*(h-2*pad);
  if(markIdx!==undefined){
    g.strokeStyle='rgba(150,175,235,.2)';g.lineWidth=1;
    g.beginPath();g.moveTo(X(markIdx),0);g.lineTo(X(markIdx),h);g.stroke()}
  for(const s of series){
    /* sanfte Flächenfüllung unter der Kurve */
    g.beginPath();
    for(let i=0;i<s.v.length;i++){const px=X(i),py=Y(s.v[i]);i?g.lineTo(px,py):g.moveTo(px,py)}
    const grad=g.createLinearGradient(0,0,0,h);
    grad.addColorStop(0,s.c.replace('#','#')+'22');
    grad.addColorStop(1,'transparent');
    g.lineTo(X(s.v.length-1),h-pad);g.lineTo(X(0),h-pad);g.closePath();
    g.fillStyle=grad;g.globalAlpha=.5;g.fill();g.globalAlpha=1;
    g.beginPath();
    for(let i=0;i<s.v.length;i++){const px=X(i),py=Y(s.v[i]);i?g.lineTo(px,py):g.moveTo(px,py)}
    g.strokeStyle=s.c;g.lineWidth=s.w||1.4;g.lineJoin='round';g.stroke();
    if(markIdx!==undefined&&s.v[markIdx]!==undefined){
      g.fillStyle=s.c;g.beginPath();g.arc(X(markIdx),Y(s.v[markIdx]),2.2,0,6.283);g.fill()}}}

function sparkBars(cv,a,b){
  const [g,w,h]=sparkSetup(cv);
  const n=Math.max(a.length,b.length),pad=3,bw=(w-2*pad)/n;
  const mx=Math.max(...a,...b,1e-9);
  for(let i=0;i<n;i++){
    const ha=((a[i]||0)/mx)*(h-2*pad),hb=((b[i]||0)/mx)*(h-2*pad),px=pad+i*bw;
    g.fillStyle='rgba(111,211,247,.6)';g.fillRect(px+bw*.12,h-pad-ha,bw*.36,ha);
    g.fillStyle='rgba(247,127,176,.6)';g.fillRect(px+bw*.52,h-pad-hb,bw*.36,hb)}}

/* ════════════════════════════════════════════════════════════════════
   9 · HUD
   ════════════════════════════════════════════════════════════════════ */
const $=id=>document.getElementById(id);
const f3=v=>(v===undefined||v===null)?'—':(Math.abs(v)<10?v.toFixed(3):v.toFixed(1));
const i0=v=>(v===undefined||v===null)?'—':String(Math.round(v));
let lastHudLayer=-1;

function updateHUD(){
  if(!D)return;
  const L=Math.round(Math.max(0,Math.min(LAYERS-1,pos)));
  if(L===lastHudLayer)return;
  lastHudLayer=L;
  const sa=D.shapes_a[L],sb=D.shapes_b[L],ra=D.rips_a[L],rb=D.rips_b[L];
  $('s-emd').textContent=f3(D.emds[L]);
  $('s-drift').textContent=f3(D.drift[L]);
  $('s-epa').textContent=f3(ra.eps); $('s-epb').textContent=f3(rb.eps);
  $('s-tra').textContent=i0(ra.n_tris); $('s-trb').textContent=i0(rb.n_tris);
  $('s-eua').textContent=i0(ra.euler); $('s-eub').textContent=i0(rb.euler);
  $('s-ea').textContent=f3(sa.eff_dim); $('s-eb').textContent=f3(sb.eff_dim);
  $('s-aa').textContent=f3(sa.anisotropy); $('s-ab').textContent=f3(sb.anisotropy);
  $('s-ca').textContent=f3(sa.curvature); $('s-cb').textContent=f3(sb.curvature);
  $('s-na').textContent=f3(D.norms_a[L]); $('s-nb').textContent=f3(D.norms_b[L]);
  $('hs').textContent=(L===0?'EMBEDDING':'BLOCK '+String(L).padStart(2,'0'))
    +' · W₁ '+f3(D.emds[L]);
  $('hp').textContent=D.pipe_label||'';
  sparkLines($('sp1'),[{v:D.emds,c:'#f77fb0',w:1.6},
                       {v:D.drift,c:'#7ee7a8',w:1.2}],L);
  sparkLines($('sp2'),[{v:D.shapes_a.map(s=>s.eff_dim),c:'#6fd3f7',w:1.6},
                       {v:D.shapes_b.map(s=>s.eff_dim),c:'#f77fb0',w:1.6}],L);
  sparkBars($('sp3'),sa.spectrum,sb.spectrum);
}

/* ════════════════════════════════════════════════════════════════════
   10 · TOKEN-LEISTE
   ════════════════════════════════════════════════════════════════════ */
function buildTokens(){
  const mk=(arr,cls,host)=>{
    host.innerHTML='';
    arr.slice(0,40).forEach(t=>{
      const e=document.createElement('div');
      e.className='tk '+cls; e.textContent=t; host.appendChild(e)})};
  mk(D.tok_a,'a',$('trowA'));
  mk(D.tok_b,'b',$('trowB'))}

/* ════════════════════════════════════════════════════════════════════
   11 · HINTERGRUND  —  Vignette + Layer-Horizonte + Tiefen-Nebel
   ════════════════════════════════════════════════════════════════════ */
function background(){
  x.fillStyle='#05070d'; x.fillRect(0,0,W,H);
  const vg=x.createRadialGradient(W*.5,H*.5,0,W*.5,H*.5,Math.max(W,H)*.72);
  vg.addColorStop(0,'rgba(22,30,52,.5)');
  vg.addColorStop(.55,'rgba(11,15,27,.3)');
  vg.addColorStop(1,'rgba(3,4,9,0)');
  x.fillStyle=vg; x.fillRect(0,0,W,H);
  if(!D)return;

  const o={};
  for(let L=0;L<LAYERS;L++){
    const d=Math.abs(L-pos);
    if(d>3.6)continue;
    const near=Math.abs(L-Math.round(pos))<.01;
    const al=Math.max(0,(1-d/3.6))*(near?.17:.085);
    if(al<.006)continue;
    x.globalAlpha=al;
    x.strokeStyle=near?'hsl(196,46%,62%)':'hsl(218,32%,56%)';
    x.lineWidth=near?.85:.55;
    x.beginPath();
    for(let k=0;k<=52;k++){
      const a=k/52*6.2832;
      project([Math.cos(a)*1.22,0,Math.sin(a)*1.22],L,o);
      k?x.lineTo(o.x,o.y):x.moveTo(o.x,o.y)}
    x.closePath(); x.stroke()}
  x.globalAlpha=1}

/* ════════════════════════════════════════════════════════════════════
   12 · LOOP  —  ohne Befehl bleibt pos stehen
   ════════════════════════════════════════════════════════════════════ */
const hn=$('hn');
let last=performance.now();
function loop(now){
  const dt=Math.min(.05,(now-last)/1000); last=now;
  const ease=(cur,t_,sp)=>cur+(t_-cur)*(1-Math.exp(-sp*dt));

  yaw=ease(yaw,tg.y,6); pit=ease(pit,tg.p,6);
  breathe+=dt*.55;                       // nur fuer das minimale Atmen

  if(D){
    if(targetPos!==null){                 // ▸ ◂ Slider-Dot: sanftes Stepping
      pos=ease(pos,targetPos,STEP_SPD);
      if(Math.abs(pos-targetPos)<.004){pos=targetPos;targetPos=null}
    }else if(playing&&scrub===null){      // ▶ explizit gedrueckt
      pos+=AUTO_SPD*dt;
      if(pos>LAYERS-1){pos=LAYERS-1;playing=false;sync()}
    }
    /* sonst: pos bleibt exakt stehen — die Flaeche ruht */
    camY=ease(camY,-pos*RISE,3.6);
  }

  background();

  if(D){
    for(let g=GH;g>=1;g--){
      const lf=pos-g*GAP; if(lf<-.05)continue;
      const op=Math.pow(.3,g)*.72;
      if(view!==1)drawManifold('proj_a','rips_a',Math.max(0,lf),HUE_A-g*4,op,false,null);
      if(view!==0)drawManifold('proj_b','rips_b',Math.max(0,lf),HUE_B-g*4,op,false,null)}
    if(view!==1)drawManifold('proj_a','rips_a',pos,HUE_A,1,true,D.tok_a);
    if(view!==0)drawManifold('proj_b','rips_b',pos,HUE_B,1,true,D.tok_b);

    hn.textContent=String(Math.round(pos)).padStart(2,'0');
    updateRail(); updateHUD()}
  requestAnimationFrame(loop)}

/* ════════════════════════════════════════════════════════════════════
   13 · SERVER-CALL
   ════════════════════════════════════════════════════════════════════ */
const msg=$('msg'), go=$('go'), boot=$('boot'), bootmsg=$('bootmsg');
let busy=false;

async function run(){
  if(busy)return; busy=true;
  go.classList.add('busy'); go.textContent='RECHNET …';
  msg.className=''; msg.textContent='Hidden States · Rips-Komplex · Transport …';
  boot.classList.remove('gone');
  bootmsg.textContent='FORWARD-PASS · VIETORIS-RIPS · PIPELINE';

  const body={
    text_a:$('ta').value, text_b:$('tb').value,
    model:$('model').value, emd:$('emd').value,
    density:parseFloat($('density').value),
    pipeline:STAGES.map(s=>({method:s.m,dims:s.d}))};

  try{
    const r=await fetch('/api/analyse',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j=await r.json();
    if(!r.ok||j.error)throw new Error(j.error||('HTTP '+r.status));

    D=j; LAYERS=D.n_layers;
    RISE = LAYERS>20 ? 118 : LAYERS>12 ? 148 : 182;
    pos=0; camY=0; targetPos=null; playing=false; lastHudLayer=-1;
    buildRail(); buildTokens(); sync();
    const tri=D.rips_a[0]?D.rips_a[0].n_tris:0;
    msg.textContent=`${D.model.split('/').pop()} · ${LAYERS} Layer · `
      +`${D.tok_a.length}/${D.tok_b.length} Tokens · ${tri} △ · ${D.device.toUpperCase()}`;
    boot.classList.add('gone');
  }catch(e){
    msg.className='err'; msg.textContent='FEHLER: '+e.message;
    bootmsg.textContent='FEHLER · '+e.message;
    setTimeout(()=>boot.classList.add('gone'),900);
  }finally{
    busy=false; go.classList.remove('busy'); go.textContent='ANALYSIEREN'}}
go.onclick=run;

sync();
requestAnimationFrame(loop);
setTimeout(()=>boot.classList.add('gone'),600);
</script>
</body>
</html>
""".replace("__OPTS__", opts).replace("__REDOPTS__", red_opts)

# ============================================================================
# 10) HTTP-SERVER
# ============================================================================

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

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
            emd = req.get("emd") or "exact"
            density = float(req.get("density") or 1.0)

            pipeline = req.get("pipeline") or [{"method": "pca", "dims": 3}]
            clean = []
            for st in pipeline[:4]:
                m = str(st.get("method", "pca")).lower()
                if m not in REDUCERS:
                    m = "pca"
                d = int(st.get("dims", 3))
                clean.append({"method": m, "dims": max(1, min(512, d))})

            lbl = " -> ".join(f'{s["method"]}:{s["dims"]}' for s in clean)
            print(f"[api] {model} | {lbl} | emd={emd} | rho={density}")

            data = analyse(ta, tb, model, clean, emd, density=density)
            self._send(200, json.dumps(data).encode("utf-8"),
                       "application/json; charset=utf-8")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send(500, json.dumps({"error": str(e)}).encode("utf-8"),
                       "application/json")

# ============================================================================
# 11) ENTRY POINT
# ============================================================================

def main():
    port = int(os.environ.get("MANIFOLD_PORT", "8051"))
    url = f"http://127.0.0.1:{port}"
    print("=" * 74)
    print("  HIDDEN-STATE MANIFOLD EXPLORER  ·  v3  ·  Vietoris-Rips")
    print("=" * 74)
    print(f"  Device    : {DEVICE.upper()}")
    print(f"  CUDA      : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU       : {torch.cuda.get_device_name(0)}")
    cpu_ok = sum(1 for v in MODELS.values() if v["tier"].startswith("CPU"))
    print(f"  Modelle   : {len(MODELS)}  ({cpu_ok} CPU-tauglich)")
    print(f"  Reducer   : {', '.join(k for k in REDUCERS if k != 'none')}")
    print(f"  URL       : {url}")
    print("=" * 74)
    print("  Bedienung : ▶ laeuft durch die Layer · ◂ ▸ einzeln · ◆ Flaeche an/aus")
    print("              Ohne Befehl steht die Mannigfaltigkeit still.")
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
