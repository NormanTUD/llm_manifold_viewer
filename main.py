# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch", "transformers", "numpy", "scikit-learn",
#   "pot", "umap-learn", "accelerate",
# ]
# ///
"""
Hidden-State Manifold Explorer  ·  v4
=====================================
n Saetze (1..8) · echte Vietoris-Rips-Flaeche · freie Reduktions-Pipeline
Per-Satz-Sichtbarkeit · Solo-Modus · kontraststarkes Dark-UI

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
import math
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

MAX_SERIES = 8          # harte Obergrenze fuer die Anzahl Saetze

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
# 6) REDUKTIONS-PIPELINE
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

    cap = REDUCERS.get(method, {}).get("max_out")
    if cap:
        dims = min(dims, cap)
    dims = max(1, min(dims, d))
    if n > 1:
        dims = min(dims, n - 1) if method in ("pca", "tsne", "isomap") else dims
    dims = max(1, dims)

    if dims >= d and method in ("pca", "svd", "rand"):
        return X

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

    except Exception as e:
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

    if Z.shape[1] > 3:
        Z = PCA(n_components=3, random_state=seed).fit_transform(Z)
    elif Z.shape[1] < 3:
        Z = np.hstack([Z, np.zeros((Z.shape[0], 3 - Z.shape[1]))])
    return Z

def joint_pipeline(blocks, pipeline, seed=0):
    """Gemeinsame Reduktion ALLER Saetze -> ein Koordinatensystem."""
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
# 7) VIETORIS-RIPS
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
    """Liefert (edges, triangles, eps, stats) des Vietoris-Rips-Komplexes."""
    n = X.shape[0]
    if n < 3:
        return [], [], 0.0, dict(n_edges=0, n_tris=0, eps=0.0, euler=n)

    D = _pairwise(np.asarray(X, dtype=np.float64))
    eps = rips_epsilon(D, k=k, density=density)

    adj = (D <= eps) & ~np.eye(n, dtype=bool)

    ei, ej = np.where(np.triu(adj, 1))
    edges = np.stack([ei, ej], 1)

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

    if len(tris) > max_tris:
        diam = np.array([max(D[a, b], D[b, c], D[a, c]) for a, b, c in tris])
        keep = np.argsort(diam)[:max_tris]
        tris = [tris[q] for q in keep]

    V, E, F = n, len(edges), len(tris)
    return (edges.tolist(), [list(t) for t in tris], float(eps),
            dict(n_edges=E, n_tris=F, eps=float(eps), euler=int(V - E + F)))

# ============================================================================
# 8) Farb-Zuweisung
# ============================================================================

GOLDEN = 137.50776405003785

def auto_hue(i: int, base: float = 193.0) -> float:
    """Goldener Winkel -> maximal getrennte Farbtoene fuer beliebig viele Reihen."""
    return (base + i * GOLDEN) % 360.0

# ============================================================================
# 9) Analyse-Pipeline  —  n Saetze
# ============================================================================

def analyse(texts, model_name, pipeline, emd_mode,
            density=1.0, rips_k=4, hues=None, labels=None):
    """
    texts : Liste von Strings (1 .. MAX_SERIES)
    Gibt eine 'series'-Liste zurueck; paarweise W1-Distanzen als
    obere Dreiecksliste pro Layer.
    """
    texts = [t for t in (s.strip() for s in texts) if t]
    if not texts:
        raise ValueError("Kein nicht-leerer Satz uebergeben.")
    if len(texts) > MAX_SERIES:
        raise ValueError(f"Maximal {MAX_SERIES} Saetze (bekommen: {len(texts)}).")

    n_series = len(texts)

    toks, HS = [], []
    for t in texts:
        tk, h = hidden_states(t, model_name)
        toks.append(tk)
        HS.append(h)

    n_layers = min(h.shape[0] for h in HS)

    proj = [[] for _ in range(n_series)]
    rips = [[] for _ in range(n_series)]
    shapes = [[] for _ in range(n_series)]
    prev = [None] * n_series
    emd_layers = []

    for L in range(n_layers):
        blocks = [HS[s][L] for s in range(n_series)]

        # --- Vietoris-Rips im Originalraum ---
        for s in range(n_series):
            e, tr, eps, st = rips_complex(blocks[s], density=density, k=rips_k)
            rips[s].append(dict(edges=e, tris=tr, **st))

        # --- gemeinsame Reduktion ueber ALLE Saetze ---
        P = joint_pipeline(blocks, pipeline, seed=0)
        for s in range(n_series):
            p = procrustes_align(P[s], prev[s])
            prev[s] = p
            proj[s].append(p)

        # --- paarweise W1, obere Dreiecksmatrix ---
        pairs = []
        for i in range(n_series):
            for j in range(i + 1, n_series):
                pairs.append(dict(i=i, j=j,
                                  w=wasserstein(blocks[i], blocks[j], mode=emd_mode)))
        emd_layers.append(pairs)

        for s in range(n_series):
            shapes[s].append(shape_metrics(blocks[s]))

    # --- gemeinsame Normierung ueber ALLE Layer und Saetze ---
    allp = np.vstack([p for s in proj for p in s])
    centre = allp.mean(0)
    scale = float(np.abs(allp - centre).max()) + 1e-9
    proj = [[((p - centre) / scale).round(4).tolist() for p in s] for s in proj]

    series = []
    for s in range(n_series    series = []
    for s in range(n_series):
        drift_s = [0.0] + [wasserstein(HS[s][L - 1], HS[s][L], mode=emd_mode)
                           for L in range(1, n_layers)]
        norms_s = [float(np.linalg.norm(HS[s][L], axis=1).mean())
                   for L in range(n_layers)]
        hue = (hues[s] if hues and s < len(hues) and hues[s] is not None
               else auto_hue(s))
        label = (labels[s] if labels and s < len(labels) and labels[s]
                 else f"S{s + 1}")
        series.append(dict(
            id=s,
            label=label,
            text=texts[s],
            hue=float(hue) % 360.0,
            tokens=toks[s],
            proj=proj[s],
            rips=rips[s],
            shapes=shapes[s],
            norms=norms_s,
            drift=drift_s,
        ))

    pipe_label = " → ".join(
        f'{REDUCERS.get(st["method"], {}).get("label", st["method"])}·{st["dims"]}D'
        for st in pipeline if st.get("method") != "none") or "PCA·3D"

    return dict(
        series=series,
        emd=emd_layers,
        n_series=n_series,
        n_layers=n_layers,
        model=model_name,
        pipeline=pipeline,
        pipe_label=pipe_label,
        density=density,
        rips_k=rips_k,
        emd_mode=emd_mode,
        device=DEVICE,
        max_series=MAX_SERIES,
    )

# ============================================================================
# 10) FRONTEND
# ============================================================================

DEFAULT_TEXTS = [
    "The cat sat quietly on the warm windowsill.",
    "Non-abelian cohomology classifies torsors over a site.",
]

def build_html() -> str:
    opts = "".join(
        f'<option value="{k}">[{v["tier"]}] {k} · {v["size"]} · {v["note"]}</option>'
        for k, v in MODELS.items())

    red_opts = "".join(
        f'<option value="{k}">{v["label"]}</option>' for k, v in REDUCERS.items())

    defaults = json.dumps(DEFAULT_TEXTS)

    return r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>hidden-state manifold · v4</title>
<style>
  /* ═══════════════════════════════════════════════════════════════
     FARB-SYSTEM  —  jeder Textwert erfuellt mind. 4.5:1 auf #05070d,
     strukturelle Labels mind. 7:1. Keine opacity auf Text.
     ═══════════════════════════════════════════════════════════════ */
  :root{
    --bg:#05070d;
    --surface:rgba(17,22,36,.70);
    --surface2:rgba(10,13,22,.58);
    --line:rgba(130,152,208,.15);
    --line-hi:rgba(165,190,245,.32);

    --txt:#c9d2e8;        /* Fliesstext        ~11.5:1 */
    --txt-dim:#9aa5c4;    /* sekundaer         ~7.4:1  */
    --txt-label:#aab6d6;  /* Label, ohne opacity ~8.9:1 */
    --txt-faint:#7d88a8;  /* Hinweise          ~5.1:1  */
    --txt-strong:#e9eefb; /* Zahlen/Werte     ~15.2:1 */
    --err:#ff8fa3;
    --ok:#8df0b4;
  }
  :root[data-hc="1"]{
    --txt:#e8eefc; --txt-dim:#c4cee8; --txt-label:#d4ddf4;
    --txt-faint:#aeb9d6; --txt-strong:#ffffff;
    --line:rgba(150,175,235,.30); --line-hi:rgba(190,212,255,.55);
    --surface:rgba(20,26,42,.86); --surface2:rgba(13,17,28,.80);
  }

  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg);overflow:hidden;
    font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--txt-dim)}
  canvas{display:block;position:fixed;inset:0}
  ::-webkit-scrollbar{width:6px}
  ::-webkit-scrollbar-track{background:transparent}
  ::-webkit-scrollbar-thumb{background:rgba(140,162,215,.28);border-radius:3px}
  ::-webkit-scrollbar-thumb:hover{background:rgba(175,200,255,.46)}
  :focus-visible{outline:2px solid var(--a);outline-offset:2px}

  /* ── Konsole links ───────────────────── */
  #ui{position:fixed;left:22px;top:22px;z-index:14;width:336px;
    max-height:calc(100vh - 44px);overflow-y:auto;padding-right:7px}
  .card{background:linear-gradient(165deg,var(--surface),var(--surface2));
    border:1px solid var(--line);border-radius:12px;
    padding:14px 15px;margin-bottom:9px;backdrop-filter:blur(18px) saturate(1.25);
    box-shadow:0 8px 32px -16px rgba(0,0,0,.92);
    transition:border-color .35s cubic-bezier(.22,1,.36,1)}
  .card:hover{border-color:var(--line-hi)}
  .card.accent{border-color:rgba(111,211,247,.26)}

  .lb{display:block;font-size:9px;letter-spacing:.26em;
    margin-bottom:8px;color:var(--txt-label);text-transform:uppercase;
    font-weight:500}
  .lb em{font-style:normal;float:right;letter-spacing:.1em;
    color:var(--a);text-transform:none}

  textarea,select,input[type=text],input[type=number]{width:100%;
    background:rgba(4,6,12,.80);
    color:var(--txt);border:1px solid var(--line);border-radius:7px;
    padding:8px 10px;font:11.5px/1.5 ui-monospace,monospace;outline:none;
    resize:none;transition:border-color .25s,box-shadow .25s}
  textarea:focus,select:focus,input:focus{border-color:var(--a);
    color:var(--txt-strong);box-shadow:0 0 0 3px rgba(111,211,247,.11)}
  textarea::placeholder,input::placeholder{color:#6b7694}
  textarea{height:46px}
  select{height:30px;cursor:pointer;appearance:none;
    background-image:linear-gradient(45deg,transparent 50%,#9aa5c4 50%),
      linear-gradient(135deg,#9aa5c4 50%,transparent 50%);
    background-position:calc(100% - 14px) 14px,calc(100% - 9px) 14px;
    background-size:5px 5px;background-repeat:no-repeat}
  select option{background:#0a0d16;color:var(--txt)}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:8px}

  /* ── Satz-Reihen ─────────────────────── */
  .ser{position:relative;margin-bottom:7px;padding:9px 10px 10px 13px;
    border:1px solid var(--line);border-radius:9px;
    background:rgba(8,11,19,.5);transition:border-color .3s,background .3s}
  .ser:before{content:'';position:absolute;left:0;top:9px;bottom:9px;width:3px;
    border-radius:0 2px 2px 0;background:var(--sc)}
  .ser.off{opacity:.62}
  .ser.off:before{background:#434b66}
  .ser:focus-within{border-color:var(--line-hi);background:rgba(11,15,26,.72)}
  .serhead{display:flex;align-items:center;gap:6px;margin-bottom:6px}
  .serhead .nm{flex:1;background:transparent;border:0;padding:2px 0;
    font:9.5px/1.3 ui-monospace,monospace;letter-spacing:.2em;
    color:var(--sc);text-transform:uppercase;outline:none;min-width:0}
  .serhead .nm:focus{color:var(--txt-strong)}
  .sb{width:23px;height:23px;flex:none;border-radius:6px;cursor:pointer;
    background:rgba(16,21,34,.75);border:1px solid var(--line);
    color:var(--txt-dim);font-size:10px;line-height:1;display:flex;
    align-items:center;justify-content:center;transition:.22s;user-select:none}
  .sb:hover{border-color:var(--line-hi);color:var(--txt-strong)}
  .sb.act{border-color:var(--sc);color:var(--txt-strong);
    box-shadow:0 0 14px -5px var(--sc)}
  .sb.del:hover{border-color:var(--err);color:var(--err)}
  .sb input[type=color]{position:absolute;width:0;height:0;opacity:0;
    pointer-events:none}
  .ser textarea{height:42px;font-size:11px}
  #serAdd{margin-top:2px}

  /* ── Pipeline-Stufen ─────────────────── */
  .row3{display:grid;grid-template-columns:1fr 56px 28px;gap:6px;
    align-items:center;margin-bottom:6px}
  .row3 .ix{font-size:8.5px;letter-spacing:.16em;color:var(--txt-faint);
    text-align:center}
  #stages{margin-bottom:4px}
  .stage{position:relative;padding-left:14px}
  .stage:before{content:'';position:absolute;left:4px;top:-4px;bottom:50%;
    width:1px;background:var(--line)}
  .stage:after{content:'';position:absolute;left:4px;top:50%;width:7px;
    height:1px;background:var(--line)}
  .stage:first-child:before{top:50%}
  .stage:last-child:before{bottom:50%}
  .mini{width:100%;padding:7px;border-radius:6px;cursor:pointer;
    background:rgba(16,21,34,.7);border:1px dashed var(--line);
    color:var(--txt-dim);font:9.5px/1 ui-monospace,monospace;letter-spacing:.18em;
    transition:.22s;text-align:center;user-select:none}
  .mini:hover{border-color:var(--a);color:var(--txt-strong);border-style:solid}
  .mini.rm:hover{border-color:var(--err);color:var(--err)}
  .mini.dis{opacity:.4;pointer-events:none}
  #pipeHint{font-size:9px;letter-spacing:.08em;color:var(--txt-faint);
    margin-top:8px;line-height:1.65}

  #go{width:100%;margin-top:4px;padding:12px;border-radius:9px;cursor:pointer;
    background:linear-gradient(145deg,rgba(28,36,56,.95),rgba(17,22,36,.95));
    border:1px solid var(--a);color:var(--txt-strong);
    font:10.5px/1 ui-monospace,monospace;letter-spacing:.28em;text-align:center;
    box-shadow:0 0 26px -10px var(--a);transition:.28s;user-select:none}
  #go:hover{background:var(--a);color:#05070d;letter-spacing:.34em;
    box-shadow:0 0 34px -6px var(--a)}
  #go.busy{opacity:.55;pointer-events:none;letter-spacing:.18em}
  #msg{font-size:9.5px;letter-spacing:.06em;margin-top:9px;min-height:13px;
    color:var(--txt-faint);line-height:1.7}
  #msg.err{color:var(--err)}
  #msg.ok{color:var(--ok)}

  /* ── Kennzahlen ──────────────────────── */
  .stat{display:flex;justify-content:space-between;align-items:baseline;
    gap:10px;padding:5px 0;border-bottom:1px solid rgba(130,152,208,.1)}
  .stat:last-child{border:0;padding-bottom:0}
  .stat>span{font-size:9px;letter-spacing:.16em;color:var(--txt-label);
    flex:none}
  .stat>b{font-weight:400;font-size:12px;color:var(--txt-strong);
    font-variant-numeric:tabular-nums;letter-spacing:.02em;text-align:right;
    overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .stat>b em{font-style:normal}
  .sep{color:var(--txt-faint);padding:0 3px}
  #emdList{margin-top:2px}
  .pairrow{display:flex;justify-content:space-between;gap:8px;padding:4px 0;
    font-size:10px;border-bottom:1px solid rgba(130,152,208,.08)}
  .pairrow:last-child{border:0}
  .pairrow span{letter-spacing:.1em}
  .pairrow b{font-weight:400;color:var(--txt-strong);
    font-variant-numeric:tabular-nums}

  /* ── Header ──────────────────────────── */
  #h{position:fixed;right:110px;top:24px;text-align:right;z-index:10;
    font-size:9.5px;letter-spacing:.28em;color:var(--txt-faint);
    pointer-events:none}
  #h b{display:block;font-size:46px;letter-spacing:.02em;color:var(--txt-strong);
    font-weight:200;margin:-4px 0 0;font-variant-numeric:tabular-nums;
    text-shadow:0 0 40px rgba(111,211,247,.2)}
  #h i{font-style:normal;font-size:9px;letter-spacing:.2em;
    color:var(--txt-dim);display:block;margin-top:2px}
  #h u{text-decoration:none;font-size:9px;letter-spacing:.16em;
    color:var(--txt-faint);display:block;margin-top:5px}

  /* ── Rail ────────────────────────────── */
  #rail{position:fixed;right:44px;top:50%;transform:translateY(-50%);
    width:46px;height:60vh;z-index:12;cursor:pointer}
  #line{position:absolute;left:23px;top:0;bottom:0;width:1px;
    background:linear-gradient(180deg,rgba(255,255,255,.04),
      rgba(255,255,255,.2),rgba(255,255,255,.04))}
  #prog{position:absolute;left:22px;bottom:0;width:3px;border-radius:2px;
    background:linear-gradient(180deg,var(--a),rgba(111,211,247,0));opacity:.8}
  .dot{position:absolute;left:19px;width:9px;height:9px;margin-top:-4.5px;
    border-radius:50%;background:#0c0f1a;border:1px solid rgba(165,188,240,.4);
    transition:.3s cubic-bezier(.22,1,.36,1)}
  .dot.pass{border-color:rgba(180,205,255,.66);background:#1b2434}
  .dot.act{background:var(--a);border-color:var(--a);
    box-shadow:0 0 16px var(--a);transform:scale(1.55)}
  .dot label{position:absolute;right:17px;top:-5px;font-size:9px;
    letter-spacing:.14em;color:var(--txt-faint);white-space:nowrap;
    pointer-events:none;transition:color .28s}
  .dot.act label{color:var(--txt-strong);font-weight:500}
  #knob{position:absolute;left:14px;width:19px;height:19px;margin-top:-9.5px;
    border-radius:50%;border:1px solid var(--a);background:rgba(8,11,20,.9);
    box-shadow:0 0 24px -3px var(--a);pointer-events:none}
  #knob:after{content:'';position:absolute;inset:5px;border-radius:50%;
    background:var(--a)}

  /* ── Transport ───────────────────────── */
  #tp{position:fixed;right:28px;bottom:24px;display:flex;gap:6px;z-index:12}
  .btn{min-width:32px;height:32px;padding:0 8px;border-radius:8px;cursor:pointer;
    background:rgba(16,21,34,.76);border:1px solid var(--line);
    color:var(--txt-dim);font-size:11.5px;display:flex;align-items:center;
    justify-content:center;backdrop-filter:blur(12px);transition:.22s;
    user-select:none}
  .btn:hover{border-color:var(--line-hi);color:var(--txt-strong);
    transform:translateY(-1px)}
  .btn.on{border-color:var(--a);color:var(--txt-strong);
    box-shadow:0 0 18px -5px var(--a)}

  /* ── Legende unten links ─────────────── */
  #leg{position:fixed;left:22px;bottom:44px;z-index:11;display:flex;
    flex-direction:column;gap:4px;max-width:300px}
  .lgi{display:flex;align-items:center;gap:7px;font-size:9.5px;
    letter-spacing:.12em;color:var(--txt-dim);cursor:pointer;
    padding:3px 7px;border-radius:5px;background:rgba(10,13,22,.6);
    border:1px solid transparent;backdrop-filter:blur(8px);transition:.2s;
    user-select:none}
  .lgi:hover{border-color:var(--line-hi);color:var(--txt-strong)}
  .lgi.off{color:var(--txt-faint)}
  .lgi u{text-decoration:none;width:9px;height:9px;border-radius:2px;
    flex:none;box-shadow:0 0 8px -1px currentColor}
  .lgi.off u{background:#444b66!important;box-shadow:none}
  .lgi s{text-decoration:none;font-size:8.5px;color:var(--txt-faint);
    letter-spacing:.1em}

  /* ── Tokens ──────────────────────────── */
  #toks{position:fixed;left:50%;bottom:18px;transform:translateX(-50%);
    z-index:11;display:flex;flex-direction:column;gap:4px;align-items:center;
    pointer-events:none;max-width:52vw}
  .trow{display:flex;gap:3px;flex-wrap:wrap;justify-content:center}
  .tk{padding:2.5px 6px;border-radius:4px;font-size:9.5px;letter-spacing:.04em;
    background:rgba(10,14,24,.78);border:1px solid;backdrop-filter:blur(8px)}

  #ft{position:fixed;left:22px;bottom:18px;font-size:9px;letter-spacing:.2em;
    color:var(--txt-faint);z-index:10;pointer-events:none}

  .spark{width:100%;height:46px;display:block;margin-top:5px}
  .legend{display:flex;gap:10px;margin-top:6px;font-size:9px;
    letter-spacing:.12em;flex-wrap:wrap}
  .legend i{font-style:normal;display:flex;align-items:center;gap:5px;
    color:var(--txt-dim)}
  .legend u{text-decoration:none;width:9px;height:2.5px;border-radius:1px;
    display:inline-block}

  #boot{position:fixed;inset:0;z-index:40;background:var(--bg);
    display:flex;align-items:center;justify-content:center;flex-direction:column;
    gap:16px;transition:opacity .75s;font-size:10px;letter-spacing:.32em;
    color:var(--txt-dim)}
  #boot.gone{opacity:0;pointer-events:none}
  #bootbar{width:210px;height:1px;background:rgba(130,152,208,.18);
    overflow:hidden}
  #bootbar i{display:block;height:100%;width:32%;background:#6fd3f7;
    box-shadow:0 0 12px #6fd3f7;
    animation:sweep 1.3s cubic-bezier(.65,0,.35,1) infinite}
  @keyframes sweep{0%{transform:translateX(-110%)}100%{transform:translateX(340%)}}
  #bootmsg{color:var(--txt-faint)}
</style>
</head>
<body>
<canvas id="c"></canvas>

<div id="ui">
  <div class="card">
    <span class="lb">Sätze <em id="serCount">0 / 8</em></span>
    <div id="serList"></div>
    <div class="mini" id="serAdd">+ SATZ HINZUFÜGEN</div>
  </div>

  <div class="card accent">
    <span class="lb">Modell</span>
    <select id="model">__OPTS__</select>

    <div style="height:13px"></div>
    <span class="lb">Reduktions-Pipeline</span>
    <div id="stages"></div>
    <div class="row" style="margin-top:4px">
      <div class="mini" id="addStage">+ STUFE</div>
      <div class="mini rm" id="delStage">− STUFE</div>
    </div>
    <div id="pipeHint">Kette wird von oben nach unten angewandt.<br>
      Alle Sätze teilen ein Koordinatensystem.</div>

    <div style="height:13px"></div>
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

    <div style="height:12px"></div>
    <div id="go">ANALYSIEREN</div>
    <div id="msg"></div>
  </div>

  <div class="card">
    <span class="lb">Layer-Kennzahlen <em id="focLabel">—</em></span>
    <div class="stat"><span>DRIFT L-1→L</span><b id="s-drift">—</b></div>
    <div class="stat"><span>RIPS ε</span><b id="s-eps">—</b></div>
    <div class="stat"><span>2-SIMPLIZES</span><b id="s-tris">—</b></div>
    <div class="stat"><span>EULER χ</span><b id="s-euler">—</b></div>
    <div class="stat"><span>EFF. DIM</span><b id="s-edim">—</b></div>
    <div class="stat"><span>ANISOTROPIE</span><b id="s-aniso">—</b></div>
    <div class="stat"><span>KRÜMMUNG</span><b id="s-curv">—</b></div>
    <div class="stat"><span>‖h‖ MITTEL</span>
<b id="s-norm">—</b></div>
  </div>

  <div class="card">
    <span class="lb">Paarweise W₁ <em id="emdCount">—</em></span>
    <div id="emdList"></div>
  </div>

  <div class="card">
    <span class="lb">Verlauf über Layer</span>
    <canvas id="sp1" class="spark"></canvas>
    <div class="legend" id="leg1"></div>
    <canvas id="sp2" class="spark" style="margin-top:10px"></canvas>
    <div class="legend" id="leg2"></div>
  </div>

  <div class="card">
    <span class="lb">Spektrum @ Layer</span>
    <canvas id="sp3" class="spark" style="height:60px"></canvas>
    <div class="legend" id="leg3"></div>
  </div>
</div>

<div id="h"><span>LAYER</span><b id="hn">00</b><i id="hs">—</i><u id="hp">—</u></div>

<div id="rail"><div id="line"></div><div id="prog"></div><div id="knob"></div></div>

<div id="tp">
  <div class="btn" id="bPlay">▶</div>
  <div class="btn" id="bPrev">◂</div>
  <div class="btn" id="bNext">▸</div>
  <div class="btn on" id="bSurf">◆</div>
  <div class="btn" id="bAll">◉</div>
  <div class="btn" id="bHC">A</div>
</div>

<div id="leg"></div>
<div id="toks"></div>

<div id="ft">DRAG ROTATE · SCROLL ZOOM · ◂ ▸ LAYER · 1–8 SATZ · ⇧+N SOLO</div>

<div id="boot"><div>HIDDEN-STATE MANIFOLD · V4</div>
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

const MAXS=8, GOLDEN=137.50776405003785;
const autoHue=i=>(193+i*GOLDEN)%360;
document.documentElement.style.setProperty('--a','hsl(193,76%,66%)');

/* ════════════════════════════════════════════════════════════════════
   1 · STATE
   ════════════════════════════════════════════════════════════════════ */
let D=null;
let LAYERS=1, RISE=168, GH=2, GAP=0.62;
let pos=0, playing=false, scrub=null, targetPos=null, surf=true;
let VIS=[];           // Sichtbarkeit pro Reihe
let SOLO=-1;          // -1 = kein Solo
let FOCUS=0;          // Reihe fuer das Kennzahlen-Panel
const STEP_SPD=5.2, AUTO_SPD=0.34;
let yaw=-.68,pit=.26,dist=1180,camY=0,tg={y:-.68,p:.26},dg=null;
let breathe=0;

const visible=i=>SOLO>=0 ? i===SOLO : !!VIS[i];

/* Eingabe-Reihen vor der Analyse */
let ROWS=__DEFAULTS__.map((t,i)=>({text:t,label:'S'+(i+1),hue:autoHue(i),on:true}));

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

function cloudAt(arr,lf){
  const n=arr.length;
  const i0=Math.max(0,Math.min(n-1,Math.floor(lf)));
  const i1=Math.max(0,Math.min(n-1,i0+1));
  const k=lf-i0, ke=k*k*(3-2*k);
  const A=arr[i0],B=arr[i1],T=A.length,out=new Array(T);
  const br=Math.sin(breathe)*0.006;
  for(let t=0;t<T;t++){
    const a=A[t],b=B[t],s=1+br;
    out[t]=[(a[0]+(b[0]-a[0])*ke)*s,
            (a[1]+(b[1]-a[1])*ke)*s,
            ((a[2]||0)+((b[2]||0)-(a[2]||0))*ke)*s]}
  return out}

function ripsAt(arr,lf){
  return arr[Math.max(0,Math.min(arr.length-1,Math.round(lf)))]}

/* ════════════════════════════════════════════════════════════════════
   4 · FLÄCHEN-RENDERER
   ════════════════════════════════════════════════════════════════════ */
function drawManifold(sr,lf,hueShift,op,front,showTok){
  const P=cloudAt(sr.proj,lf), T=P.length;
  const RC=ripsAt(sr.rips,lf);
  const hue=sr.hue+hueShift;
  const t=LAYERS>1?lf/(LAYERS-1):0;

  const S=new Array(T);
  let zmin=1e9,zmax=-1e9;
  for(let i=0;i<T;i++){
    const o={};project(P[i],lf,o);S[i]=o;
    if(o.z<zmin)zmin=o.z; if(o.z>zmax)zmax=o.z}
  const zr=1/Math.max(1,zmax-zmin);

  /* ── 2-Simplizes: die Fläche ── */
  if(surf&&RC&&RC.tris&&RC.tris.length){
    const tris=RC.tris, M=tris.length;
    const key=new Float32Array(M), idx=new Int32Array(M), lit=new Float32Array(M);
    for(let m=0;m<M;m++){
      const T3=tris[m],a=S[T3[0]],b=S[T3[1]],cc=S[T3[2]];
      if(!a||!b||!cc){key[m]=1e9;idx[m]=m;continue}
      key[m]=(a.z+b.z+cc.z)/3; idx[m]=m;
      const nz=(b.x-a.x)*(cc.y-a.y)-(b.y-a.y)*(cc.x-a.x);
      let L=nz/5200+.5; lit[m]=L<0?0:L>1?1:L}
    const ord=Array.prototype.slice.call(idx).sort((m,n)=>key[n]-key[m]);
    for(const m of ord){
      const T3=tris[m],a=S[T3[0]],b=S[T3[1]],cc=S[T3[2]];
      if(!a||!b||!cc)continue;
      const dep=1-(key[m]-zmin)*zr, L=lit[m];
      x.beginPath();
      x.moveTo(a.x,a.y);x.lineTo(b.x,b.y);x.lineTo(cc.x,cc.y);x.closePath();
      x.globalAlpha=op*(.72+dep*.28);
      x.fillStyle=`hsl(${hue-8+t*22},${28+t*16}%,${5+L*14+t*4+dep*3}%)`;
      x.fill();
      const al=(.09+L*.30+dep*.16)*(.5+.5*t)*op;
      if(al>.02){
        x.globalAlpha=al>.58?.58:al;
        x.strokeStyle=`hsl(${hue+L*26+t*16},${56+t*18}%,${52+L*26}%)`;
        x.lineWidth=(.45+dep*.5)*(front?1:.55);
        x.stroke()}}
    x.globalAlpha=1}

  /* ── 1-Simplizes, wenn keine Fläche da ist ── */
  if(RC&&RC.edges&&(!surf||!RC.tris||RC.tris.length<4)){
    x.lineWidth=front?.85:.5;
    for(const e of RC.edges){
      const a=S[e[0]],b=S[e[1]]; if(!a||!b)continue;
      const dep=1-((a.z+b.z)*.5-zmin)*zr;
      const al=(.12+dep*.28)*op*(.45+.55*t);
      if(al<.018)continue;
      x.globalAlpha=al>.5?.5:al;
      x.strokeStyle=`hsl(${hue+22},${52+t*16}%,${50+dep*22}%)`;
      x.beginPath();x.moveTo(a.x,a.y);x.lineTo(b.x,b.y);x.stroke()}
    x.globalAlpha=1}

  /* ── Sequenzpfad ── */
  if(T>1&&front){
    x.globalAlpha=op*(.24+.3*t);
    x.strokeStyle=`hsl(${hue},68%,70%)`;
    x.lineWidth=1.35;
    x.beginPath();x.moveTo(S[0].x,S[0].y);
    for(let i=1;i<T;i++)x.lineTo(S[i].x,S[i].y);
    x.stroke();x.globalAlpha=1}

  /* ── Knoten ── */
  const ord2=S.map((s,i)=>i).sort((a,b)=>S[b].z-S[a].z);
  for(const i of ord2){
    const s=S[i], dep=1-(s.z-zmin)*zr, r=(1.1+s.f*2.3)*(front?1:.6);
    const al=(.3+t*.26+dep*.2)*op;
    x.globalAlpha=al>1?1:al;
    x.fillStyle=`hsl(${hue+i*1.2},${52+t*20}%,${60+dep*16}%)`;
    x.beginPath();x.arc(s.x,s.y,r,0,6.283);x.fill()}
  x.globalAlpha=1;

  if(front&&op>.9){
    for(const i of ord2.slice(-6)){
      const s=S[i],rr=3.4+s.f*7.5;
      const g=x.createRadialGradient(s.x,s.y,0,s.x,s.y,rr);
      g.addColorStop(0,`hsla(${hue+44},92%,78%,${.18*op})`);
      g.addColorStop(1,`hsla(${hue+44},92%,66%,0)`);
      x.globalCompositeOperation='lighter';
      x.fillStyle=g;x.beginPath();x.arc(s.x,s.y,rr,0,6.283);x.fill();
      x.globalCompositeOperation='source-over'}
    x.globalAlpha=1}

  /* ── Token-Labels ── */
  if(front&&showTok&&op>.9){
    x.font='10px ui-monospace,monospace'; x.textAlign='center';
    for(let i=0;i<T;i++){
      const s=S[i]; if(s.f<.66)continue;
      x.globalAlpha=Math.min(.9,(s.f-.6)*2.3);
      x.fillStyle=`hsl(${hue+16},54%,86%)`;
      x.fillText(sr.tokens[i]||'',s.x,s.y-9)}}
  x.globalAlpha=1}

/* ════════════════════════════════════════════════════════════════════
   5 · SATZ-EDITOR
   ════════════════════════════════════════════════════════════════════ */
const serList=document.getElementById('serList'),
      serAdd=document.getElementById('serAdd'),
      serCount=document.getElementById('serCount');

function hsl2hex(h){
  const s=.62,l=.60,a=s*Math.min(l,1-l);
  const f=n=>{const k=(n+h/30)%12;
    const v=l-a*Math.max(-1,Math.min(k-3,Math.min(9-k,1)));
    return Math.round(v*255).toString(16).padStart(2,'0')};
  return '#'+f(0)+f(8)+f(4)}
function hex2hue(hex){
  const r=parseInt(hex.slice(1,3),16)/255,g=parseInt(hex.slice(3,5),16)/255,
        b=parseInt(hex.slice(5,7),16)/255;
  const mx=Math.max(r,g,b),mn=Math.min(r,g,b),d=mx-mn;
  if(d<1e-6)return 0;
  let h;
  if(mx===r)h=((g-b)/d+6)%6; else if(mx===g)h=(b-r)/d+2; else h=(r-g)/d+4;
  return (h*60+360)%360}
const esc=s=>String(s).replace(/[&<>"]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[m]));

function renderRows(){
  serList.innerHTML='';
  ROWS.forEach((rw,i)=>{
    const d=document.createElement('div');
    d.className='ser'+(rw.on?'':' off');
    d.style.setProperty('--sc',`hsl(${rw.hue},66%,66%)`);
    d.innerHTML=
      '<div class="serhead">'+
        `<input class="nm" data-k="label" data-i="${i}" value="${esc(rw.label)}" maxlength="10" aria-label="Name">`+
        '<label class="sb" title="Farbe" style="position:relative">◍'+
          `<input type="color" data-k="hue" data-i="${i}" value="${hsl2hex(rw.hue)}"></label>`+
        `<div class="sb${rw.on?' act':''}" data-k="on" data-i="${i}" title="Ein/Aus" role="button" tabindex="0">${rw.on?'◉':'○'}</div>`+
        `<div class="sb del" data-k="del" data-i="${i}" title="Entfernen" role="button" tabindex="0">✕</div>`+
      '</div>'+
      `<textarea data-k="text" data-i="${i}" placeholder="Satz eingeben …">${esc(rw.text)}</textarea>`;
    serList.appendChild(d)});

  serList.querySelectorAll('[data-k]').forEach(el=>{
    const i=+el.dataset.i, k=el.dataset.k;
    if(k==='del'){el.onclick=()=>{if(ROWS.length<=1)return;ROWS.splice(i,1);renderRows()};return}
    if(k==='on'){el.onclick=()=>{ROWS[i].on=!ROWS[i].on;renderRows()};return}
    if(k==='hue'){el.oninput=()=>{ROWS[i].hue=hex2hue(el.value);
      el.closest('.ser').style.setProperty('--sc',`hsl(${ROWS[i].hue},66%,66%)`)};return}
    el.oninput=()=>{ROWS[i][k]=el.value}});

  serCount.textContent=ROWS.length+' / '+MAXS;
  serAdd.classList.toggle('dis',ROWS.length>=MAXS)}

serAdd.onclick=()=>{
  if(ROWS.length>=MAXS)return;
  const i=ROWS.length;
  ROWS.push({text:'',label:'S'+(i+1),hue:autoHue(i),on:true});
  renderRows();
  const tas=serList.querySelectorAll('textarea');
  tas[tas.length-1].focus()};
renderRows();

/* ════════════════════════════════════════════════════════════════════
   6 · PIPELINE-EDITOR
   ════════════════════════════════════════════════════════════════════ */
const RED_OPTS=`__REDOPTS__`;
const stagesHost=document.getElementById('stages');
let STAGES=[{m:'umap',d:12},{m:'pca',d:3}];

function renderStages(){
  stagesHost.innerHTML='';
  STAGES.forEach((st,i)=>{
    const row=document.createElement('div');
    row.className='row3 stage';
    row.innerHTML=
      `<select data-k="m" data-i="${i}" aria-label="Methode">${RED_OPTS}</select>`+
      `<input type="text" data-k="d" data-i="${i}" value="${st.d}" aria-label="Dimensionen">`+
      `<div class="ix">${i===STAGES.length-1?'→3D':'↓'}</div>`;
    stagesHost.appendChild(row);
    row.querySelector('[data-k="m"]').value=st.m});
  stagesHost.querySelectorAll('select,input').forEach(el=>{
    el.onchange=el.oninput=()=>{
      const i=+el.dataset.i,k=el.dataset.k;
      if(k==='m')STAGES[i].m=el.value;
      else{const v=parseInt(el.value,10);
        STAGES[i].d=isNaN(v)?3:Math.max(1,Math.min(512,v))}}})}

document.getElementById('addStage').onclick=()=>{
  if(STAGES.length>=4)return;
  STAGES.splice(STAGES.length-1,0,{m:'pca',d:50}); renderStages()};
document.getElementById('delStage').onclick=()=>{
  if(STAGES.length<=1)return; STAGES.splice(STAGES.length-2,1); renderStages()};
renderStages();

/* ════════════════════════════════════════════════════════════════════
   7 · SLIDER
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
    d.innerHTML='<label>'+lbl+'</label>';
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
   8 · TRANSPORT · LEGENDE · SICHTBARKEIT
   ════════════════════════════════════════════════════════════════════ */
const bPlay=document.getElementById('bPlay'),
      bSurf=document.getElementById('bSurf'),
      bAll=document.getElementById('bAll'),
      bHC=document.getElementById('bHC'),
      legHost=document.getElementById('leg');

function sync(){
  bPlay.textContent=playing?'❚❚':'▶';
  bPlay.classList.toggle('on',playing)}

function refresh(){lastHudLayer=-1;renderLegend();buildTokens();updateHUD()}

function renderLegend(){
  legHost.innerHTML='';
  if(!D)return;
  D.series.forEach((s,i)=>{
    const d=document.createElement('div');
    const vis=visible(i);
    d.className='lgi'+(vis?'':' off')+(i===FOCUS?' foc':'');
    d.innerHTML=`<u style="background:hsl(${s.hue},70%,64%);color:hsl(${s.hue},70%,64%)"></u>`+
      `<span>${esc(s.label)}</span><s>${i+1}</s>`;
    d.title=s.text+'  ·  Klick: ein/aus  ·  ⇧+Klick: Solo  ·  Alt+Klick: Fokus';
    d.onclick=e=>{
      if(e.altKey){FOCUS=i}
      else if(e.shiftKey){SOLO=(SOLO===i?-1:i);FOCUS=i}
      else{if(SOLO>=0)SOLO=-1; VIS[i]=!VIS[i];
        if(!VIS.some(v=>v))VIS[i]=true;
        if(FOCUS===i&&!VIS[i]){const nx=VIS.findIndex(v=>v);if(nx>=0)FOCUS=nx}}
      refresh()};
    legHost.appendChild(d)});
  bAll.classList.toggle('on',SOLO<0&&VIS.every(v=>v))}

bPlay.onclick=()=>{if(!D)return;playing=!playing;targetPos=null;sync()};
document.getElementById('bPrev').onclick=()=>{playing=false;
  targetPos=Math.max(0,Math.round(pos)-1);sync()};
document.getElementById('bNext').onclick=()=>{playing=false;
  targetPos=Math.min(LAYERS-1,Math.round(pos)+1);sync()};
bSurf.onclick=()=>{surf=!surf;bSurf.classList.toggle('on',surf);
  bSurf.textContent=surf?'◆':'◇'};
bAll.onclick=()=>{
  if(!D)return;
  const all=SOLO<0&&VIS.every(v=>v);
  SOLO=-1; VIS=VIS.map(()=>!all);
  if(!VIS.some(v=>v))VIS[0]=true;
  refresh()};
bHC.onclick=()=>{
  const on=document.documentElement.getAttribute('data-hc')==='1';
  document.documentElement.setAttribute('data-hc',on?'0':'1');
  bHC.classList.toggle('on',!on);
  try{localStorage.setItem('mf_hc',on?'0':'1')}catch(_){} };
if((()=>{try{return localStorage.getItem('mf_hc')==='1'}catch(_){return false}})()){
  document.documentElement.setAttribute('data-hc','1');bHC.classList.add('on')}

addEventListener('keydown',e=>{
  const tn=e.target.tagName;
  if(tn==='TEXTAREA'||tn==='SELECT'||tn==='INPUT')return;
  if(e.code==='Space'){e.preventDefault();
    if(D){playing=!playing;targetPos=null;sync()}return}
  if(e.code==='ArrowUp'||e.code==='ArrowRight'){e.preventDefault();playing=false;
    targetPos=Math.min(LAYERS-1,Math.round(pos)+1);sync();return}
  if(e.code==='ArrowDown'||e.code==='ArrowLeft'){e.preventDefault();playing=false;
    targetPos=Math.max(0,Math.round(pos)-1);sync();return}
  if(e.code==='Home'){e.preventDefault();playing=false;targetPos=0;sync();return}
  if(e.code==='End'){e.preventDefault();playing=false;targetPos=LAYERS-1;sync();return}
  if(e.code==='KeyS'){bSurf.onclick();return}
  if(e.code==='KeyA'){bAll.onclick();return}
  if(e.code==='KeyH'){bHC.onclick();return}
  if(e.key>='1'&&e.key<='8'&&D){
    const i=+e.key-1; if(i>=D.series.length)return;
    if(e.shiftKey){SOLO=(SOLO===i?-1:i);FOCUS=i}
    else{if(SOLO>=0)SOLO=-1;VIS[i]=!VIS[i];
      if(!VIS.some(v=>v))VIS[i]=true}
    refresh(); return}
  if(e.code==='Enter'&&(e.metaKey||e.ctrlKey))run()});

/* ════════════════════════════════════════════════════════════════════
   9 · SPARKLINES
   ════════════════════════════════════════════════════════════════════ */
function sparkSetup(cv){
  const r=cv.getBoundingClientRect();
  const dpr=Math.min(devicePixelRatio||1,2);
  cv.width=Math.max(1,r.width*dpr); cv.height=Math.max(1,r.height*dpr);
  const g=cv.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0);
  g.clearRect(0,0,r.width,r.height); return [g,r.width,r.height]}

function sparkLines(cv,series,markIdx){
  const [g,w,h]=sparkSetup(cv);
  if(!series.length)return;
  let lo=Infinity,hi=-Infinity;
  for(const s of series)for(const v of s.v){if(v<lo)lo=v;if(v>hi)hi=v}
  if(!isFinite(lo))return;
  if(hi-lo<1e-9)hi=lo+1;
  const pad=5,n=series[0].v.length;
  const X=i=>pad+(n>1?i/(n-1):0)*(w-2*pad);
  const Y=v=>h-pad-((v-lo)/(hi-lo))*(h-2*pad);
  if(markIdx!==undefined){
    g.strokeStyle='rgba(170,195,250,.34)';g.lineWidth=1;
    g.beginPath();g.moveTo(X(markIdx),0);g.lineTo(X(markIdx),h);g.stroke()}
  for(const s of series){
    g.beginPath();
    for(let i=0;i<s.v.length;i++){const px=X(i),py=Y(s.v[i]);i?g.lineTo(px,py):g.moveTo(px,py)}
    g.lineTo(X(s.v.length-1),h-pad);g.lineTo(X(0),h-pad);g.closePath();
    const grad=g.createLinearGradient(0,0,0,h);
    grad.addColorStop(0,s.c.replace(')',',.26)').replace('hsl(','hsla('));
    grad.addColorStop(1,'rgba(0,0,0,0)');
    g.fillStyle=grad;g.fill();
    g.beginPath();
    for(let i=0;i<s.v.length;i++){const px=X(i),py=Y(s.v[i]);i?g.lineTo(px,py):g.moveTo(px,py)}
    g.strokeStyle=s.c;g.lineWidth=s.w||1.5;g.lineJoin='round';g.stroke();
    if(markIdx!==undefined&&s.v[markIdx]!==undefined){
      g.fillStyle=s.c;g.beginPath();g.arc(X(markIdx),Y(s.v[markIdx]),2.4,0,6.283);g.fill()}}}

function sparkBars(cv,groups){
  const [g,w,h]=sparkSetup(cv);
  if(!groups.length)return;
  const n=Math.max(...groups.map(q=>q.v.length),1),pad=4;
  const bw=(w-2*pad)/n, k=groups.length, sw=bw*0.82/k;
  let mx=1e-9; for(const q of groups)for(const v of q.v)if(v>mx)mx=v;
  for(let i=0;i<n;i++){
    const px=pad+i*bw+bw*0.09;
    groups.forEach((q,s)=>{
      const hh=((q.v[i]||0)/mx)*(h-2*pad);
      g.fillStyle=q.c.replace('hsl(','hsla(').replace(')',',.78)');
      g.fillRect(px+s*sw,h-pad-hh,sw*0.86,hh)})}}

/* ════════════════════════════════════════════════════════════════════
   10 · HUD
   ════════════════════════════════════════════════════════════════════ */
const $=id=>document.getElementById(id);
const f3=v=>(v===undefined||v===null)?'—':(Math.abs(v)<10?v.toFixed(3):v.toFixed(1));
const i0=v=>(v===undefined||v===null)?'—':String(Math.round(v));
let lastHudLayer=-1;

function activeIdx(){
  const out=[];
  if(!D)return out;
  for(let i=0;i<D.series.length;i++)if(visible(i))out.push(i);
  return out}

function updateHUD(force){
  if(!D)return;
  const L=Math.round(Math.max(0,Math.min(LAYERS-1,pos)));
  if(L===lastHudLayer&&!force)return;
  lastHudLayer=L;

  const act=activeIdx();
  const fi=visible(FOCUS)?FOCUS:(act.length?act[0]:0);
  const s=D.series[fi];
  const sh=s.shapes[L], rc=s.rips[L];

  $('focLabel').textContent=s.label;
  $('focLabel').style.color=`hsl(${s.hue},70%,66%)`;
  $('s-drift').textContent=f3(s.drift[L]);
  $('s-eps').textContent=f3(rc.eps);
  $('s-tris').textContent=i0(rc.n_tris);
  $('s-euler').textContent=i0(rc.euler);
  $('s-edim').textContent=f3(sh.eff_dim);
  $('s-aniso').textContent=f3(sh.anisotropy);
  $('s-curv').textContent=f3(sh.curvature);
  $('s-norm').textContent=f3(s.norms[L]);

  /* ── paarweise W1 ── */
  const host=$('emdList'), pairs=(D.emd[L]||[])
    .filter(p=>visible(p.i)&&visible(p.j))
    .sort((a,b)=>b.w-a.w);
  $('emdCount').textContent=pairs.length?pairs.length+' PAARE':'—';
  if(!pairs.length){
    host.innerHTML=`<div class="pairrow"><span>${
      act.length<2?'MIND. 2 SÄTZE NÖTIG':'—'}</span><b>—</b></div>`;
  }else{
    host.innerHTML=pairs.map(p=>{
      const a=D.series[p.i],b=D.series[p.j];
      return `<div class="pairrow"><span>`+
        `<u style="display:inline-block;width:7px;height:7px;border-radius:2px;`+
        `background:hsl(${a.hue},70%,64%);margin-right:4px"></u>${esc(a.label)}`+
        `<em style="font-style:normal;color:var(--txt-faint);padding:0 4px">↔</em>`+
        `<u style="display:inline-block;width:7px;height:7px;border-radius:2px;`+
        `background:hsl(${b.hue},70%,64%);margin-right:4px"></u>${esc(b.label)}`+
        `</span><b>${f3(p.w)}</b></div>`}).join('')}

  $('hs').textContent=(L===0?'EMBEDDING':'BLOCK '+String(L).padStart(2,'0'))
    +' · '+act.length+(act.length===1?' SATZ':' SÄTZE');
  $('hp').textContent=D.pipe_label||'';

  /* ── Sparklines ── */
  const ser1=act.map(i=>({v:D.series[i].drift,
    c:`hsl(${D.series[i].hue},68%,64%)`,w:1.5}));
  sparkLines($('sp1'),ser1,L);
  $('leg1').innerHTML='<i><u style="background:'+
    'linear-gradient(90deg,#8aa0c8,#8aa0c8)"></u>DRIFT L-1→L</i>'+
    act.map(i=>`<i><u style="background:hsl(${D.series[i].hue},68%,64%)"></u>`+
      `${esc(D.series[i].label)}</i>`).join('');

  const ser2=act.map(i=>({v:D.series[i].shapes.map(q=>q.eff_dim),
    c:`hsl(${D.series[i].hue},68%,64%)`,w:1.5}));
  sparkLines($('sp2'),ser2,L);
  $('leg2').innerHTML='<i>EFF. DIM</i>';

  sparkBars($('sp3'),act.map(i=>({v:D.series[i].shapes[L].spectrum,
    c:`hsl(${D.series[i].hue},68%,62%)`})));
  $('leg3').innerHTML='<i>EIGENWERT-ANTEILE · LAYER '+
    String(L).padStart(2,'0')+'</i>'}

/* ════════════════════════════════════════════════════════════════════
   11 · TOKEN-LEISTE
   ════════════════════════════════════════════════════════════════════ */
function buildTokens(){
  const host=$('toks'); host.innerHTML='';
  if(!D)return;
  const act=activeIdx();
  if(act.length>4)return;              // sonst wird es unlesbar
  act.forEach(i=>{
    const s=D.series[i];
    const row=document.createElement('div'); row.className='trow';
    s.tokens.slice(0,40).forEach(t=>{
      const e=document.createElement('div');
      e.className='tk'; e.textContent=t;
      e.style.color=`hsl(${s.hue},62%,78%)`;
      e.style.borderColor=`hsla(${s.hue},62%,62%,.28)`;
      row.appendChild(e)});
    host.appendChild(row)})}

/* ════════════════════════════════════════════════════════════════════
   12 · HINTERGRUND
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
    const al=Math.max(0,(1-d/3.6))*(near?.2:.1);
    if(al<.006)continue;
    x.globalAlpha=al;
    x.strokeStyle=near?'hsl(196,48%,66%)':'hsl(218,34%,60%)';
    x.lineWidth=near?.9:.55;
    x.beginPath();
    for(let k=0;k<=52;k++){
      const a=k/52*6.2832;
      project([Math.cos(a)*1.22,0,Math.sin(a)*1.22],L,o);
      k?x.lineTo(o.x,o.y):x.moveTo(o.x,o.y)}
    x.closePath(); x.stroke()}
  x.globalAlpha=1}

/* ════════════════════════════════════════════════════════════════════
   13 · LOOP
   ════════════════════════════════════════════════════════════════════ */
const hn=$('hn');
let last=performance.now();
function loop(now){
  const dt=Math.min(.05,(now-last)/1000); last=now;
  const ease=(cur,t_,sp)=>cur+(t_-cur)*(1-Math.exp(-sp*dt));

  yaw=ease(yaw,tg.y,6); pit=ease(pit,tg.p,6);
  breathe+=dt*.55;

  if(D){
    if(targetPos!==null){
      pos=ease(pos,targetPos,STEP_SPD);
      if(Math.abs(pos-targetPos)<.004){pos=targetPos;targetPos=null}
    }else if(playing&&scrub===null){
      pos+=AUTO_SPD*dt;
      if(pos>LAYERS-1){pos=LAYERS-1;playing=false;sync()}}
    camY=ease(camY,-pos*RISE,3.6)}

  background();

  if(D){
    const act=activeIdx();
    /* Geister-Layer unterhalb */
    for(let g=GH;g>=1;g--){
      const lf=pos-g*GAP; if(lf<-.05)continue;
      const op=Math.pow(.3,g)*.72;
      for(const i of act)
        drawManifold(D.series[i],Math.max(0,lf),-g*4,op,false,false)}
    /* aktive Ebene */
    const showTok=act.length<=3;
    for(const i of act)
      drawManifold(D.series[i],pos,0,1,true,showTok);

    hn.textContent=String(Math.round(pos)).padStart(2,'0');
    updateRail(); updateHUD()}
  requestAnimationFrame(loop)}

/* ════════════════════════════════════════════════════════════════════
   14 · SERVER-CALL
   ════════════════════════════════════════════════════════════════════ */
const msg=$('msg'), go=$('go'), boot=$('boot'), bootmsg=$('bootmsg');
let busy=false;

async function run(){
  if(busy)return;
  const act=ROWS.filter(r=>r.on&&r.text.trim());
  if(!act.length){
    msg.className='err'; msg.textContent='Kein aktiver Satz mit Text.'; return}
  if(act.length>MAXS){
    msg.className='err'; msg.textContent='Maximal '+MAXS+' Sätze.'; return}

  busy=true;
  go.classList.add('busy'); go.textContent='RECHNET …';
  msg.className=''; msg.textContent='Hidden States · Rips-Komplex · Transport …';
  boot.classList.remove('gone');
  bootmsg.textContent='FORWARD-PASS · VIETORIS-RIPS · PIPELINE';

  const body={
    texts:act.map(r=>r.text),
    labels:act.map(r=>r.label),
    hues:act.map(r=>r.hue),
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
    pos=0; camY=0; targetPos=null; playing=false;
    VIS=D.series.map(()=>true); SOLO=-1; FOCUS=0; lastHudLayer=-1;
    buildRail(); renderLegend(); buildTokens(); sync(); updateHUD(true);

    const tks=D.series.map(s=>s.tokens.length).join('/');
    const tri=D.series[0].rips[0]?D.series[0].rips[0].n_tris:0;
    msg.className='ok';
    msg.textContent=`${D.model.split('/').pop()} · ${LAYERS} Layer · `
      +`${D.n_series} Satz${D.n_series>1?'·e':''} · ${tks} Tokens · `
      +`${tri} △ · ${D.device.toUpperCase()}`;
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
""".replace("__OPTS__", opts) \
   .replace("__REDOPTS__", red_opts) \
   .replace("__DEFAULTS__", defaults)

# ============================================================================
# 11) HTTP-SERVER
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

            # --- Texte: KEIN stiller Default mehr ---------------------------
            raw = req.get("texts")
            if raw is None:                      # Rueckwaerts-Kompatibilitaet
                raw = [req.get("text_a"), req.get("text_b")]
            texts = [str(t).strip() for t in (raw or []) if t and str(t).strip()]
            if not texts:
                raise ValueError("Keine Sätze übergeben.")
            if len(texts) > MAX_SERIES:
                texts = texts[:MAX_SERIES]

            labels = req.get("labels") or []
            hues = req.get("hues") or []
            labels = [str(l)[:10] for l in labels][:len(texts)]
            hues = [float(h) % 360.0 for h in hues
                    if isinstance(h, (int, float))][:len(texts)]

            model = req.get("model") or "distilbert-base-uncased"
            emd = req.get("emd") or "exact"
            if emd not in ("exact", "sliced", "sinkhorn"):
                emd = "exact"
            density = float(req.get("density") or 1.0)
            density = max(0.2, min(6.0, density))

            pipeline = req.get("pipeline") or [{"method": "pca", "dims": 3}]
            clean = []
            for st in pipeline[:4]:
                m = str(st.get("method", "pca")).lower()
                if m not in REDUCERS:
                    m = "pca"
                d = int(st.get("dims", 3))
                clean.append({"method": m, "dims": max(1, min(512, d))})

            lbl = " -> ".join(f'{s["method"]}:{s["dims"]}' for s in clean)
            print(f"[api] {model} | n={len(texts)} | {lbl} | "
                  f"emd={emd} | rho={density}")

            data = analyse(texts, model, clean, emd, density=density,
                           hues=hues or None, labels=labels or None)
            self._send(200, json.dumps(data).encode("utf-8"),
                       "application/json; charset=utf-8")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send(500, json.dumps({"error": str(e)}).encode("utf-8"),
                       "application/json; charset=utf-8")

# ============================================================================
# 12) ENTRY POINT
# ============================================================================

def main():
    port = int(os.environ.get("MANIFOLD_PORT", "8051"))
    url = f"http://127.0.0.1:{port}"
    print("=" * 74)
    print("  HIDDEN-STATE MANIFOLD EXPLORER  ·  v4  ·  n Saetze · Vietoris-Rips")
    print("=" * 74)
    print(f"  Device    : {DEVICE.upper()}")
    print(f"  CUDA      : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU       : {torch.cuda.get_device_name(0)}")
    cpu_ok = sum(1 for v in MODELS.values() if v["tier"].startswith("CPU"))
    print(f"  Modelle   : {len(MODELS)}  ({cpu_ok} CPU-tauglich)")
    print(f"  Reducer   : {', '.join(k for k in REDUCERS if k != 'none')}")
    print(f"  Saetze    : 1 .. {MAX_SERIES}")
    print(f"  URL       : {url}")
    print("=" * 74)
    print("  Bedienung : ▶ Durchlauf · ◂ ▸ Layer einzeln · ◆ Flaeche · ◉ alle")
    print("              1-8 Satz an/aus · ⇧+1-8 Solo · A Hochkontrast")
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
