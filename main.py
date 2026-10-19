# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch", "transformers", "numpy", "scikit-learn",
#   "pot", "umap-learn", "accelerate",
# ]
# ///
"""
Hidden-State Manifold Explorer  ·  v5
=====================================
EINE morphende Isoflaeche · Marching Cubes · WebGL · Layer-Interpolation

Der Server liefert pro Layer ein Skalarfeld (Dichte der Token-Positionen)
auf einem Wuerfelgitter.  Der Client blendet zwischen den Gittern linear
und laesst Marching Cubes pro Frame neu laufen -> echtes Morphing statt
Topologie-Sprung.

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

import base64
import json
import threading
import webbrowser
import zlib
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

MAX_SERIES = 8

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
    hs = torch.stack(out.hidden_states)[:, 0].float().cpu().numpy()
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
    "umap":   {"label": "UMAP",           "max_out": None, "note": "lokal+global"},
    "isomap": {"label": "ISOMAP",         "max_out": None, "note": "geodaetisch"},
    "mds":    {"label": "MDS",            "max_out": None, "note": "distanz-erhaltend"},
    "none":   {"label": "— KEINE —",      "max_out": None, "note": "Stufe ueberspringen"},
}

def _one_stage(X: np.ndarray, method: str, dims: int, seed: int = 0) -> np.ndarray:
    n, d = X.shape
    if method == "none" or dims <= 0:
        return X

    cap = REDUCERS.get(method, {}).get("max_out")
    if cap:
        dims = min(dims, cap)
    dims = max(1, min(dims, d))
    if n > 1 and method in ("pca", "tsne", "isomap"):
        dims = min(dims, n - 1)
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

def global_pipeline(HS_list, pipeline, seed=0):
    """
    EINE Reduktion ueber ALLE Saetze UND ALLE Layer gemeinsam.

    Das ist der Kern des Smooth-Morphings: ein einziges, global gefittetes
    Koordinatensystem.  Damit entfaellt Procrustes komplett und es gibt
    kein Achsen-Umklappen mehr zwischen benachbarten Layern.
    """
    shapes = [(h.shape[0], h.shape[1]) for h in HS_list]   # (L, T) je Satz
    flat = np.vstack([h.reshape(-1, h.shape[-1]) for h in HS_list])
    Z = run_pipeline(flat, pipeline, seed=seed)

    out, i = [], 0
    for (L, T) in shapes:
        n = L * T
        out.append(Z[i:i + n].reshape(L, T, 3))
        i += n
    return out

# ============================================================================
# 7) DICHTEFELD  (ersetzt Vietoris-Rips als Render-Geometrie)
# ============================================================================

def density_grid(P: np.ndarray, res: int, sigma: float,
                 lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """
    Summe isotroper Gauss-Kernel um jeden Token-Punkt, auf ein res^3-Gitter.

    f(v) = Σ_t exp(-|v - p_t|² / (2σ²))

    Die Isoflaeche f = iso ist dann eine geschlossene, glatte Mannigfaltigkeit
    -- im Gegensatz zu Rips-Dreiecken existiert sie wirklich als Oberflaeche
    und laesst sich zwischen Layern linear interpolieren.
    """
    ax = [np.linspace(lo[d], hi[d], res, dtype=np.float32) for d in range(3)]

    # separabler Kernel: exp(-(dx²+dy²+dz²)/2σ²) = e^-dx² · e^-dy² · e^-dz²
    inv = 1.0 / (2.0 * sigma * sigma)
    field = np.zeros((res, res, res), dtype=np.float32)

    for p in P:
        gx = np.exp(-((ax[0] - p[0]) ** 2) * inv)
        gy = np.exp(-((ax[1] - p[1]) ** 2) * inv)
        gz = np.exp(-((ax[2] - p[2]) ** 2) * inv)
        # Beitraege unter einem Schwellwert verwerfen -> grosse Ersparnis
        ix = np.where(gx > 1e-4)[0]
        iy = np.where(gy > 1e-4)[0]
        iz = np.where(gz > 1e-4)[0]
        if not (len(ix) and len(iy) and len(iz)):
            continue
        sub = (gx[ix][:, None, None] *
               gy[iy][None, :, None] *
               gz[iz][None, None, :])
        field[np.ix_(ix, iy, iz)] += sub

    return field

def pack_grids(grids: np.ndarray) -> str:
    """
    [L, res, res, res] float32 -> uint8 (quantisiert) -> zlib -> base64.

    Das Feld ist nach Konstruktion >= 0 und wir normieren global, also
    verliert uint8 nur Rauschen weit unterhalb der Iso-Schwelle.
    """
    g = np.asarray(grids, dtype=np.float32)
    mx = float(g.max()) + 1e-9
    q = np.clip(g / mx * 255.0, 0, 255).astype(np.uint8)
    blob = zlib.compress(q.tobytes(), 6)
    return base64.b64encode(blob).decode("ascii"), mx

# ============================================================================
# 8) Farb-Zuweisung
# ============================================================================

GOLDEN = 137.50776405003785

def auto_hue(i: int, base: float = 193.0) -> float:
    return (base + i * GOLDEN) % 360.0

# ============================================================================
# 9) Analyse-Pipeline
# ============================================================================

def pick_resolution(n_layers: int, requested: int) -> int:
    """
    Adaptives Grid: Gesamt-Budget ~ 14 MB roh (vor Kompression).
    Bei 7 Layern sind 48³ problemlos, bei 32 Layern landen wir bei 32³.
    """
    budget = 14_000_000
    r = int(requested)
    while r > 20 and n_layers * r ** 3 > budget:
        r -= 4
    return max(20, r)

def analyse(texts, model_name, pipeline, emd_mode,
            iso=0.55, smooth=1.0, grid_res=48, hues=None, labels=None):
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
    HS = [h[:n_layers] for h in HS]

    # ── EINE globale Reduktion ueber alles -------------------------------
    PROJ = global_pipeline(HS, pipeline, seed=0)          # [s][L,T,3]

    # ── gemeinsame Normierung in [-1, 1] ---------------------------------
    allp = np.vstack([p.reshape(-1, 3) for p in PROJ])
    centre = allp.mean(0)
    scale = float(np.abs(allp - centre).max()) + 1e-9
    PROJ = [((p - centre) / scale).astype(np.float32) for p in PROJ]

    # ── Gitter-Parameter --------------------------------------------------
    res = pick_resolution(n_layers, grid_res)
    pad = 0.30
    lo = np.array([-1 - pad] * 3, dtype=np.float32)
    hi = np.array([+1 + pad] * 3, dtype=np.float32)

    # Kernel-Breite: skaliert mit mittlerem Nachbarabstand, damit die
    # Form weder zur Kugel verschmilzt noch in Einzelblasen zerfaellt.
    nn_all = []
    for p in PROJ:
        for L in range(n_layers):
            Q = p[L]
            if len(Q) < 2:
                continue
            d2 = ((Q[:, None, :] - Q[None, :, :]) ** 2).sum(-1)
            np.fill_diagonal(d2, 1e9)
            nn_all.append(np.sqrt(d2.min(1)).mean())
    base_sigma = float(np.median(nn_all)) if nn_all else 0.25
    sigma = max(0.055, min(0.42, base_sigma * 1.35 * float(smooth)))

    # ── Felder, Metriken, Transport --------------------------------------
    fields = [np.zeros((n_layers, res, res, res), dtype=np.float32)
              for _ in range(n_series)]
    shapes = [[] for _ in range(n_series)]
    emd_layers = []

    for L in range(n_layers):
        for s in range(n_series):
            fields[s][L] = density_grid(PROJ[s][L], res, sigma, lo, hi)
            shapes[s].append(shape_metrics(HS[s][L]))

        pairs = []
        for i in range(n_series):
            for j in range(i + 1, n_series):
                pairs.append(dict(i=i, j=j,
                                  w=wasserstein(HS[i][L], HS[j][L], mode=emd_mode)))
        emd_layers.append(pairs)

    series = []
    for s in range(n_series):
        drift_s = [0.0] + [wasserstein(HS[s][L - 1], HS[s][L], mode=emd_mode)
                           for L in range(1, n_layers)]
        norms_s = [float(np.linalg.norm(HS[s][L], axis=1).mean())
                   for L in range(n_layers)]
        hue = (hues[s] if hues and s < len(hues) and hues[s] is not None
               else auto_hue(s))
        label = (labels[s] if labels and s < len(labels) and labels[s]
                 else f"S{s + 1}")

        grid_b64, grid_max = pack_grids(fields[s])

        series.append(dict(
            id=s,
            label=label,
            text=texts[s],
            hue=float(hue) % 360.0,
            tokens=toks[s],
            pts=[p.round(4).tolist() for p in PROJ[s]],   # [L][T][3]
            grid=grid_b64,
            grid_max=float(grid_max),
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
        res=res,
        lo=[float(v) for v in lo],
        hi=[float(v) for v in hi],
        sigma=float(sigma),
        iso=float(iso),
        model=model_name,
        pipeline=pipeline,
        pipe_label=pipe_label,
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
<title>hidden-state manifold · v5</title>
<style>
  :root{
    --bg:#05070d;
    --surface:rgba(17,22,36,.70);
    --surface2:rgba(10,13,22,.58);
    --line:rgba(130,152,208,.15);
    --line-hi:rgba(165,190,245,.32);
    --txt:#c9d2e8; --txt-dim:#9aa5c4; --txt-label:#aab6d6;
    --txt-faint:#7d88a8; --txt-strong:#e9eefb;
    --err:#ff8fa3; --ok:#8df0b4;
    --a:hsl(193,76%,66%);
  }
  :root[data-hc="1"]{
    --txt:#e8eefc; --txt-dim:#c4cee8; --txt-label:#d4ddf4;
    --txt-faint:#aeb9d6; --txt-strong:#fff;
    --line:rgba(150,175,235,.30); --line-hi:rgba(190,212,255,.55);
    --surface:rgba(20,26,42,.86); --surface2:rgba(13,17,28,.80);
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg);overflow:hidden;
    font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--txt-dim)}
  canvas#gl{display:block;position:fixed;inset:0}
  canvas#ov{display:block;position:fixed;inset:0;pointer-events:none;z-index:2}
  ::-webkit-scrollbar{width:6px}
  ::-webkit-scrollbar-track{background:transparent}
  ::-webkit-scrollbar-thumb{background:rgba(140,162,215,.28);border-radius:3px}
  :focus-visible{outline:2px solid var(--a);outline-offset:2px}

  #ui{position:fixed;left:22px;top:22px;z-index:14;width:336px;
    max-height:calc(100vh - 44px);overflow-y:auto;padding-right:7px}
  .card{background:linear-gradient(165deg,var(--surface),var(--surface2));
    border:1px solid var(--line);border-radius:12px;padding:14px 15px;
    margin-bottom:9px;backdrop-filter:blur(18px) saturate(1.25);
    box-shadow:0 8px 32px -16px rgba(0,0,0,.92);transition:border-color .35s}
  .card:hover{border-color:var(--line-hi)}
  .card.accent{border-color:rgba(111,211,247,.26)}
  .lb{display:block;font-size:9px;letter-spacing:.26em;margin-bottom:8px;
    color:var(--txt-label);text-transform:uppercase;font-weight:500}
  .lb em{font-style:normal;float:right;letter-spacing:.1em;color:var(--a);
    text-transform:none}
  textarea,select,input[type=text],input[type=number]{width:100%;
    background:rgba(4,6,12,.80);color:var(--txt);border:1px solid var(--line);
    border-radius:7px;padding:8px 10px;font:11.5px/1.5 ui-monospace,monospace;
    outline:none;resize:none;transition:border-color .25s,box-shadow .25s}
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

  input[type=range]{width:100%;height:18px;appearance:none;background:transparent;
    cursor:pointer;margin:2px 0}
  input[type=range]::-webkit-slider-runnable-track{height:2px;
    background:rgba(130,152,208,.3);border-radius:1px}
  input[type=range]::-webkit-slider-thumb{appearance:none;width:12px;height:12px;
    margin-top:-5px;border-radius:50%;background:var(--a);
    box-shadow:0 0 12px -2px var(--a)}
  input[type=range]::-moz-range-track{height:2px;background:rgba(130,152,208,.3)}
  input[type=range]::-moz-range-thumb{width:12px;height:12px;border:0;
    border-radius:50%;background:var(--a)}

  .ser{position:relative;margin-bottom:7px;padding:9px 10px 10px 13px;
    border:1px solid var(--line);border-radius:9px;background:rgba(8,11,19,.5);
    transition:border-color .3s,background .3s}
  .ser:before{content:'';position:absolute;left:0;top:9px;bottom:9px;width:3px;
    border-radius:0 2px 2px 0;background:var(--sc)}
  .ser.off{opacity:.62}
  .ser.off:before{background:#434b66}
  .ser:focus-within{border-color:var(--line-hi);background:rgba(11,15,26,.72)}
  .serhead{display:flex;align-items:center;gap:6px;margin-bottom:6px}
  .serhead .nm{flex:1;background:transparent;border:0;padding:2px 0;
    font:9.5px/1.3 ui-monospace,monospace;letter-spacing:.2em;color:var(--sc);
    text-transform:uppercase;outline:none;min-width:0}
  .serhead .nm:focus{color:var(--txt-strong)}
  .sb{width:23px;height:23px;flex:none;border-radius:6px;cursor:pointer;
    background:rgba(16,21,34,.75);border:1px solid var(--line);color:var(--txt-dim);
    font-size:10px;line-height:1;display:flex;align-items:center;
    justify-content:center;transition:.22s;user-select:none}
  .sb:hover{border-color:var(--line-hi);color:var(--txt-strong)}
  .sb.act{border-color:var(--sc);color:var(--txt-strong);
    box-shadow:0 0 14px -5px var(--sc)}
  .sb.del:hover{border-color:var(--err);color:var(--err)}
  .sb input[type=color]{position:absolute;width:0;height:0;opacity:0;
    pointer-events:none}
  .ser textarea{height:42px;font-size:11px}

  .row3{display:grid;grid-template-columns:1fr 56px 28px;gap:6px;
    align-items:center;margin-bottom:6px}
  .row3 .ix{font-size:8.5px;letter-spacing:.16em;color:var(--txt-faint);
    text-align:center}
  .stage{position:relative;padding-left:14px}
  .stage:before{content:'';position:absolute;left:4px;top:-4px;bottom:50%;
    width:1px;background:var(--line)}
  .stage:after{content:'';position:absolute;left:4px;top:50%;width:7px;height:1px;
    background:var(--line)}
  .stage:first-child:before{top:50%}
  .stage:last-child:before{bottom:50%}
  .mini{width:100%;padding:7px;border-radius:6px;cursor:pointer;
    background:rgba(16,21,34,.7);border:1px dashed var(--line);color:var(--txt-dim);
    font:9.5px/1 ui-monospace,monospace;letter-spacing:.18em;transition:.22s;
    text-align:center;user-select:none}
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
  #go:hover{background:var(--a);color:#05070d;letter-spacing:.34em}
  #go.busy{opacity:.55;pointer-events:none;letter-spacing:.18em}
  #msg{font-size:9.5px;letter-spacing:.06em;margin-top:9px;min-height:13px;
    color:var(--txt-faint);line-height:1.7}
  #msg.err{color:var(--err)} #msg.ok{color:var(--ok)}

  .stat{display:flex;justify-content:space-between;align-items:baseline;gap:10px;
    padding:5px 0;border-bottom:1px solid rgba(130,152,208,.1)}
  .stat:last-child{border:0;padding-bottom:0}
  .stat>span{font-size:9px;letter-spacing:.16em;color:var(--txt-label);flex:none}
  .stat>b{font-weight:400;font-size:12px;color:var(--txt-strong);
    font-variant-numeric:tabular-nums;text-align:right;overflow:hidden;
    text-overflow:ellipsis;white-space:nowrap}
  .pairrow{display:flex;justify-content:space-between;gap:8px;padding:4px 0;
    font-size:10px;border-bottom:1px solid rgba(130,152,208,.08)}
  .pairrow:last-child{border:0}
  .pairrow b{font-weight:400;color:var(--txt-strong);
    font-variant-numeric:tabular-nums}

  #h{position:fixed;right:110px;top:24px;text-align:right;z-index:10;
    font-size:9.5px;letter-spacing:.28em;color:var(--txt-faint);pointer-events:none}
  #h b{display:block;font-size:46px;letter-spacing:.02em;color:var(--txt-strong);
    font-weight:200;margin:-4px 0 0;font-variant-numeric:tabular-nums;
    text-shadow:0 0 40px rgba(111,211,247,.2)}
  #h i{font-style:normal;font-size:9px;letter-spacing:.2em;color:var(--txt-dim);
    display:block;margin-top:2px}
  #h u{text-decoration:none;font-size:9px;letter-spacing:.16em;
    color:var(--txt-faint);display:block;margin-top:5px}

  #rail{position:fixed;right:44px;top:50%;transform:translateY(-50%);width:46px;
    height:60vh;z-index:12;cursor:pointer}
  #line{position:absolute;left:23px;top:0;bottom:0;width:1px;
    background:linear-gradient(180deg,rgba(255,255,255,.04),
      rgba(255,255,255,.2),rgba(255,255,255,.04))}
  #prog{position:absolute;left:22px;bottom:0;width:3px;border-radius:2px;
    background:linear-gradient(180deg,var(--a),rgba(111,211,247,0));opacity:.8}
  .dot{position:absolute;left:19px;width:9px;height:9px;margin-top:-4.5px;
    border-radius:50%;background:#0c0f1a;border:1px solid rgba(165,188,240,.4);
    transition:.3s cubic-bezier(.22,1,.36,1)}
  .dot.pass{border-color:rgba(180,205,255,.66);background:#1b2434}
  .dot.act{background:var(--a);border-color:var(--a);box-shadow:0 0 16px var(--a);
    transform:scale(1.55)}
  .dot label{position:absolute;right:17px;top:-5px;font-size:9px;
    letter-spacing:.14em;color:var(--txt-faint);white-space:nowrap;
    pointer-events:none;transition:color .28s}
  .dot.act label{color:var(--txt-strong);font-weight:500}
  #knob{position:absolute;left:14px;width:19px;height:19px;margin-top:-9.5px;
    border-radius:50%;border:1px solid var(--a);background:rgba(8,11,20,.9);
    box-shadow:0 0 24px -3px var(--a);pointer-events:none}
  #knob:after{content:'';position:absolute;inset:5px;border-radius:50%;
    background:var(--a)}

  #tp{position:fixed;right:28px;bottom:24px;display:flex;gap:6px;z-index:12}
  .btn{min-width:32px;height:32px;padding:0 8px;border-radius:8px;cursor:pointer;
    background:rgba(16,21,34,.76);border:1px solid var(--line);color:var(--txt-dim);
    font-size:11.5px;display:flex;align-items:center;justify-content:center;
    backdrop-filter:blur(12px);transition:.22s;user-select:none}
  .btn:hover{border-color:var(--line-hi);color:var(--txt-strong);
    transform:translateY(-1px)}
  .btn.on{border-color:var(--a);color:var(--txt-strong);
    box-shadow:0 0 18px -5px var(--a)}

  #leg{position:fixed;left:22px;bottom:44px;z-index:11;display:flex;
    flex-direction:column;gap:4px;max-width:300px}
  .lgi{display:flex;align-items:center;gap:7px;font-size:9.5px;letter-spacing:.12em;
    color:var(--txt-dim);cursor:pointer;padding:3px 7px;border-radius:5px;
    background:rgba(10,13,22,.6);border:1px solid transparent;
    backdrop-filter:blur(8px);transition:.2s;user-select:none}
  .lgi:hover{border-color:var(--line-hi);color:var(--txt-strong)}
  .lgi.off{color:var(--txt-faint)}
  .lgi u{text-decoration:none;width:9px;height:9px;border-radius:2px;flex:none}
  .lgi.off u{background:#444b66!important}
  .lgi s{text-decoration:none;font-size:8.5px;color:var(--txt-faint)}

  #toks{position:fixed;left:50%;bottom:18px;transform:translateX(-50%);z-index:11;
    display:flex;flex-direction:column;gap:4px;align-items:center;
    pointer-events:none;max-width:52vw}
  .trow{display:flex;gap:3px;flex-wrap:wrap;justify-content:center}
  .tk{padding:2.5px 6px;border-radius:4px;font-size:9.5px;
    background:rgba(10,14,24,.78);border:1px solid;backdrop-filter:blur(8px)}

  #ft{position:fixed;left:22px;bottom:18px;font-size:9px;letter-spacing:.2em;
    color:var(--txt-faint);z-index:10;pointer-events:none}

  .spark{width:100%;height:46px;display:block;margin-top:5px}
  .legend{display:flex;gap:10px;margin-top:6px;font-size:9px;letter-spacing:.12em;
    flex-wrap:wrap}
  .legend i{font-style:normal;display:flex;align-items:center;gap:5px;
    color:var(--txt-dim)}
  .legend u{text-decoration:none;width:9px;height:2.5px;border-radius:1px;
    display:inline-block}

  #boot{position:fixed;inset:0;z-index:40;background:var(--bg);display:flex;
    align-items:center;justify-content:center;flex-direction:column;gap:16px;
    transition:opacity .75s;font-size:10px;letter-spacing:.32em;color:var(--txt-dim)}
  #boot.gone{opacity:0;pointer-events:none}
  #bootbar{width:210px;height:1px;background:rgba(130,152,208,.18);overflow:hidden}
  #bootbar i{display:block;height:100%;width:32%;background:#6fd3f7;
    box-shadow:0 0 12px #6fd3f7;animation:sweep 1.3s cubic-bezier(.65,0,.35,1) infinite}
  @keyframes sweep{0%{transform:translateX(-110%)}100%{transform:translateX(340%)}}
  #bootmsg{color:var(--txt-faint)}
</style>
</head>
<body>
<canvas id="gl"></canvas>
<canvas id="ov"></canvas>

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
    <div id="pipeHint">Eine Reduktion über alle Sätze UND alle Layer.<br>
      Garantiert ein stabiles Koordinatensystem.</div>

    <div style="height:13px"></div>
    <div class="row">
      <div><span class="lb">Transport</span>
        <select id="emd">
          <option value="exact">EXAKT · EMD</option>
          <option value="sliced">SLICED</option>
          <option value="sinkhorn">SINKHORN</option>
        </select></div>
      <div><span class="lb">Gitter</span>
        <select id="gres">
          <option value="32">32³ · schnell</option>
          <option value="48" selected>48³ · normal</option>
          <option value="64">64³ · fein</option>
        </select></div>
    </div>

    <div style="height:11px"></div>
    <span class="lb">Hüllweite σ <em id="smLab">1.00</em></span>
    <input type="range" id="smooth" min="0.5" max="2.2" step="0.05" value="1.0">

    <div style="height:12px"></div>
    <div id="go">ANALYSIEREN</div>
    <div id="msg"></div>
  </div>

  <div class="card">
    <span class="lb">Isofläche <em id="isoLab">0.55</em></span>
    <input type="range" id="iso" min="0.12" max="1.60" step="0.01" value="0.55">
    <div style="font-size:9px;color:var(--txt-faint);margin-top:4px">
      Live · keine Neuberechnung nötig</div>
  </div>

  <div class="card">
    <span class="lb">Layer-Kennzahlen <em id="focLabel">—</em></span>
    <div class="stat"><span>DRIFT L-1→L</span><b id="s-drift">—</b></div>
    <div class="stat"><span>DREIECKE</span><b id="s-tris">—</b></div>
    <div class="stat"><span>KOMPONENTEN</span><b id="s-comp">—</b></div>
    <div class="stat"><span>EFF. DIM</span><b id="s-edim">—</b></div>
    <div class="stat"><span>ANISOTROPIE</span><b id="s-aniso">—</b></div>
    <div class="stat"><span>KRÜMMUNG</span><b id="s-curv">—</b></div>
    <div class="stat"><span>‖h‖ MITTEL</span><b id="s-norm">—</b></div>
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
  <div class="btn" id="bPts">·</div>
  <div class="btn" id="bAll">◉</div>
  <div class="btn" id="bHC">A</div>
</div>

<div id="leg"></div>
<div id="toks"></div>

<div id="ft">DRAG ROTATE · SCROLL ZOOM · ◂ ▸ LAYER · 1–8 SATZ · ⇧+N SOLO</div>

<div id="boot"><div>HIDDEN-STATE MANIFOLD · V5</div>
  <div id="bootbar"><i></i></div>
  <div id="bootmsg">BEREIT · ANALYSE STARTEN</div></div>

<script>
/* ════════════════════════════════════════════════════════════════════
   0 · WEBGL-KONTEXT
   ════════════════════════════════════════════════════════════════════ */
const glc = document.getElementById('gl');
const gl  = glc.getContext('webgl2', {antialias:true, alpha:false,
                                      depth:true, premultipliedAlpha:false})
         || glc.getContext('webgl',  {antialias:true, alpha:false, depth:true});
const GL2 = !!(gl && gl.drawBuffers);          // webgl2?
if(!gl){ document.body.innerHTML =
  '<div style="padding:40px;color:#ff8fa3;font:13px monospace">'+
  'WebGL wird von diesem Browser nicht unterstützt.</div>'; }

const ovc = document.getElementById('ov'), ox = ovc.getContext('2d');
let W,H,DPR;
function rs(){
  DPR=Math.min(devicePixelRatio||1,2); W=innerWidth; H=innerHeight;
  glc.width=W*DPR; glc.height=H*DPR;
  ovc.width=W*DPR; ovc.height=H*DPR;
  ox.setTransform(DPR,0,0,DPR,0,0);
  if(gl) gl.viewport(0,0,glc.width,glc.height);
}
addEventListener('resize',rs); rs();

const MAXS=8, GOLDEN=137.50776405003785;
const autoHue=i=>(193+i*GOLDEN)%360;
document.documentElement.style.setProperty('--a','hsl(193,76%,66%)');

/* ════════════════════════════════════════════════════════════════════
   1 · SHADER
   ════════════════════════════════════════════════════════════════════ */
const VS_SURF = `
precision highp float;
attribute vec3 aPos;
attribute vec3 aNrm;
uniform mat4 uMVP;
uniform mat3 uNormalMat;
varying vec3 vN;
varying vec3 vP;
void main(){
  vN = normalize(uNormalMat * aNrm);
  vP = aPos;
  gl_Position = uMVP * vec4(aPos,1.0);
}`;

/* Physikalisch grob plausibel, aber bewusst stilisiert:
   zwei Lichter + Fresnel-Rim + Tiefen-Dunkelung.          */
const FS_SURF = `
precision highp float;
varying vec3 vN;
varying vec3 vP;
uniform vec3 uCol;
uniform vec3 uRim;
uniform float uAlpha;
uniform vec3 uEye;
void main(){
  vec3 N = normalize(vN);
  vec3 V = normalize(uEye - vP);
  if(dot(N,V) < 0.0) N = -N;            // doppelseitig

  vec3 L1 = normalize(vec3( 0.42, 0.78, 0.55));
  vec3 L2 = normalize(vec3(-0.60, 0.15,-0.42));

  float d1 = max(dot(N,L1),0.0);
  float d2 = max(dot(N,L2),0.0) * 0.38;

  vec3 Hv  = normalize(L1 + V);
  float sp = pow(max(dot(N,Hv),0.0), 34.0) * 0.55;

  float fres = pow(1.0 - max(dot(N,V),0.0), 2.6);

  vec3 col = uCol * (0.14 + d1*0.86 + d2)
           + vec3(sp)
           + uRim * fres * 1.25;

  col = col / (col + vec3(0.85));          // Reinhard
  col = pow(col, vec3(0.4545));            // Gamma

  gl_FragColor = vec4(col, uAlpha);
}`;

const VS_PT = `
precision highp float;
attribute vec3 aPos;
attribute float aIdx;
uniform mat4 uMVP;
uniform float uSize;
varying float vI;
void main(){
  vI = aIdx;
  vec4 p = uMVP * vec4(aPos,1.0);
  gl_Position = p;
  gl_PointSize = uSize / max(p.w*0.0016, 0.35);
}`;

const FS_PT = `
precision highp float;
varying float vI;
uniform vec3 uCol;
void main(){
  vec2 d = gl_PointCoord - vec2(0.5);
  float r = length(d);
  if(r > 0.5) discard;
  float a = smoothstep(0.5, 0.18, r);
  float core = smoothstep(0.30, 0.0, r);
  vec3 c = mix(uCol, vec3(1.0), core*0.65);
  gl_FragColor = vec4(c, a);
}`;

const VS_LINE = `
precision highp float;
attribute vec3 aPos;
uniform mat4 uMVP;
void main(){ gl_Position = uMVP * vec4(aPos,1.0); }`;

const FS_LINE = `
precision highp float;
uniform vec4 uCol;
void main(){ gl_FragColor = uCol; }`;

function mkShader(src, type){
  const s=gl.createShader(type); gl.shaderSource(s,src); gl.compileShader(s);
  if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))
    console.error('shader:',gl.getShaderInfoLog(s),src);
  return s}
function mkProg(vs,fs){
  const p=gl.createProgram();
  gl.attachShader(p,mkShader(vs,gl.VERTEX_SHADER));
  gl.attachShader(p,mkShader(fs,gl.FRAGMENT_SHADER));
  gl.linkProgram(p);
  if(!gl.getProgramParameter(p,gl.LINK_STATUS))
    console.error('link:',gl.getProgramInfoLog(p));
  return p}

let PR_SURF, PR_PT, PR_LINE, LOC={};
if(gl){
  PR_SURF=mkProg(VS_SURF,FS_SURF);
  PR_PT  =mkProg(VS_PT,  FS_PT);
  PR_LINE=mkProg(VS_LINE,FS_LINE);
  LOC.surf={
    aPos:gl.getAttribLocation(PR_SURF,'aPos'),
    aNrm:gl.getAttribLocation(PR_SURF,'aNrm'),
    uMVP:gl.getUniformLocation(PR_SURF,'uMVP'),
    uNM :gl.getUniformLocation(PR_SURF,'uNormalMat'),
    uCol:gl.getUniformLocation(PR_SURF,'uCol'),
    uRim:gl.getUniformLocation(PR_SURF,'uRim'),
    uAl :gl.getUniformLocation(PR_SURF,'uAlpha'),
    uEye:gl.getUniformLocation(PR_SURF,'uEye')};
  LOC.pt={
    aPos:gl.getAttribLocation(PR_PT,'aPos'),
    aIdx:gl.getAttribLocation(PR_PT,'aIdx'),
    uMVP:gl.getUniformLocation(PR_PT,'uMVP'),
    uSize:gl.getUniformLocation(PR_PT,'uSize'),
    uCol:gl.getUniformLocation(PR_PT,'uCol')};
  LOC.line={
    aPos:gl.getAttribLocation(PR_LINE,'aPos'),
    uMVP:gl.getUniformLocation(PR_LINE,'uMVP'),
    uCol:gl.getUniformLocation(PR_LINE,'uCol')};
}

/* ════════════════════════════════════════════════════════════════════
   2 · MATRIZEN
   ════════════════════════════════════════════════════════════════════ */
function m4ident(){return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1])}
function m4mul(a,b){
  const o=new Float32Array(16);
  for(let i=0;i<4;i++)for(let j=0;j<4;j++){
    let s=0; for(let k=0;k<4;k++) s+=a[k*4+j]*b[i*4+k];
    o[i*4+j]=s}
  return o}
function m4persp(fov,asp,n,f){
  const t=1/Math.tan(fov/2);
  return new Float32Array([t/asp,0,0,0, 0,t,0,0,
    0,0,(f+n)/(n-f),-1, 0,0,2*f*n/(n-f),0])}
function m4lookAt(e,c,u){
  const z=[e[0]-c[0],e[1]-c[1],e[2]-c[2]];
  let l=Math.hypot(z[0],z[1],z[2])||1; z[0]/=l;z[1]/=l;z[2]/=l;
  const X=[u[1]*z[2]-u[2]*z[1], u[2]*z[0]-u[0]*z[2], u[0]*z[1]-u[1]*z[0]];
  l=Math.hypot(X[0],X[1],X[2])||1; X[0]/=l;X[1]/=l;X[2]/=l;
  const Y=[z[1]*X[2]-z[2]*X[1], z[2]*X[0]-z[0]*X[2], z[0]*X[1]-z[1]*X[0]];
  return new Float32Array([
    X[0],Y[0],z[0],0, X[1],Y[1],z[1],0, X[2],Y[2],z[2],0,
    -(X[0]*e[0]+X[1]*e[1]+X[2]*e[2]),
    -(Y[0]*e[0]+Y[1]*e[1]+Y[2]*e[2]),
    -(z[0]*e[0]+z[1]*e[1]+z[2]*e[2]), 1])}
function m3fromM4(m){
  return new Float32Array([m[0],m[1],m[2], m[4],m[5],m[6], m[8],m[9],m[10]])}

/* ════════════════════════════════════════════════════════════════════
   3 · MARCHING CUBES · Tabellen
   ════════════════════════════════════════════════════════════════════ */
const MC_EDGE = new Int32Array([
0x0  ,0x109,0x203,0x30a,0x406,0x50f,0x605,0x70c,0x80c,0x905,0xa0f,0xb06,0xc0a,0xd03,0xe09,0xf00,
0x190,0x99 ,0x393,0x29a,0x596,0x49f,0x795,0x69c,0x99c,0x895,0xb9f,0xa96,0xd9a,0xc93,0xf99,0xe90,
0x230,0x339,0x33 ,0x13a,0x636,0x73f,0x435,0x53c,0xa3c,0xb35,0x83f,0x936,0xe3a,0xf33,0xc39,0xd30,
0x3a0,0x2a9,0x1a3,0xaa ,0x7a6,0x6af,0x5a5,0x4ac,0xbac,0xaa5,0x9af,0x8a6,0xfaa,0xea3,0xda9,0xca0,
0x460,0x569,0x663,0x76a,0x66 ,0x16f,0x265,0x36c,0xc6c,0xd65,0xe6f,0xf66,0x86a,0x963,0xa69,0xb60,
0x5f0,0x4f9,0x7f3,0x6fa,0x1f6,0xff ,0x3f5,0x2fc,0xdfc,0xcf5,0xfff,0xef6,0x9fa,0x8f3,0xbf9,0xaf0,
0x650,0x759,0x453,0x55a,0x256,0x35f,0x55 ,0x15c,0xe5c,0xf55,0xc5f,0xd56,0xa5a,0xb53,0x859,0x950,
0x7c0,0x6c9,0x5c3,0x4ca,0x3c6,0x2cf,0x1c5,0xcc ,0xfcc,0xec5,0xdcf,0xcc6,0xbca,0xac3,0x9c9,0x8c0,
0x8c0,0x9c9,0xac3,0xbca,0xcc6,0xdcf,0xec5,0xfcc,0xcc ,0x1c5,0x2cf,0x3c6,0x4ca,0x5c3,0x6c9,0x7c0,
0x950,0x859,0xb53,0xa5a,0xd56,0xc5f,0xf55,0xe5c,0x15c,0x55 ,0x35f,0x256,0x55a,0x453,0x759,0x650,
0xaf0,0xbf9,0x8f3,0x9fa,0xef6,0xfff,0xcf5,0xdfc,0x2fc,0x3f5,0xff ,0x1f6,0x6fa,0x7f3,0x4f9,0x5f0,
0xb60,0xa69,0x963,0x86a,0xf66,0xe6f,0xd65,0xc6c,0x36c,0x265,0x16f,0x66 ,0x76a,0x663,0x569,0x460,
0xca0,0xda9,0xea3,0xfaa,0x8a6,0x9af,0xaa5,0xbac,0x4ac,0x5a5,0x6af,0x7a6,0xaa ,0x1a3,0x2a9,0x3a0,
0xd30,0xc39,0xf33,0xe3a,0x936,0x83f,0xb35,0xa3c,0x53c,0x435,0x73f,0x636,0x13a,0x33 ,0x339,0x230,
0xe90,0xf99,0xc93,0xd9a,0xa96,0xb9f,0x895,0x99c,0x69c,0x795,0x49f,0x596,0x29a,0x393,0x99 ,0x190,
0xf00,0xe09,0xd03,0xc0a,0xb06,0xa0f,0x905,0x80c,0x70c,0x605,0x50f,0x406,0x30a,0x203,0x109,0x0  ]);

/* 256 × bis zu 16 Einträge, -1 terminiert. Als kompakter String
   kodiert (Basis 17: 'a'..'p' = 0..15, 'z' = -1), spart ~8 kB Quelltext. */
const MC_TRI_SRC =
"z|aij z|abj z|bijb z|bkc z|aijbkc z|ackj z|jckija z|cld z|cldaij z|abjcld z|"+
"bijbcld z|dlbkl z|aijdlbk z|abjdlbk z|ijjdlbk z|emk z|emkaij z|emkabj z|"+
"emkbijb z|emkbkc z|emkaijbkc z|emkackj z|emkjckija z|emkcld z|emkcldaij z|"+
"emkabjcld z|emkbijbcld z|emkdlbkl z|emkaijdlbk z|emkabjdlbk z|emkijjdlbk z";
/* -- siehe unten: die vollständige Tabelle wird numerisch erzeugt. */

/* Die klassische 256×16-Tri-Tabelle. Vollständig, als flaches Int8Array. */
const MC_TRI = (function(){
  const s =
  "-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "0,1,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "1,8,3,9,8,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "0,8,3,1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "9,2,10,0,2,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "2,8,3,2,10,8,10,9,8,-1,-1,-1,-1,-1,-1,-1,"+
  "3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "0,11,2,8,11,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "1,9,0,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "1,11,2,1,9,11,9,8,11,-1,-1,-1,-1,-1,-1,-1,"+
  "3,10,1,11,10,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "0,10,1,0,8,10,8,11,10,-1,-1,-1,-1,-1,-1,-1,"+
  "3,9,0,3,11,9,11,10,9,-1,-1,-1,-1,-1,-1,-1,"+
  "9,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "4,3,0,7,3,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "0,1,9,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "4,1,9,4,7,1,7,3,1,-1,-1,-1,-1,-1,-1,-1,"+
  "1,2,10,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "3,4,7,3,0,4,1,2,10,-1,-1,-1,-1,-1,-1,-1,"+
  "9,2,10,9,0,2,8,4,7,-1,-1,-1,-1,-1,-1,-1,"+
  "2,10,9,2,9,7,2,7,3,7,9,4,-1,-1,-1,-1,"+
  "8,4,7,3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "11,4,7,11,2,4,2,0,4,-1,-1,-1,-1,-1,-1,-1,"+
  "9,0,1,8,4,7,2,3,11,-1,-1,-1,-1,-1,-1,-1,"+
  "4,7,11,9,4,11,9,11,2,9,2,1,-1,-1,-1,-1,"+
  "3,10,1,3,11,10,7,8,4,-1,-1,-1,-1,-1,-1,-1,"+
  "1,11,10,1,4,11,1,0,4,7,11,4,-1,-1,-1,-1,"+
  "4,7,8,9,0,11,9,11,10,11,0,3,-1,-1,-1,-1,"+
  "4,7,11,4,11,9,9,11,10,-1,-1,-1,-1,-1,-1,-1,"+
  "9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "9,5,4,0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "0,5,4,1,5,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "8,5,4,8,3,5,3,1,5,-1,-1,-1,-1,-1,-1,-1,"+
  "1,2,10,9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "3,0,8,1,2,10,4,9,5,-1,-1,-1,-1,-1,-1,-1,"+
  "5,2,10,5,4,2,4,0,2,-1,-1,-1,-1,-1,-1,-1,"+
  "2,10,5,3,2,5,3,5,4,3,4,8,-1,-1,-1,-1,"+
  "9,5,4,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "0,11,2,0,8,11,4,9,5,-1,-1,-1,-1,-1,-1,-1,"+
  "0,5,4,0,1,5,2,3,11,-1,-1,-1,-1,-1,-1,-1,"+
  "2,1,5,2,5,8,2,8,11,4,8,5,-1,-1,-1,-1,"+
  "10,3,11,10,1,3,9,5,4,-1,-1,-1,-1,-1,-1,-1,"+
  "4,9,5,0,8,1,8,10,1,8,11,10,-1,-1,-1,-1,"+  "5,4,0,5,0,11,5,11,10,11,0,3,-1,-1,-1,-1,"+
  "5,4,8,5,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1,"+
  "9,7,8,5,7,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "9,3,0,9,5,3,5,7,3,-1,-1,-1,-1,-1,-1,-1,"+
  "0,7,8,0,1,7,1,5,7,-1,-1,-1,-1,-1,-1,-1,"+
  "1,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "9,7,8,9,5,7,10,1,2,-1,-1,-1,-1,-1,-1,-1,"+
  "10,1,2,9,5,0,5,3,0,5,7,3,-1,-1,-1,-1,"+
  "8,0,2,8,2,5,8,5,7,10,5,2,-1,-1,-1,-1,"+
  "2,10,5,2,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1,"+
  "7,9,5,7,8,9,3,11,2,-1,-1,-1,-1,-1,-1,-1,"+
  "9,5,7,9,7,2,9,2,0,2,7,11,-1,-1,-1,-1,"+
  "2,3,11,0,1,8,1,7,8,1,5,7,-1,-1,-1,-1,"+
  "11,2,1,11,1,7,7,1,5,-1,-1,-1,-1,-1,-1,-1,"+
  "9,5,8,8,5,7,10,1,3,10,3,11,-1,-1,-1,-1,"+
  "5,7,0,5,0,9,7,11,0,1,0,10,11,10,0,-1,"+
  "11,10,0,11,0,3,10,5,0,8,0,7,5,7,0,-1,"+
  "11,10,5,7,11,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "0,8,3,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "9,0,1,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "1,8,3,1,9,8,5,10,6,-1,-1,-1,-1,-1,-1,-1,"+
  "1,6,5,2,6,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "1,6,5,1,2,6,3,0,8,-1,-1,-1,-1,-1,-1,-1,"+
  "9,6,5,9,0,6,0,2,6,-1,-1,-1,-1,-1,-1,-1,"+
  "5,9,8,5,8,2,5,2,6,3,2,8,-1,-1,-1,-1,"+
  "2,3,11,10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "11,0,8,11,2,0,10,6,5,-1,-1,-1,-1,-1,-1,-1,"+
  "0,1,9,2,3,11,5,10,6,-1,-1,-1,-1,-1,-1,-1,"+
  "5,10,6,1,9,2,9,11,2,9,8,11,-1,-1,-1,-1,"+
  "6,3,11,6,5,3,5,1,3,-1,-1,-1,-1,-1,-1,-1,"+
  "0,8,11,0,11,5,0,5,1,5,11,6,-1,-1,-1,-1,"+
  "3,11,6,0,3,6,0,6,5,0,5,9,-1,-1,-1,-1,"+
  "6,5,9,6,9,11,11,9,8,-1,-1,-1,-1,-1,-1,-1,"+
  "5,10,6,4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "4,3,0,4,7,3,6,5,10,-1,-1,-1,-1,-1,-1,-1,"+
  "1,9,0,5,10,6,8,4,7,-1,-1,-1,-1,-1,-1,-1,"+
  "10,6,5,1,9,7,1,7,3,7,9,4,-1,-1,-1,-1,"+
  "6,1,2,6,5,1,4,7,8,-1,-1,-1,-1,-1,-1,-1,"+
  "1,2,5,5,2,6,3,0,4,3,4,7,-1,-1,-1,-1,"+
  "8,4,7,9,0,5,0,6,5,0,2,6,-1,-1,-1,-1,"+
  "7,3,9,7,9,4,3,2,9,5,9,6,2,6,9,-1,"+
  "3,11,2,7,8,4,10,6,5,-1,-1,-1,-1,-1,-1,-1,"+
  "5,10,6,4,7,2,4,2,0,2,7,11,-1,-1,-1,-1,"+
  "0,1,9,4,7,8,2,3,11,5,10,6,-1,-1,-1,-1,"+
  "9,2,1,9,11,2,9,4,11,7,11,4,5,10,6,-1,"+
  "8,4,7,3,11,5,3,5,1,5,11,6,-1,-1,-1,-1,"+
  "5,1,11,5,11,6,1,0,11,7,11,4,0,4,11,-1,"+
  "0,5,9,0,6,5,0,3,6,11,6,3,8,4,7,-1,"+
  "6,5,9,6,9,11,4,7,9,7,11,9,-1,-1,-1,-1,"+
  "10,4,9,6,4,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "4,10,6,4,9,10,0,8,3,-1,-1,-1,-1,-1,-1,-1,"+
  "10,0,1,10,6,0,6,4,0,-1,-1,-1,-1,-1,-1,-1,"+
  "8,3,1,8,1,6,8,6,4,6,1,10,-1,-1,-1,-1,"+
  "1,4,9,1,2,4,2,6,4,-1,-1,-1,-1,-1,-1,-1,"+
  "3,0,8,1,2,9,2,4,9,2,6,4,-1,-1,-1,-1,"+
  "0,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "8,3,2,8,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1,"+
  "10,4,9,10,6,4,11,2,3,-1,-1,-1,-1,-1,-1,-1,"+
  "0,8,2,2,8,11,4,9,10,4,10,6,-1,-1,-1,-1,"+
  "3,11,2,0,1,6,0,6,4,6,1,10,-1,-1,-1,-1,"+
  "6,4,1,6,1,10,4,8,1,2,1,11,8,11,1,-1,"+
  "9,6,4,9,3,6,9,1,3,11,6,3,-1,-1,-1,-1,"+
  "8,11,1,8,1,0,11,6,1,9,1,4,6,4,1,-1,"+
  "3,11,6,3,6,0,0,6,4,-1,-1,-1,-1,-1,-1,-1,"+
  "6,4,8,11,6,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "7,10,6,7,8,10,8,9,10,-1,-1,-1,-1,-1,-1,-1,"+
  "0,7,3,0,10,7,0,9,10,6,7,10,-1,-1,-1,-1,"+
  "10,6,7,1,10,7,1,7,8,1,8,0,-1,-1,-1,-1,"+
  "10,6,7,10,7,1,1,7,3,-1,-1,-1,-1,-1,-1,-1,"+
  "1,2,6,1,6,8,1,8,9,8,6,7,-1,-1,-1,-1,"+
  "2,6,9,2,9,1,6,7,9,0,9,3,7,3,9,-1,"+
  "7,8,0,7,0,6,6,0,2,-1,-1,-1,-1,-1,-1,-1,"+
  "7,3,2,6,7,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "2,3,11,10,6,8,10,8,9,8,6,7,-1,-1,-1,-1,"+
  "2,0,7,2,7,11,0,9,7,6,7,10,9,10,7,-1,"+
  "1,8,0,1,7,8,1,10,7,6,7,10,2,3,11,-1,"+
  "11,2,1,11,1,7,10,6,1,6,7,1,-1,-1,-1,-1,"+
  "8,9,6,8,6,7,9,1,6,11,6,3,1,3,6,-1,"+
  "0,9,1,11,6,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "7,8,0,7,0,6,3,11,0,11,6,0,-1,-1,-1,-1,"+
  "7,11,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "3,0,8,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "0,1,9,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "8,1,9,8,3,1,11,7,6,-1,-1,-1,-1,-1,-1,-1,"+
  "10,1,2,6,11,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "1,2,10,3,0,8,6,11,7,-1,-1,-1,-1,-1,-1,-1,"+
  "2,9,0,2,10,9,6,11,7,-1,-1,-1,-1,-1,-1,-1,"+
  "6,11,7,2,10,3,10,8,3,10,9,8,-1,-1,-1,-1,"+
  "7,2,3,6,2,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "7,0,8,7,6,0,6,2,0,-1,-1,-1,-1,-1,-1,-1,"+
  "2,7,6,2,3,7,0,1,9,-1,-1,-1,-1,-1,-1,-1,"+
  "1,6,2,1,8,6,1,9,8,8,7,6,-1,-1,-1,-1,"+
  "10,7,6,10,1,7,1,3,7,-1,-1,-1,-1,-1,-1,-1,"+
  "10,7,6,1,7,10,1,8,7,1,0,8,-1,-1,-1,-1,"+
  "0,3,7,0,7,10,0,10,9,6,10,7,-1,-1,-1,-1,"+
  "7,6,10,7,10,8,8,10,9,-1,-1,-1,-1,-1,-1,-1,"+
  "6,8,4,11,8,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "3,6,11,3,0,6,0,4,6,-1,-1,-1,-1,-1,-1,-1,"+
  "8,6,11,8,4,6,9,0,1,-1,-1,-1,-1,-1,-1,-1,"+
  "9,4,6,9,6,3,9,3,1,11,3,6,-1,-1,-1,-1,"+
  "6,8,4,6,11,8,2,10,1,-1,-1,-1,-1,-1,-1,-1,"+
  "1,2,10,3,0,11,0,6,11,0,4,6,-1,-1,-1,-1,"+
  "4,11,8,4,6,11,0,2,9,2,10,9,-1,-1,-1,-1,"+
  "10,9,3,10,3,2,9,4,3,11,3,6,4,6,3,-1,"+
  "8,2,3,8,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1,"+
  "0,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "1,9,0,2,3,4,2,4,6,4,3,8,-1,-1,-1,-1,"+
  "1,9,4,1,4,2,2,4,6,-1,-1,-1,-1,-1,-1,-1,"+
  "8,1,3,8,6,1,8,4,6,6,10,1,-1,-1,-1,-1,"+
  "10,1,0,10,0,6,6,0,4,-1,-1,-1,-1,-1,-1,-1,"+
  "4,6,3,4,3,8,6,10,3,0,3,9,10,9,3,-1,"+
  "10,9,4,6,10,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "4,9,5,7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "0,8,3,4,9,5,11,7,6,-1,-1,-1,-1,-1,-1,-1,"+
  "5,0,1,5,4,0,7,6,11,-1,-1,-1,-1,-1,-1,-1,"+
  "11,7,6,8,3,4,3,5,4,3,1,5,-1,-1,-1,-1,"+
  "9,5,4,10,1,2,7,6,11,-1,-1,-1,-1,-1,-1,-1,"+
  "6,11,7,1,2,10,0,8,3,4,9,5,-1,-1,-1,-1,"+
  "7,6,11,5,4,10,4,2,10,4,0,2,-1,-1,-1,-1,"+
  "3,4,8,3,5,4,3,2,5,10,5,2,11,7,6,-1,"+
  "7,2,3,7,6,2,5,4,9,-1,-1,-1,-1,-1,-1,-1,"+
  "9,5,4,0,8,6,0,6,2,6,8,7,-1,-1,-1,-1,"+
  "3,6,2,3,7,6,1,5,0,5,4,0,-1,-1,-1,-1,"+  "6,2,3,6,3,9,6,9,5,0,9,3,-1,-1,-1,-1,"+
  "9,5,6,9,6,0,0,6,2,-1,-1,-1,-1,-1,-1,-1,"+
  "1,5,6,2,1,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "1,5,6,1,6,2,3,0,8,-1,-1,-1,-1,-1,-1,-1,"+
  "0,1,9,2,3,11,5,10,7,10,6,7,-1,-1,-1,-1,"+
  "6,1,5,6,2,1,8,3,0,-1,-1,-1,-1,-1,-1,-1,"+
  "0,3,8,5,6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "10,5,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "11,5,10,7,5,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "11,5,10,11,7,5,8,3,0,-1,-1,-1,-1,-1,-1,-1,"+
  "5,11,7,5,10,11,1,9,0,-1,-1,-1,-1,-1,-1,-1,"+
  "10,7,5,10,11,7,9,8,1,8,3,1,-1,-1,-1,-1,"+
  "11,1,2,11,7,1,7,5,1,-1,-1,-1,-1,-1,-1,-1,"+
  "0,8,3,1,2,7,1,7,5,7,2,11,-1,-1,-1,-1,"+
  "9,7,5,9,2,7,9,0,2,2,11,7,-1,-1,-1,-1,"+
  "7,5,2,7,2,11,5,9,2,3,2,8,9,8,2,-1,"+
  "2,5,10,2,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1,"+
  "8,2,0,8,5,2,8,7,5,10,2,5,-1,-1,-1,-1,"+
  "9,0,1,5,10,3,5,3,7,3,10,2,-1,-1,-1,-1,"+
  "9,8,2,9,2,1,8,7,2,10,2,5,7,5,2,-1,"+
  "1,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "0,8,7,0,7,1,1,7,5,-1,-1,-1,-1,-1,-1,-1,"+
  "9,0,3,9,3,5,5,3,7,-1,-1,-1,-1,-1,-1,-1,"+
  "9,8,7,5,9,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "5,8,4,5,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1,"+
  "5,0,4,5,11,0,5,10,11,11,3,0,-1,-1,-1,-1,"+
  "0,1,9,8,4,10,8,10,11,10,4,5,-1,-1,-1,-1,"+
  "10,11,4,10,4,5,11,3,4,9,4,1,3,1,4,-1,"+
  "2,5,1,2,8,5,2,11,8,4,5,8,-1,-1,-1,-1,"+
  "0,4,11,0,11,3,4,5,11,2,11,1,5,1,11,-1,"+
  "0,2,5,0,5,9,2,11,5,4,5,8,11,8,5,-1,"+
  "9,4,5,2,11,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "2,5,10,3,5,2,3,4,5,3,8,4,-1,-1,-1,-1,"+
  "5,10,2,5,2,4,4,2,0,-1,-1,-1,-1,-1,-1,-1,"+
  "3,10,2,3,5,10,3,8,5,4,5,8,0,1,9,-1,"+
  "5,10,2,5,2,4,1,9,2,9,4,2,-1,-1,-1,-1,"+
  "8,4,5,8,5,3,3,5,1,-1,-1,-1,-1,-1,-1,-1,"+
  "0,4,5,1,0,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "8,4,5,8,5,3,9,0,5,0,3,5,-1,-1,-1,-1,"+
  "9,4,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "4,11,7,4,9,11,9,10,11,-1,-1,-1,-1,-1,-1,-1,"+
  "0,8,3,4,9,7,9,11,7,9,10,11,-1,-1,-1,-1,"+
  "1,10,11,1,11,4,1,4,0,7,4,11,-1,-1,-1,-1,"+
  "3,1,4,3,4,8,1,10,4,7,4,11,10,11,4,-1,"+
  "4,11,7,9,11,4,9,2,11,9,1,2,-1,-1,-1,-1,"+
  "9,7,4,9,11,7,9,1,11,2,11,1,0,8,3,-1,"+
  "11,7,4,11,4,2,2,4,0,-1,-1,-1,-1,-1,-1,-1,"+
  "11,7,4,11,4,2,8,3,4,3,2,4,-1,-1,-1,-1,"+
  "2,9,10,2,7,9,2,3,7,7,4,9,-1,-1,-1,-1,"+
  "9,10,7,9,7,4,10,2,7,8,7,0,2,0,7,-1,"+
  "3,7,10,3,10,2,7,4,10,1,10,0,4,0,10,-1,"+
  "1,10,2,8,7,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "4,9,1,4,1,7,7,1,3,-1,-1,-1,-1,-1,-1,-1,"+
  "4,9,1,4,1,7,0,8,1,8,7,1,-1,-1,-1,-1,"+
  "4,0,3,7,4,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "4,8,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "9,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "3,0,9,3,9,11,11,9,10,-1,-1,-1,-1,-1,-1,-1,"+
  "0,1,10,0,10,8,8,10,11,-1,-1,-1,-1,-1,-1,-1,"+
  "3,1,10,11,3,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "1,2,11,1,11,9,9,11,8,-1,-1,-1,-1,-1,-1,-1,"+
  "3,0,9,3,9,11,1,2,9,2,11,9,-1,-1,-1,-1,"+
  "0,2,11,8,0,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "3,2,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "2,3,8,2,8,10,10,8,9,-1,-1,-1,-1,-1,-1,-1,"+
  "9,10,2,0,9,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "2,3,8,2,8,10,0,1,8,1,10,8,-1,-1,-1,-1,"+
  "1,10,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "1,3,8,9,1,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "0,9,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "0,3,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"+
  "-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1";
  const a = s.split(','), out = new Int8Array(a.length);
  for(let i=0;i<a.length;i++) out[i] = parseInt(a[i],10);
  return out;
})();

/* Kanten-Endpunkte des Einheitswürfels (Standard-MC-Nummerierung) */
const MC_CORNER = [
  [0,0,0],[1,0,0],[1,1,0],[0,1,0],
  [0,0,1],[1,0,1],[1,1,1],[0,1,1]];
const MC_EDGE_V = [
  [0,1],[1,2],[2,3],[3,0],
  [4,5],[5,6],[6,7],[7,4],
  [0,4],[1,5],[2,6],[3,7]];

/* ════════════════════════════════════════════════════════════════════
   4 · MARCHING CUBES · Extraktion
   ════════════════════════════════════════════════════════════════════
   Erwartet ein Float32Array der Groesse res³ in x-major-Reihenfolge
   (Index = (ix*res + iy)*res + iz, exakt wie numpy es schreibt).
   Liefert interleaved Positionen + gemittelte Normalen.             */

const MC = (function(){
  // wiederverwendete Puffer, damit pro Frame nichts alloziert wird
  let capV = 0, posBuf = null, nrmBuf = null;
  let edgeCache = null, cacheRes = 0;

  function ensure(n){
    if(n <= capV) return;
    capV = Math.max(n, capV ? capV*2 : 65536);
    posBuf = new Float32Array(capV*3);
    nrmBuf = new Float32Array(capV*3);
  }

  function build(field, res, iso, lo, hi){
    const r = res, r2 = r*r;
    const sx = (hi[0]-lo[0])/(r-1),
          sy = (hi[1]-lo[1])/(r-1),
          sz = (hi[2]-lo[2])/(r-1);

    // --- Zentrale Differenzen fuer die Gradienten-Normale ----------
    const at = (i,j,k)=>field[i*r2 + j*r + k];
    const grad = (i,j,k,out)=>{
      const ip = i<r-1?i+1:i, im = i>0?i-1:i;
      const jp = j<r-1?j+1:j, jm = j>0?j-1:j;
      const kp = k<r-1?k+1:k, km = k>0?k-1:k;
      out[0] = (at(im,j,k) - at(ip,j,k)) / ((ip-im)*sx || 1);
      out[1] = (at(i,jm,k) - at(i,jp,k)) / ((jp-jm)*sy || 1);
      out[2] = (at(i,km,j===j?k:k) !== undefined
                ? at(i,j,km) - at(i,j,kp) : 0) / ((kp-km)*sz || 1);
    };

    let nv = 0;
    ensure(1 << 16);

    const cv = new Float32Array(8);          // Eckwerte
    const gA = new Float32Array(3), gB = new Float32Array(3);
    const vp = new Float32Array(36), vn = new Float32Array(36);

    for(let i=0;i<r-1;i++){
      for(let j=0;j<r-1;j++){
        for(let k=0;k<r-1;k++){
          let cube = 0;
          for(let c=0;c<8;c++){
            const C = MC_CORNER[c];
            const v = field[(i+C[0])*r2 + (j+C[1])*r + (k+C[2])];
            cv[c] = v;
            if(v < iso) cube |= (1<<c);
          }
          if(cube === 0 || cube === 255) continue;

          const em = MC_EDGE[cube];
          if(em === 0) continue;

          // --- Schnittpunkte auf den 12 Kanten ---------------------
          for(let e=0;e<12;e++){
            if(!(em & (1<<e))) continue;
            const [a,b] = MC_EDGE_V[e];
            const va = cv[a], vb = cv[b];
            let t = (iso - va) / ((vb - va) || 1e-9);
            if(t < 0) t = 0; else if(t > 1) t = 1;

            const CA = MC_CORNER[a], CB = MC_CORNER[b];
            const ia = i+CA[0], ja = j+CA[1], ka = k+CA[2];
            const ib = i+CB[0], jb = j+CB[1], kb = k+CB[2];

            vp[e*3  ] = lo[0] + (ia + (ib-ia)*t) * sx;
            vp[e*3+1] = lo[1] + (ja + (jb-ja)*t) * sy;
            vp[e*3+2] = lo[2] + (ka + (kb-ka)*t) * sz;

            gradAt(field, r, r2, ia, ja, ka, sx, sy, sz, gA);
            gradAt(field, r, r2, ib, jb, kb, sx, sy, sz, gB);
            let nx = gA[0] + (gB[0]-gA[0])*t;
            let ny = gA[1] + (gB[1]-gA[1])*t;
            let nz = gA[2] + (gB[2]-gA[2])*t;
            const L = Math.hypot(nx,ny,nz) || 1;
            vn[e*3  ] = nx/L; vn[e*3+1] = ny/L; vn[e*3+2] = nz/L;
          }

          // --- Dreiecke emittieren ---------------------------------
          const base = cube*16;
          for(let t3=0; t3<16; t3+=3){
            const e0 = MC_TRI[base+t3];
            if(e0 === -1) break;
            const e1 = MC_TRI[base+t3+1], e2 = MC_TRI[base+t3+2];
            ensure(nv+3);
            for(const e of [e0,e1,e2]){
              posBuf[nv*3  ] = vp[e*3  ];
              posBuf[nv*3+1] = vp[e*3+1];
              posBuf[nv*3+2] = vp[e*3+2];
              nrmBuf[nv*3  ] = vn[e*3  ];
              nrmBuf[nv*3+1] = vn[e*3+1];
              nrmBuf[nv*3+2] = vn[e*3+2];
              nv++;
            }
          }
        }
      }
    }
    return {pos:posBuf, nrm:nrmBuf, count:nv};
  }

  function gradAt(f, r, r2, i, j, k, sx, sy, sz, out){
    const ip = i<r-1?i+1:i, im = i>0?i-1:i;
    const jp = j<r-1?j+1:j, jm = j>0?j-1:j;
    const kp = k<r-1?k+1:k, km = k>0?k-1:k;
    out[0] = (f[im*r2+j*r+k] - f[ip*r2+j*r+k]) / (((ip-im)||1)*sx);
    out[1] = (f[i*r2+jm*r+k] - f[i*r2+jp*r+k]) / (((jp-jm)||1)*sy);
    out[2] = (f[i*r2+j*r+km] - f[i*r2+j*r+kp]) / (((kp-km)||1)*sz);
  }

  return {build};
})();

/* ════════════════════════════════════════════════════════════════════
   5 · GRID-DEKODIERUNG  (base64 → zlib → uint8 → float32)
   ════════════════════════════════════════════════════════════════════
   Die Inflate-Routine ist eine kompakte, vollstaendige RFC-1950/1951-
   Implementierung. Kein CDN, keine Abhaengigkeit.                   */

function b64ToU8(b64){
  const bin = atob(b64), n = bin.length, out = new Uint8Array(n);
  for(let i=0;i<n;i++) out[i] = bin.charCodeAt(i);
  return out;
}

const inflate = (function(){
  const LBASE=[3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258];
  const LEXT =[0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0];
  const DBASE=[1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577];
  const DEXT =[0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13];
  const CLORD=[16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15];

  function buildTree(lens){
    const maxBits = Math.max(...lens);
    const blCount = new Int32Array(maxBits+1);
    for(const l of lens) if(l) blCount[l]++;
    const nextCode = new Int32Array(maxBits+2);
    let code = 0;
    for(let b=1;b<=maxBits;b++){ code = (code + blCount[b-1]) << 1; nextCode[b] = code; }
    const codes = new Int32Array(lens.length);
    for(let i=0;i<lens.length;i++) if(lens[i]) codes[i] = nextCode[lens[i]]++;
    return {lens, codes, maxBits};
  }

  function decodeSym(br, tree){
    let code = 0, len = 0;
    while(len < tree.maxBits){
      code = (code<<1) | br.bit();
      len++;
      for(let i=0;i<tree.lens.length;i++)
        if(tree.lens[i] === len && tree.codes[i] === code) return i;
    }
    throw new Error('inflate: ungueltiges Symbol');
  }

  // Schnellere Variante: Lookup pro Laenge
  function makeFast(tree){
    const first = new Int32Array(tree.maxBits+2);
    const idx   = new Int32Array(tree.maxBits+2);
    const sorted= [];
    for(let b=1;b<=tree.maxBits;b++){
      first[b] = sorted.length;
      for(let i=0;i<tree.lens.length;i++) if(tree.lens[i]===b) sorted.push(i);
      idx[b] = sorted.length;
    }
    const firstCode = new Int32Array(tree.maxBits+2);
    let c = 0;
    for(let b=1;b<=tree.maxBits;b++){
      c = (c + (b>1 ? (idx[b-1]-first[b-1]) : 0)) << 1;
      firstCode[b] = c;
    }
    return {sorted, first, idx, firstCode, maxBits:tree.maxBits};
  }

  function fastSym(br, ft){
    let code = 0;
    for(let b=1;b<=ft.maxBits;b++){
      code = (code<<1) | br.bit();
      const n = ft.idx[b] - ft.first[b];
      if(n > 0){
        const off = code - ft.firstCode[b];
        if(off >= 0 && off < n) return ft.sorted[ft.first[b] + off];
      }
    }
    throw new Error('inflate: Symbol nicht gefunden');
  }

  function reader(buf, start){
    let p = start, b = 0, nb = 0;
    return {
      bit(){ if(!nb){ b = buf[p++]; nb = 8; } const v = b & 1; b >>= 1; nb--; return v; },
      bits(n){ let v=0; for(let i=0;i<n;i++) v |= this.bit()<<i; return v; },
      align(){ nb = 0; },
      byte(){ return buf[p++]; },
      pos(){ return p; }
    };
  }

  return function(src){
    // zlib-Header ueberspringen
    let off = 0;
    if((src[0] & 0x0f) === 8) off = 2;
    const br = reader(src, off);
    const out = [];
    let outBuf = new Uint8Array(1<<20), olen = 0;
    const push = v => {
      if(olen >= outBuf.length){
        const nxt = new Uint8Array(outBuf.length*2);
        nxt.set(outBuf); outBuf = nxt;
      }
      outBuf[olen++] = v;
    };

    let fixedLit = null, fixedDist = null;

    for(;;){
      const final = br.bit();
      const type  = br.bits(2);

      if(type === 0){
        br.align();
        const len = br.byte() | (br.byte()<<8);
        br.byte(); br.byte();
        for(let i=0;i<len;i++) push(br.byte());
      } else {
        let lt, dt;
        if(type === 1){
          if(!fixedLit){
            const ll = new Int32Array(288);
            for(let i=0;i<144;i++) ll[i]=8;
            for(let i=144;i<256;i++) ll[i]=9;
            for(let i=256;i<280;i++) ll[i]=7;
            for(let i=280;i<288;i++) ll[i]=8;
            const dl = new Int32Array(30).fill(5);
            fixedLit  = makeFast(buildTree(ll));
            fixedDist = makeFast(buildTree(dl));
          }
          lt = fixedLit; dt = fixedDist;
        } else if(type === 2){
          const hlit  = br.bits(5) + 257;
          const hdist = br.bits(5) + 1;
          const hclen = br.bits(4) + 4;
          const clLens = new Int32Array(19);
          for(let i=0;i<hclen;i++) clLens[CLORD[i]] = br.bits(3);
          const clT = makeFast(buildTree(clLens));
          const all = new Int32Array(hlit + hdist);
          let i = 0;
          while(i < all.length){
            const s = fastSym(br, clT);
            if(s < 16){ all[i++] = s; }
            else if(s === 16){ const r = br.bits(2)+3, pv = all[i-1];
              for(let q=0;q<r;q++) all[i++] = pv; }
            else if(s === 17){ const r = br.bits(3)+3; i += r; }
            else { const r = br.bits(7)+11; i += r; }
          }
          const litLens  = all.slice(0, hlit);
          const distLens = all.slice(hlit);
          lt = makeFast(buildTree(litLens));
          dt = makeFast(buildTree(distLens));
        } else {
          throw new Error('inflate: reservierter Blocktyp');
        }

        for(;;){
          const s = fastSym(br, lt);
          if(s === 256) break;
          if(s < 256){ push(s); continue; }
          const li = s - 257;
          const len = LBASE[li] + br.bits(LEXT[li]);
          const ds  = fastSym(br, dt);
          const dis = DBASE[ds] + br.bits(DEXT[ds]);
          const start = olen - dis;
          for(let q=0;q<len;q++) push(outBuf[start+q]);
        }
      }
      if(final) break;
    }
    return outBuf.subarray(0, olen);
  };
})();

/* uint8-quantisiertes Feld -> Float32Array, auf [0,1] renormiert */
function decodeGrids(b64, nLayers, res, gmax){
  const raw = inflate(b64ToU8(b64));
  const per = res*res*res;
  const need = nLayers*per;
  if(raw.length < need)
    throw new Error(`Grid zu kurz: ${raw.length} < ${need}`);
  const out = [];
  for(let L=0;L<nLayers;L++){
    const f = new Float32Array(per);
    const off = L*per;
    for(let i=0;i<per;i++) f[i] = raw[off+i] / 255.0;
    out.push(f);
  }
  return out;
}

/* ════════════════════════════════════════════════════════════════════
   6 · STATE
   ════════════════════════════════════════════════════════════════════ */
let D = null;
let LAYERS = 1, RES = 48;
let LO = [-1.3,-1.3,-1.3], HI = [1.3,1.3,1.3];
let pos = 0, playing = false, scrub = null, targetPos = null;
let surf = true, showPts = false;
let VIS = [], SOLO = -1, FOCUS = 0;
let ISO = 0.55;
const STEP_SPD = 5.6, AUTO_SPD = 0.34;

let yaw = -0.68, pit = 0.26, dist = 3.35;
let tg = {y:-0.68, p:0.26}, dg = null;

const visible = i => SOLO>=0 ? i===SOLO : !!VIS[i];

/* Pro Serie: dekodierte Grids + GPU-Puffer + gemischtes Arbeitsfeld */
let SER = [];     // {grids:[Float32Array], blend:Float32Array, vbo, nbo, count, pbo, pcount}

let ROWS = __DEFAULTS__.map((t,i)=>({text:t,label:'S'+(i+1),hue:autoHue(i),on:true}));

/* ════════════════════════════════════════════════════════════════════
   7 · INTERAKTION
   ════════════════════════════════════════════════════════════════════ */
glc.onpointerdown = e => dg = {x:e.clientX, y:e.clientY, a:tg.y, b:tg.p};
addEventListener('pointerup', ()=>{ dg=null; scrub=null; });
addEventListener('pointermove', e=>{
  if(scrub !== null){
    const r = rail.getBoundingClientRect();
    const u = 1 - Math.max(0, Math.min(1, (e.clientY-r.top)/r.height));
    pos = u*(LAYERS-1); targetPos = null; return;
  }
  if(!dg) return;
  tg.y = dg.a + (e.clientX-dg.x)*0.0055;
  tg.p = Math.max(-1.35, Math.min(1.35, dg.b + (e.clientY-dg.y)*0.0042));
});
addEventListener('wheel', e=>{
  dist = Math.max(1.45, Math.min(9.0, dist + e.deltaY*0.0016));
}, {passive:1});

/* ════════════════════════════════════════════════════════════════════
   8 · GEOMETRIE PRO FRAME
   ════════════════════════════════════════════════════════════════════
   Hier passiert das eigentliche Morphing: Feld(L) und Feld(L+1) werden
   mit smoothstep gemischt, DANN laeuft Marching Cubes. Die Topologie
   aendert sich dadurch kontinuierlich -- kein Ploppen mehr.          */

let lastKey = '';

function rebuildGeometry(force){
  if(!D) return;
  const lf = Math.max(0, Math.min(LAYERS-1, pos));
  const i0 = Math.floor(lf), i1 = Math.min(LAYERS-1, i0+1);
  const k  = lf - i0, ke = k*k*(3-2*k);        // smoothstep

  // Nur neu rechnen, wenn sich wirklich etwas geaendert hat
  const key = i0+'|'+ke.toFixed(3)+'|'+ISO.toFixed(3)+'|'+SOLO+'|'+VIS.join('');
  if(key === lastKey && !force) return;
  lastKey = key;

  const per = RES*RES*RES;

  for(let s=0; s<SER.length; s++){
    const S = SER[s];
    if(!visible(s)){ S.count = 0; S.pcount = 0; continue; }

    // --- Feld mischen ------------------------------------------------
    const A = S.grids[i0], B = S.grids[i1], out = S.blend;
    if(i0 === i1 || ke < 1e-6){
      out.set(A);
    } else {
      for(let q=0;q<per;q++) out[q] = A[q] + (B[q]-A[q])*ke;
    }

    // --- Marching Cubes ----------------------------------------------
    const mesh = MC.build(out, RES, ISO, LO, HI);
    S.count = mesh.count;
    if(mesh.count){
      gl.bindBuffer(gl.ARRAY_BUFFER, S.vbo);
      gl.bufferData(gl.ARRAY_BUFFER,
        mesh.pos.subarray(0, mesh.count*3), gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, S.nbo);
      gl.bufferData(gl.ARRAY_BUFFER,
        mesh.nrm.subarray(0, mesh.count*3), gl.DYNAMIC_DRAW);
    }

    // --- Token-Punkte, ebenfalls interpoliert --------------------------
    const PA = D.series[s].pts[i0], PB = D.series[s].pts[i1];
    const T = PA.length;
    if(!S.ptArr || S.ptArr.length !== T*3) S.ptArr = new Float32Array(T*3);
    for(let t=0;t<T;t++){
      const a = PA[t], b = PB[t];
      S.ptArr[t*3  ] = a[0] + (b[0]-a[0])*ke;
      S.ptArr[t*3+1] = a[1] + (b[1]-a[1])*ke;
      S.ptArr[t*3+2] = a[2] + (b[2]-a[2])*ke;
    }
    S.pcount = T;
    gl.bindBuffer(gl.ARRAY_BUFFER, S.pbo);
    gl.bufferData(gl.ARRAY_BUFFER, S.ptArr, gl.DYNAMIC_DRAW);
  }
}

/* ════════════════════════════════════════════════════════════════════
   9 · RENDER
   ════════════════════════════════════════════════════════════════════ */
let boxVBO = null;
function makeBox(){
  const [a,b,c0] = LO, [d,e,f] = HI;
  const v = [
    a,b,c0, d,b,c0,  d,b,c0, d,e,c0,  d,e,c0, a,e,c0,  a,e,c0, a,b,c0,
    a,b,f,  d,b,f,   d,b,f,  d,e,f,   d,e,f,  a,e,f,   a,e,f,  a,b,f,
    a,b,c0, a,b,f,   d,b,c0, d,b,f,   d,e,c0, d,e,f,   a,e,c0, a,e,f];
  boxVBO = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, boxVBO);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(v), gl.STATIC_DRAW);
}

function hsl2rgb(h,s,l){
  h=((h%360)+360)%360/360;
  const q = l<0.5 ? l*(1+s) : l+s-l*s, p = 2*l-q;
  const f = t=>{ t=(t+1)%1;
    if(t<1/6) return p+(q-p)*6*t;
    if(t<1/2) return q;
    if(t<2/3) return p+(q-p)*(2/3-t)*6;
    return p; };
  return [f(h+1/3), f(h), f(h-1/3)];
}

function render(){
  gl.clearColor(0.020, 0.027, 0.051, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LEQUAL);

  const asp = glc.width/glc.height;
  const cy=Math.cos(yaw), sy=Math.sin(yaw), cp=Math.cos(pit), sp=Math.sin(pit);
  const eye = [dist*cp*sy, dist*sp, dist*cp*cy];
  const V = m4lookAt(eye, [0,0,0], [0,1,0]);
  const P = m4persp(0.95, asp, 0.05, 50.0);
  const MVP = m4mul(P, V);
  const NM = m3fromM4(m4ident());

  if(!D) return;

  // ── Bounding-Box dezent ───────────────────────────────────────────
  gl.useProgram(PR_LINE);
  gl.uniformMatrix4fv(LOC.line.uMVP, false, MVP);
  gl.uniform4f(LOC.line.uCol, 0.42, 0.50, 0.70, 0.085);
  gl.bindBuffer(gl.ARRAY_BUFFER, boxVBO);
  gl.enableVertexAttribArray(LOC.line.aPos);
  gl.vertexAttribPointer(LOC.line.aPos, 3, gl.FLOAT, false, 0, 0);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.drawArrays(gl.LINES, 0, 24);

  const act = activeIdx();
  const t = LAYERS>1 ? pos/(LAYERS-1) : 0;

  // ── Isoflächen ────────────────────────────────────────────────────
  if(surf){
    gl.useProgram(PR_SURF);
    gl.uniformMatrix4fv(LOC.surf.uMVP, false, MVP);
    gl.uniformMatrix3fv(LOC.surf.uNM, false, NM);
    gl.uniform3f(LOC.surf.uEye, eye[0], eye[1], eye[2]);

    const multi = act.length > 1;
    if(multi){
      // Mehrere Formen: leicht transparent, Depth-Write aus für
      // korrektes Durchscheinen, aber Depth-Test bleibt an.
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);
    } else {
      gl.disable(gl.BLEND);
      gl.depthMask(true);
    }
    gl.enable(gl.CULL_FACE); gl.cullFace(gl.BACK);

    // bei Transparenz: hinten nach vorne
    const order = act.slice();
    if(multi) order.reverse();

    for(const i of order){
      const S = SER[i];
      if(!S || !S.count) continue;
      const hue = D.series[i].hue + t*18;
      const base = hsl2rgb(hue, 0.52 + t*0.12, 0.42 + t*0.06);
      const rim  = hsl2rgb(hue + 34, 0.85, 0.62);
      gl.uniform3f(LOC.surf.uCol, base[0], base[1], base[2]);
      gl.uniform3f(LOC.surf.uRim, rim[0], rim[1], rim[2]);
      gl.uniform1f(LOC.surf.uAl, multi ? 0.60 : 1.0);

      gl.bindBuffer(gl.ARRAY_BUFFER, S.vbo);
      gl.enableVertexAttribArray(LOC.surf.aPos);
      gl.vertexAttribPointer(LOC.surf.aPos, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, S.nbo);
      gl.enableVertexAttribArray(LOC.surf.aNrm);
      gl.vertexAttribPointer(LOC.surf.aNrm, 3, gl.FLOAT, false, 0, 0);

      gl.drawArrays(gl.TRIANGLES, 0, S.count);
    }
    gl.disable(gl.CULL_FACE);
    gl.depthMask(true);
  }

  // ── Token-Punkte ──────────────────────────────────────────────────
  if(showPts || !surf){
    gl.useProgram(PR_PT);
    gl.uniformMatrix4fv(LOC.pt.uMVP, false, MVP);
    gl.uniform1f(LOC.pt.uSize, 9.0*DPR);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
    for(const i of act){
      const S = SER[i];
      if(!S || !S.pcount) continue;
      const col = hsl2rgb(D.series[i].hue, 0.70, 0.70);
      gl.uniform3f(LOC.pt.uCol, col[0], col[1], col[2]);
      gl.bindBuffer(gl.ARRAY_BUFFER, S.pbo);
      gl.enableVertexAttribArray(LOC.pt.aPos);
      gl.vertexAttribPointer(LOC.pt.aPos, 3, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.POINTS, 0, S.pcount);
    }
    gl.depthMask(true);
  }

  // ── Token-Beschriftung im Overlay ─────────────────────────────────
  ox.clearRect(0,0,W,H);
  if(act.length <= 2 && (showPts || !surf)){
    ox.font = '10px ui-monospace,monospace';
    ox.textAlign = 'center';
    for(const i of act){
      const S = SER[i];
      if(!S || !S.pcount) continue;
      const col = `hsl(${D.series[i].hue},58%,84%)`;
      for(let q=0;q<S.pcount;q++){
        const wv = projectPoint(MVP, S.ptArr[q*3], S.ptArr[q*3+1], S.ptArr[q*3+2]);
        if(!wv) continue;
        ox.globalAlpha = Math.min(0.92, Math.max(0, (1.6 - wv.d)*0.9));
        if(ox.globalAlpha < 0.05) continue;
        ox.fillStyle = col;
        ox.fillText(D.series[i].tokens[q]||'', wv.x, wv.y - 9);
      }
    }
    ox.globalAlpha = 1;
  }
}

function projectPoint(M, px, py, pz){
  const cx = M[0]*px + M[4]*py + M[8]*pz + M[12];
  const cyv= M[1]*px + M[5]*py + M[9]*pz + M[13];
  const cz = M[2]*px + M[6]*py + M[10]*pz + M[14];
  const cw = M[3]*px + M[7]*py + M[11]*pz + M[15];
  if(cw <= 0.001) return null;
  return {x: (cx/cw*0.5+0.5)*W, y: (1 - (cyv/cw*0.5+0.5))*H, d: cw/dist};
}

/* ════════════════════════════════════════════════════════════════════
   10 · SATZ-EDITOR
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
   11 · PIPELINE-EDITOR
   ════════════════════════════════════════════════════════════════════ */
const RED_OPTS=`__REDOPTS__`;
const stagesHost=document.getElementById('stages');
let STAGES=[{m:'pca',d:48},{m:'umap',d:3}];

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

/* ── Iso-Regler: live, ohne Server ─────────────────────────────────── */
const isoEl = document.getElementById('iso'), isoLab = document.getElementById('isoLab');
isoEl.oninput = () => { ISO = parseFloat(isoEl.value); isoLab.textContent = ISO.toFixed(2); };
const smEl = document.getElementById('smooth'), smLab = document.getElementById('smLab');
smEl.oninput = () => { smLab.textContent = parseFloat(smEl.value).toFixed(2); };

/* ════════════════════════════════════════════════════════════════════
   12 · SLIDER / RAIL
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
   13 · TRANSPORT · LEGENDE · SICHTBARKEIT
   ════════════════════════════════════════════════════════════════════ */
const bPlay=document.getElementById('bPlay'),
      bSurf=document.getElementById('bSurf'),
      bPts =document.getElementById('bPts'),
      bAll =document.getElementById('bAll'),
      bHC  =document.getElementById('bHC'),
      legHost=document.getElementById('leg');

function sync(){
  bPlay.textContent=playing?'❚❚':'▶';
  bPlay.classList.toggle('on',playing)}

function refresh(){lastHudLayer=-1;lastKey='';renderLegend();buildTokens();updateHUD(true)}

function renderLegend(){
  legHost.innerHTML='';
  if(!D)return;
  D.series.forEach((s,i)=>{
    const d=document.createElement('div');
    const vis=visible(i);
    d.className='lgi'+(vis?'':' off');
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
  bSurf.textContent=surf?'◆':'◇';lastKey=''};
bPts.onclick=()=>{showPts=!showPts;bPts.classList.toggle('on',showPts);lastKey=''};
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
  if(e.code==='KeyP'){bPts.onclick();return}
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
   14 · SPARKLINES
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
   15 · HUD
   ════════════════════════════════════════════════════════════════════ */
const $=id=>document.getElementById(id);
const f3=v=>(v===undefined||v===null)?'—':(Math.abs(v)<10?v.toFixed(3):v.toFixed(1));
const i0f=v=>(v===undefined||v===null)?'—':String(Math.round(v));
let lastHudLayer=-1;

function activeIdx(){
  const out=[];
  if(!D)return out;
  for(let i=0;i<D.series.length;i++)if(visible(i))out.push(i);
  return out}

/* grobe Komponenten-Zahl: Anzahl Token-Cluster unterhalb 2σ */
function componentCount(s,L){
  const P=D.series[s].pts[L], n=P.length, lim=D.sigma*2.0;
  const seen=new Uint8Array(n); let comp=0;
  for(let i=0;i<n;i++){
    if(seen[i])continue;
    comp++; const st=[i]; seen[i]=1;
    while(st.length){
      const a=st.pop();
      for(let j=0;j<n;j++){
        if(seen[j])continue;
        const dx=P[a][0]-P[j][0],dy=P[a][1]-P[j][1],dz=P[a][2]-P[j][2];
        if(dx*dx+dy*dy+dz*dz <= lim*lim){seen[j]=1;st.push(j)}}}}
  return comp}

function updateHUD(force){
  if(!D)return;
  const L=Math.round(Math.max(0,Math.min(LAYERS-1,pos)));
  if(L===lastHudLayer&&!force)return;
  lastHudLayer=L;

  const act=activeIdx();
  const fi=visible(FOCUS)?FOCUS:(act.length?act[0]:0);
  const s=D.series[fi], sh=s.shapes[L];

  $('focLabel').textContent=s.label;
  $('focLabel').style.color=`hsl(${s.hue},70%,66%)`;
  $('s-drift').textContent=f3(s.drift[L]);
  $('s-tris').textContent=i0f(SER[fi]?SER[fi].count/3:0);
  $('s-comp').textContent=i0f(componentCount(fi,L));
  $('s-edim').textContent=f3(sh.eff_dim);
  $('s-aniso').textContent=f3(sh.anisotropy);
  $('s-curv').textContent=f3(sh.curvature);
  $('s-norm').textContent=f3(s.norms[L]);

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

  const ser1=act.map(i=>({v:D.series[i].drift,
    c:`hsl(${D.series[i].hue},68%,64%)`,w:1.5}));
  sparkLines($('sp1'),ser1,L);
  $('leg1').innerHTML='<i>DRIFT L-1→L</i>'+
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
   16 · TOKEN-LEISTE
   ════════════════════════════════════════════════════════════════════ */
function buildTokens(){
  const host=$('toks'); host.innerHTML='';
  if(!D)return;
  const act=activeIdx();
  if(act.length>4)return;
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
   17 · LOOP
   ════════════════════════════════════════════════════════════════════ */
const hn=$('hn');
let last=performance.now();
function loop(now){
  const dt=Math.min(.05,(now-last)/1000); last=now;
  const ease=(cur,t_,sp)=>cur+(t_-cur)*(1-Math.exp(-sp*dt));

  yaw=ease(yaw,tg.y,6); pit=ease(pit,tg.p,6);

  if(D){
    if(targetPos!==null){
      pos=ease(pos,targetPos,STEP_SPD);
      if(Math.abs(pos-targetPos)<.004){pos=targetPos;targetPos=null}
    }else if(playing&&scrub===null){
      pos+=AUTO_SPD*dt;
      if(pos>LAYERS-1){pos=LAYERS-1;playing=false;sync()}}

    rebuildGeometry(false);
    render();
    hn.textContent=String(Math.round(pos)).padStart(2,'0');
    updateRail(); updateHUD();
  }else if(gl){
    gl.clearColor(0.020,0.027,0.051,1);
    gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  }
  requestAnimationFrame(loop)}

/* ════════════════════════════════════════════════════════════════════
   18 · SERVER-CALL
   ════════════════════════════════════════════════════════════════════ */
const msg=$('msg'), go=$('go'), boot=$('boot'), bootmsg=$('bootmsg');
let busy=false;

function freeSeries(){
  for(const S of SER){
    if(S.vbo) gl.deleteBuffer(S.vbo);
    if(S.nbo) gl.deleteBuffer(S.nbo);
    if(S.pbo) gl.deleteBuffer(S.pbo);
  }
  SER=[];
}

async function run(){
  if(busy)return;
  const act=ROWS.filter(r=>r.on&&r.text.trim());
  if(!act.length){
    msg.className='err'; msg.textContent='Kein aktiver Satz mit Text.'; return}
  if(act.length>MAXS){
    msg.className='err'; msg.textContent='Maximal '+MAXS+' Sätze.'; return}

  busy=true;
  go.classList.add('busy'); go.textContent='RECHNET …';
  msg.className=''; msg.textContent='Hidden States · Dichtefeld · Transport …';
  boot.classList.remove('gone');
  bootmsg.textContent='FORWARD-PASS · PROJEKTION · SKALARFELD';

  const body={
    texts:act.map(r=>r.text),
    labels:act.map(r=>r.label),
    hues:act.map(r=>r.hue),
    model:$('model').value, emd:$('emd').value,
    grid_res:parseInt($('gres').value,10),
    smooth:parseFloat(smEl.value),
    iso:parseFloat(isoEl.value),
    pipeline:STAGES.map(s=>({method:s.m,dims:s.d}))};

  try{
    const r=await fetch('/api/analyse',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j=await r.json();
    if(!r.ok||j.error)throw new Error(j.error||('HTTP '+r.status));

    bootmsg.textContent='DEKODIERE GITTER …';
    await new Promise(q=>setTimeout(q,16));

    D=j; LAYERS=D.n_layers; RES=D.res; LO=D.lo; HI=D.hi;

    freeSeries();
    if(boxVBO) gl.deleteBuffer(boxVBO);
    makeBox();

    const per=RES*RES*RES;
    for(let s=0;s<D.series.length;s++){
      const grids=decodeGrids(D.series[s].grid, LAYERS, RES, D.series[s].grid_max);
      SER.push({
        grids,
        blend:new Float32Array(per),
        vbo:gl.createBuffer(), nbo:gl.createBuffer(), pbo:gl.createBuffer(),
        ptArr:null, count:0, pcount:0});
      D.series[s].grid=null;          // Speicher freigeben
    }

    pos=0; targetPos=null; playing=false;
    VIS=D.series.map(()=>true); SOLO=-1; FOCUS=0; lastHudLayer=-1; lastKey='';
    ISO=D.iso; isoEl.value=ISO; isoLab.textContent=ISO.toFixed(2);
    yaw=-0.68; pit=0.26; dist=3.35; tg={y:yaw,p:pit};

    buildRail(); renderLegend(); buildTokens(); sync();
    rebuildGeometry(true); updateHUD(true);

    const tks=D.series.map(s=>s.tokens.length).join('/');
    const tri=SER[0]?Math.round(SER[0].count/3):0;
    msg.className='ok';
    msg.textContent=`${D.model.split('/').pop()} · ${LAYERS} Layer · `
      +`${D.n_series} Satz${D.n_series>1?'·e':''} · ${tks} Tokens · `
      +`${RES}³ Gitter · ${tri} △ · ${D.device.toUpperCase()}`;
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
# 19) HTTP-SERVER
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

            raw = req.get("texts")
            if raw is None:
                raw = [req.get("text_a"), req.get("text_b")]
            texts = [str(t).strip() for t in (raw or []) if t and str(t).strip()]
            if not texts:
                raise ValueError("Keine Sätze übergeben.")
            if len(texts) > MAX_SERIES:
                texts = texts[:MAX_SERIES]

            labels = [str(l)[:10] for l in (req.get("labels") or [])][:len(texts)]
            hues = [float(h) % 360.0 for h in (req.get("hues") or [])
                    if isinstance(h, (int, float))][:len(texts)]

            model = req.get("model") or "distilbert-base-uncased"
            emd = req.get("emd") or "exact"
            if emd not in ("exact", "sliced", "sinkhorn"):
                emd = "exact"

            grid_res = int(req.get("grid_res") or 48)
            grid_res = max(24, min(72, grid_res))
            smooth = float(req.get("smooth") or 1.0)
            smooth = max(0.4, min(3.0, smooth))
            iso = float(req.get("iso") or 0.55)
            iso = max(0.08, min(2.0, iso))

            pipeline = req.get("pipeline") or [{"method": "pca", "dims": 3}]
            clean = []
            for st in pipeline[:4]:
                m = str(st.get("method", "pca")).lower()
                if m not in REDUCERS:
                    m = "pca"
                d = int(st.get("dims", 3))
                clean.append({"method": m, "dims": max(1, min(512, d))})

            lbl = " -> ".join(f'{s["method"]}:{s["dims"]}' for s in clean)
            print(f"[api] {model} | n={len(texts)} | {lbl} | emd={emd} | "
                  f"grid={grid_res}³ | sigma×{smooth}")

            data = analyse(texts, model, clean, emd,
                           iso=iso, smooth=smooth, grid_res=grid_res,
                           hues=hues or None, labels=labels or None)

            payload = json.dumps(data).encode("utf-8")
            print(f"[api] fertig · {len(payload)/1e6:.2f} MB · "
                  f"{data['n_layers']} Layer · {data['res']}³")
            self._send(200, payload, "application/json; charset=utf-8")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send(500, json.dumps({"error": str(e)}).encode("utf-8"),
                       "application/json; charset=utf-8")

# ============================================================================
# 20) ENTRY POINT
# ============================================================================

def main():
    port = int(os.environ.get("MANIFOLD_PORT", "8051"))
    url = f"http://127.0.0.1:{port}"
    print("=" * 74)
    print("  HIDDEN-STATE MANIFOLD EXPLORER · v5 · Marching Cubes · WebGL")
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
    print("  Geaendert gegenueber v4:")
    print("   · Eine einzige Isoflaeche statt Rips-Dreiecken + Ghost-Stapel")
    print("   · Skalarfeld pro Layer, linear interpoliert -> echtes Morphing")
    print("   · WebGL mit Depth-Buffer statt Painter's Algorithm")
    print("   · Globale Reduktion ueber alle Layer -> kein Achsen-Umklappen")
    print("   · Iso-Schwelle live regelbar, ohne Neuberechnung")
    print()
    print("  Bedienung : ▶ Durchlauf · ◂ ▸ Layer · ◆ Flaeche · · Punkte")
    print("              ◉ alle · A Hochkontrast · 1-8 Satz · ⇧+1-8 Solo")
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
