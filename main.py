#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers",
#   "numpy",
#   "scikit-learn",
#   "pot",
#   "plotly",
#   "dash",
#   "pandas",
#   "umap-learn",
#   "accelerate",
# ]
# ///
"""
Hidden-State Manifold Explorer
==============================
Zeigt, wie ein Satz durch die Layer eines LLM "wandert":
 - Token-Embeddings -> Layer 0..N hidden states
 - Dimensionsreduktion (PCA / t-SNE / UMAP)
 - Shape-Metriken der Mannigfaltigkeit pro Layer
 - Wasserstein-Distanz (EMD) zwischen den Mannigfaltigkeiten zweier Saetze

Start:  python3 manifold.py
"""

import os
import subprocess
import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# 0) BOOTSTRAP: uv-basierte Selbstinstallation + Neustart
# ----------------------------------------------------------------------------

REQUIRED = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("ot", "pot"),
    ("plotly", "plotly"),
    ("dash", "dash"),
    ("pandas", "pandas"),
    ("umap", "umap-learn"),
    ("accelerate", "accelerate"),
]

VENV_DIR = Path(__file__).resolve().parent / ".venv-manifold"
BOOTSTRAP_FLAG = "MANIFOLD_BOOTSTRAPPED"

def _venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"

def _have_uv() -> bool:
    from shutil import which
    return which("uv") is not None

def _install_uv():
    print("[bootstrap] 'uv' nicht gefunden -> installiere via pip ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "-q", "uv"])
    # PATH ggf. um user-base/bin erweitern
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
    """Stellt sicher, dass alle Deps da sind. Startet sich notfalls neu."""
    if os.environ.get(BOOTSTRAP_FLAG) == "1":
        missing = _missing_modules()
        if missing:
            print(f"[bootstrap] FEHLER: nach Neustart fehlen immer noch: {missing}")
            sys.exit(1)
        return  # alles gut

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
    print("[bootstrap] Installiere Abhaengigkeiten via uv (dauert beim ersten Mal) ...")
    env = dict(os.environ, VIRTUAL_ENV=str(VENV_DIR))
    subprocess.check_call(["uv", "pip", "install", "--python", str(_venv_python()), *pkgs], env=env)

    print("[bootstrap] Neustart im venv ...")
    env[BOOTSTRAP_FLAG] = "1"
    os.execve(str(_venv_python()), [str(_venv_python()), os.path.abspath(__file__), *sys.argv[1:]], env)

bootstrap()

# ----------------------------------------------------------------------------
# 1) Imports (ab hier garantiert verfuegbar)
# ----------------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import ot  # POT: Python Optimal Transport
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer

import dash
from dash import dcc, html, Input, Output, State

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# ----------------------------------------------------------------------------
# 2) Modell-Katalog
# ----------------------------------------------------------------------------

MODELS = {
    # --- CPU-freundlich (klein, laeuft ueberall) ---
    "distilbert-base-uncased":            {"size": "66M",  "tier": "CPU",  "note": "6 Layer, schnell"},
    "bert-base-uncased":                  {"size": "110M", "tier": "CPU",  "note": "12 Layer, Klassiker"},
    "sentence-transformers/all-MiniLM-L6-v2": {"size": "22M", "tier": "CPU", "note": "winzig, 6 Layer"},
    "roberta-base":                       {"size": "125M", "tier": "CPU",  "note": "12 Layer"},
    "gpt2":                               {"size": "124M", "tier": "CPU",  "note": "decoder-only, 12 Layer"},
    "EleutherAI/pythia-160m":             {"size": "160M", "tier": "CPU",  "note": "decoder-only, RoPE"},

    # --- Mittel (CPU moeglich, GPU angenehmer) ---
    "gpt2-medium":                        {"size": "355M", "tier": "CPU+", "note": "24 Layer"},
    "EleutherAI/pythia-410m":             {"size": "410M", "tier": "CPU+", "note": "24 Layer"},
    "microsoft/deberta-v3-base":          {"size": "184M", "tier": "CPU+", "note": "disentangled attn"},

    # --- GPU empfohlen ---
    "EleutherAI/pythia-1.4b":             {"size": "1.4B", "tier": "GPU",  "note": "24 Layer"},
    "Qwen/Qwen2.5-0.5B":                  {"size": "0.5B", "tier": "GPU",  "note": "modern, RoPE"},
    "Qwen/Qwen2.5-1.5B":                  {"size": "1.5B", "tier": "GPU",  "note": "28 Layer"},
    "meta-llama/Llama-3.2-1B":            {"size": "1.2B", "tier": "GPU",  "note": "gated, HF-Login noetig"},
    "mistralai/Mistral-7B-v0.1":          {"size": "7B",   "tier": "GPU",  "note": "braucht >=16GB VRAM"},
}

_CACHE = {}

def load_model(name: str):
    if name in _CACHE:
        return _CACHE[name]
    print(f"[model] Lade {name} auf {DEVICE} ...")
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float32 if DEVICE == "cpu" else torch.float16
    model = AutoModel.from_pretrained(name, torch_dtype=dtype, output_hidden_states=True)
    model.to(DEVICE).eval()
    _CACHE[name] = (tok, model)
    return tok, model

# ----------------------------------------------------------------------------
# 3) Hidden States extrahieren
# ----------------------------------------------------------------------------

@torch.no_grad()
def hidden_states(text: str, model_name: str):
    """Gibt (tokens, [L+1, T, D]-Array) zurueck. Index 0 = Embedding-Layer."""
    tok, model = load_model(model_name)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    out = model(**enc)
    hs = torch.stack(out.hidden_states)            # [L+1, 1, T, D]
    hs = hs[:, 0].float().cpu().numpy()            # [L+1, T, D]
    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
    tokens = [t.replace("Ġ", "_").replace("▁", "_") for t in tokens]
    return tokens, hs

# ----------------------------------------------------------------------------
# 4) Shape-Metriken der Mannigfaltigkeit
# ----------------------------------------------------------------------------

def shape_metrics(X: np.ndarray) -> dict:
    """X: [T, D] Punktwolke eines Layers."""
    Xc = X - X.mean(0, keepdims=True)
    T = Xc.shape[0]
    if T < 2:
        return dict(eff_dim=1.0, anisotropy=0.0, radius=0.0, spectrum=[1.0])

    # Singulaerwerte -> Varianzspektrum
    s = np.linalg.svd(Xc, compute_uv=False)
    ev = (s ** 2) / max(T - 1, 1)
    p = ev / (ev.sum() + 1e-12)

    # Partizipationsratio = "effektive Dimension" der Mannigfaltigkeit
    eff_dim = float((ev.sum() ** 2) / ((ev ** 2).sum() + 1e-12))

    # Anisotropie: mittlere paarweise Cosine-Sim (hoch = degeneriert/"kegelartig")
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    G = Xn @ Xn.T
    iu = np.triu_indices(T, k=1)
    anisotropy = float(G[iu].mean()) if len(iu[0]) else 0.0

    # Gyrationsradius: raeumliche Ausdehnung
    radius = float(np.sqrt((Xc ** 2).sum(1).mean()))

    return dict(
        eff_dim=eff_dim,
        anisotropy=anisotropy,
        radius=radius,
        spectrum=p[:12].tolist(),
        var_top2=float(p[:2].sum()),
    )

# ----------------------------------------------------------------------------
# 5) Wasserstein / Earth Mover's Distance
# ----------------------------------------------------------------------------

def wasserstein(A: np.ndarray, B: np.ndarray, mode="exact", normalize=True, n_proj=256):
    """EMD zwischen zwei Punktwolken A:[n,D], B:[m,D] mit Uniform-Gewichten."""
    A = np.ascontiguousarray(A, dtype=np.float64)
    B = np.ascontiguousarray(B, dtype=np.float64)

    if normalize:
        # Skaleninvariant machen -> Layer vergleichbar (Norm waechst sonst stark)
        scale = (np.linalg.norm(A - A.mean(0), axis=1).mean() +
                 np.linalg.norm(B - B.mean(0), axis=1).mean()) / 2 + 1e-12
        A, B = A / scale, B / scale

    a = np.ones(len(A)) / len(A)
    b = np.ones(len(B)) / len(B)

    if mode == "sliced":
        return float(ot.sliced_wasserstein_distance(A, B, a, b, n_projections=n_proj, seed=0))

    M = ot.dist(A, B, metric="euclidean")   # Ground-Metrik
    if mode == "sinkhorn":
        G = ot.sinkhorn(a, b, M, reg=0.05, numItermax=500)
        return float((G * M).sum())
    return float(ot.emd2(a, b, M))          # exakte EMD

# ----------------------------------------------------------------------------
# 6) Dimensionsreduktion
# ----------------------------------------------------------------------------

def reduce(X: np.ndarray, method="pca", dims=2, seed=0):
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

def joint_reduce(blocks, method="pca", dims=2):
    """Reduziert mehrere Punktwolken in EINEM gemeinsamen Raum (vergleichbar!)."""
    sizes = [b.shape[0] for b in blocks]
    Z = reduce(np.vstack(blocks), method=method, dims=dims)
    out, i = [], 0
    for s in sizes:
        out.append(Z[i:i + s])
        i += s
    return out

# ----------------------------------------------------------------------------
# 7) Dash-App
# ----------------------------------------------------------------------------

app = dash.Dash(__name__)
app.title = "LLM Manifold Explorer"

PALETTE = ["#4cc9f0", "#f72585"]

app.layout = html.Div(style={"fontFamily": "monospace", "background": "#0d1117",
                             "color": "#e6edf3", "padding": "18px"}, children=[
    html.H2("🌀 LLM Hidden-State Manifold Explorer"),
    html.Div(f"Device: {DEVICE.upper()}  |  CUDA: {torch.cuda.is_available()}",
             style={"color": "#7ee787", "marginBottom": "14px"}),

    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px"}, children=[
        html.Div([html.Label("Satz A"),
                  dcc.Textarea(id="txt-a", value="The cat sat quietly on the warm windowsill.",
                               style={"width": "100%", "height": "60px"})]),
        html.Div([html.Label("Satz B (fuer Wasserstein-Vergleich)"),
                  dcc.Textarea(id="txt-b", value="A dog slept loudly beside the cold doorway.",
                               style={"width": "100%", "height": "60px"})]),
    ]),

    html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)",
                    "gap": "10px", "marginTop": "12px"}, children=[
        html.Div([html.Label("Modell"),
                  dcc.Dropdown(id="model", value="distilbert-base-uncased",
                               options=[{"label": f"[{v['tier']}] {k} ({v['size']}) – {v['note']}",
                                         "value": k} for k, v in MODELS.items()],
                               style={"color": "#000"})]),
        html.Div([html.Label("Projektion"),
                  dcc.Dropdown(id="method", value="pca", style={"color": "#000"},
                               options=[{"label": m.upper(), "value": m}
                                        for m in ["pca", "tsne", "umap"]])]),
        html.Div([html.Label("Dimensionen"),
                  dcc.Dropdown(id="dims", value=3, style={"color": "#000"},
                               options=[{"label": "2D", "value": 2}, {"label": "3D", "value": 3}])]),
        html.Div([html.Label("EMD-Modus"),
                  dcc.Dropdown(id="emd", value="exact", style={"color": "#000"},
                               options=[{"label": "Exakt (EMD)", "value": "exact"},
                                        {"label": "Sliced", "value": "sliced"},
                                        {"label": "Sinkhorn", "value": "sinkhorn"}])]),
    ]),

    html.Button("▶ Analysieren", id="run", n_clicks=0,
                style={"marginTop": "14px", "padding": "10px 26px", "background": "#238636",
                       "color": "#fff", "border": "none", "borderRadius": "6px",
                       "cursor": "pointer", "fontSize": "15px"}),

    dcc.Loading(color="#4cc9f0", children=[
        html.Div(id="status", style={"marginTop": "10px", "color": "#ffa657"}),
        html.Label("Layer", style={"marginTop": "14px", "display": "block"}),
        dcc.Slider(id="layer", min=0, max=0, step=1, value=0,
                   tooltip={"placement": "bottom", "always_visible": True}),
        dcc.Graph(id="g-manifold", style={"height": "600px"}),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px"}, children=[
            dcc.Graph(id="g-emd", style={"height": "380px"}),
            dcc.Graph(id="g-shape", style={"height": "380px"}),
        ]),
        dcc.Graph(id="g-spectrum", style={"height": "380px"}),
    ]),

    dcc.Store(id="store"),
])

@app.callback(
    Output("store", "data"), Output("layer", "max"), Output("layer", "marks"),
    Output("layer", "value"), Output("status", "children"),
    Input("run", "n_clicks"),
    State("txt-a", "value"), State("txt-b", "value"),
    State("model", "value"), State("method", "value"),
    State("dims", "value"), State("emd", "value"),
    prevent_initial_call=True,
)
def compute(_, ta, tb, model_name, method, dims, emd_mode):
    tok_a, HA = hidden_states(ta, model_name)
    tok_b, HB = hidden_states(tb, model_name)
    n_layers = HA.shape[0]

    proj_a, proj_b, emds, shapes_a, shapes_b = [], [], [], [], []
    for L in range(n_layers):
        A, B = HA[L], HB[L]
        pa, pb = joint_reduce([A, B], method=method, dims=dims)
        proj_a.append(pa.tolist())
        proj_b.append(pb.tolist())
        emds.append(wasserstein(A, B, mode=emd_mode))
        shapes_a.append(shape_metrics(A))
        shapes_b.append(shape_metrics(B))

    # Drift: wie weit bewegt sich die Mannigfaltigkeit von Layer zu Layer?
    drift = [0.0] + [wasserstein(HA[L - 1], HA[L], mode=emd_mode) for L in range(1, n_layers)]

    marks = {i: str(i) for i in range(0, n_layers, max(1, n_layers // 12))}
    data = dict(tok_a=tok_a, tok_b=tok_b, proj_a=proj_a, proj_b=proj_b,
                emds=emds, drift=drift, shapes_a=shapes_a, shapes_b=shapes_b,
                dims=dims, n_layers=n_layers, model=model_name, method=method)
    status = (f"✔ {model_name} | {n_layers} Layer | "
              f"A: {len(tok_a)} Tokens, B: {len(tok_b)} Tokens | Projektion: {method.upper()}")
    return data, n_layers - 1, marks, 0, status

@app.callback(Output("g-manifold", "figure"), Input("layer", "value"), Input("store", "data"))
def draw_manifold(L, d):
    if not d:
        return go.Figure()
    dims = d["dims"]
    fig = go.Figure()
    for idx, (key, toks, name) in enumerate([("proj_a", "tok_a", "Satz A"),
                                             ("proj_b", "tok_b", "Satz B")]):
        P = np.array(d[key][L])
        labels = d[toks]
        common = dict(mode="markers+lines+text", text=labels, textposition="top center",
                      name=name, marker=dict(size=7, color=PALETTE[idx]),
                      line=dict(color=PALETTE[idx], width=1, dash="dot"),
                      textfont=dict(size=10, color=PALETTE[idx]))
        if dims == 3 and P.shape[1] >= 3:
            fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], **common))
        else:
            fig.add_trace(go.Scatter(x=P[:, 0], y=P[:, 1], **common))

    sa, sb = d["shapes_a"][L], d["shapes_b"][L]
    fig.update_layout(
        title=(f"Layer {L}/{d['n_layers']-1} — eff.Dim A={sa['eff_dim']:.2f} "
               f"B={sb['eff_dim']:.2f} | Anisotropie A={sa['anisotropy']:.3f} "
               f"B={sb['anisotropy']:.3f} | W₁={d['emds'][L]:.4f}"),
        template="plotly_dark", margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig

@app.callback(Output("g-emd", "figure"), Input("store", "data"), Input("layer", "value"))
def draw_emd(d, L):
    if not d:
        return go.Figure()
    layers = list(range(d["n_layers"]))
    fig = go.Figure()

    # W1(A,B) pro Layer
    fig.add_trace(go.Scatter(
        x=layers, y=d["emds"], mode="lines+markers", name="W₁(A, B)",
        line=dict(color="#f72585", width=3), marker=dict(size=7),
        hovertemplate="Layer %{x}<br>W₁ = %{y:.4f}<extra></extra>"))

    # Layer-zu-Layer-Drift von Satz A
    fig.add_trace(go.Scatter(
        x=layers, y=d["drift"], mode="lines+markers", name="W₁(A_L-1, A_L)  [Drift]",
        line=dict(color="#7ee787", width=2, dash="dash"), marker=dict(size=5),
        hovertemplate="Layer %{x}<br>Drift = %{y:.4f}<extra></extra>"))

    # Marker fuer den aktuell gewaehlten Layer
    fig.add_vline(x=L, line=dict(color="#58a6ff", width=2, dash="dot"))

    fig.update_layout(
        title="Wasserstein-Distanz (EMD) über die Layer",
        xaxis_title="Layer", yaxis_title="W₁ (scale-normalisiert)",
        template="plotly_dark", margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(orientation="h", y=-0.22))
    return fig

@app.callback(Output("g-shape", "figure"), Input("store", "data"), Input("layer", "value"))
def draw_shape(d, L):
    if not d:
        return go.Figure()
    layers = list(range(d["n_layers"]))
    ea = [s["eff_dim"] for s in d["shapes_a"]]
    eb = [s["eff_dim"] for s in d["shapes_b"]]
    aa = [s["anisotropy"] for s in d["shapes_a"]]
    ab = [s["anisotropy"] for s in d["shapes_b"]]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=layers, y=ea, name="eff.Dim A",
                             line=dict(color="#4cc9f0", width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=layers, y=eb, name="eff.Dim B",
                             line=dict(color="#f72585", width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=layers, y=aa, name="Anisotropie A",
                             line=dict(color="#4cc9f0", width=1.5, dash="dot")), secondary_y=True)
    fig.add_trace(go.Scatter(x=layers, y=ab, name="Anisotropie B",
                             line=dict(color="#f72585", width=1.5, dash="dot")), secondary_y=True)

    fig.add_vline(x=L, line=dict(color="#58a6ff", width=2, dash="dot"))
    fig.update_yaxes(title_text="effektive Dimension (PR)", secondary_y=False)
    fig.update_yaxes(title_text="mittlere Cosine-Sim", secondary_y=True, range=[-1, 1])
    fig.update_layout(title="Shape der Mannigfaltigkeit pro Layer",
                      xaxis_title="Layer", template="plotly_dark",
                      margin=dict(l=50, r=50, t=50, b=40),
                      legend=dict(orientation="h", y=-0.22))
    return fig

@app.callback(Output("g-spectrum", "figure"), Input("store", "data"), Input("layer", "value"))
def draw_spectrum(d, L):
    if not d:
        return go.Figure()
    sa, sb = d["shapes_a"][L], d["shapes_b"][L]
    k = max(len(sa["spectrum"]), len(sb["spectrum"]))
    idx = list(range(1, k + 1))

    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        f"Varianzspektrum @ Layer {L}", "Heatmap: erklärte Varianz (PC1) über alle Layer"))

    fig.add_trace(go.Bar(x=idx, y=sa["spectrum"], name="A",
                         marker_color="#4cc9f0"), row=1, col=1)
    fig.add_trace(go.Bar(x=idx, y=sb["spectrum"], name="B",
                         marker_color="#f72585"), row=1, col=1)

    # Heatmap: Spektrum ueber alle Layer (Satz A)
    M = np.array([s["spectrum"] + [0.0] * (k - len(s["spectrum"]))
                  for s in d["shapes_a"]]).T
    fig.add_trace(go.Heatmap(z=M, x=list(range(d["n_layers"])), y=idx,
                             colorscale="Viridis", showscale=True,
                             colorbar=dict(title="Anteil", x=1.02)), row=1, col=2)

    fig.update_xaxes(title_text="Hauptkomponente", row=1, col=1)
    fig.update_yaxes(title_text="Varianzanteil", row=1, col=1)
    fig.update_xaxes(title_text="Layer", row=1, col=2)
    fig.update_yaxes(title_text="Hauptkomponente", row=1, col=2)
    fig.update_layout(template="plotly_dark", barmode="group",
                      margin=dict(l=50, r=70, t=60, b=40),
                      legend=dict(orientation="h", y=-0.18))
    return fig

# ----------------------------------------------------------------------------
# 8) Entry Point
# ----------------------------------------------------------------------------

def main():
    port = int(os.environ.get("MANIFOLD_PORT", "8051"))
    print("=" * 72)
    print("  LLM Hidden-State Manifold Explorer")
    print("=" * 72)
    print(f"  Device      : {DEVICE.upper()}")
    print(f"  CUDA        : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print(f"  Modelle     : {len(MODELS)} ({sum(1 for v in MODELS.values() if v['tier'].startswith('CPU'))} CPU-tauglich)")
    print(f"  URL         : http://127.0.0.1:{port}")
    print("=" * 72)
    app.run(debug=False, port=port, host="127.0.0.1")

if __name__ == "__main__":
    main()
