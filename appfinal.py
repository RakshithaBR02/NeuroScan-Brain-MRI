import hashlib
import html as html_lib
import io
import json
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image
from torchvision import transforms

from model_architectures_combined import (
    MDUNet,
    SwinCTClassifier,
    HCACClassifier,
)

# =========================================================
# CONFIGURATION
# =========================================================
APP_NAME = "NeuroScan"
APP_TITLE = "NeuroScan \u2013 Brain MRI Analysis"

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
ARTIFACT_DIR = BASE_DIR / "app_artifacts"
ARTIFACT_MODEL_DIR = ARTIFACT_DIR / "models"

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
TEST_ACCURACY = 92.75
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    # Leave two cores free so the web UI stays responsive while PyTorch works in the background.
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

METADATA_CANDIDATES = [
    ARTIFACT_DIR / "hcac_metadata.json",
    BASE_DIR / "hcac_metadata.json",
]

CHECKPOINT_CANDIDATES = {
    "mdunet": [
        MODEL_DIR / "mdunet_figshare_best.pth",
        ARTIFACT_MODEL_DIR / "mdunet_figshare_best.pth",
        BASE_DIR / "mdunet_figshare_best.pth",
    ],
    "swinct": [
        MODEL_DIR / "swinct_classifier_best.pth",
        ARTIFACT_MODEL_DIR / "swinct_classifier_best.pth",
        BASE_DIR / "swinct_classifier_best.pth",
    ],
    "hcac": [
        MODEL_DIR / "hcac_classifier_corrected.pth",
        ARTIFACT_MODEL_DIR / "hcac_classifier_corrected.pth",
        BASE_DIR / "hcac_classifier_corrected.pth",
    ],
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Actual test-set classification report (1,600 test images, 400 per class).
REPORT_DF = pd.DataFrame({
    "Class": CLASS_NAMES,
    "Precision": [0.98, 0.91, 0.89, 0.95],
    "Recall": [0.79, 0.93, 1.00, 0.99],
    "F1-score": [0.87, 0.92, 0.94, 0.97],
    "Support": [400, 400, 400, 400],
})

PAGES = [
    "Dashboard",
    "MRI Analysis",
    "Segmentation & XAI",
    "Model Performance",
    "Prediction History",
]

COLOR_TEAL = "#14b8a6"
COLOR_CYAN = "#22d3ee"
COLOR_BLUE = "#60a5fa"
COLOR_MUTED_BAR = "#3b6a99"


# =========================================================
# STREAMLIT VERSION COMPATIBILITY
# =========================================================
def _version_tuple(version):
    parts = []
    for piece in str(version).split(".")[:2]:
        match = re.match(r"\d+", piece)
        parts.append(int(match.group()) if match else 0)
    while len(parts) < 2:
        parts.append(0)
    return tuple(parts)


if _version_tuple(st.__version__) >= (1, 50):
    _STRETCH_OPTIONS = [{"width": "stretch"}, {"use_container_width": True}, {}]
else:
    _STRETCH_OPTIONS = [{"use_container_width": True}, {"width": "stretch"}, {}]


def stretch_call(function, *args, **kwargs):
    """Call a Streamlit element so it fills its container on any recent version."""
    last_error = None
    for extra in _STRETCH_OPTIONS:
        try:
            return function(*args, **extra, **kwargs)
        except (TypeError, st.errors.StreamlitAPIException) as error:
            last_error = error
    raise last_error


# =========================================================
# STYLING
# =========================================================
APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --nx-bg: #071426;
    --nx-panel: #0d2038;
    --nx-panel-alt: #0a1a2e;
    --nx-border: #1e3a5a;
    --nx-border-soft: #173352;
    --nx-cyan: #22d3ee;
    --nx-teal: #14b8a6;
    --nx-text: #e8f2fc;
    --nx-body: #c3d5e8;
    --nx-muted: #8ba6c4;
}

.stApp, .stApp p, .stApp label, .stApp li, .stApp button, .stApp input,
.stApp select, .stApp textarea, h1, h2, h3, h4, [class*="nx-"] {
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
}
.stApp {
    background: linear-gradient(180deg, #0a1b31 0%, #071426 45%, #060f1d 100%);
    color: var(--nx-text);
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2.5rem; max-width: 1220px; }

/* Smooth, deliberate page transitions: outgoing content fades away first,
   then the incoming page settles into place. */
@keyframes nxPageEnter {
    0%   { opacity: 0; transform: translateY(12px) scale(.995); filter: blur(3px); }
    65%  { opacity: 1; filter: blur(0); }
    100% { opacity: 1; transform: translateY(0) scale(1); filter: blur(0); }
}
[data-testid="stMainBlockContainer"] {
    transition: opacity 240ms ease, transform 240ms ease, filter 240ms ease;
    will-change: opacity, transform;
}
[data-testid="stMainBlockContainer"].nx-page-leaving {
    opacity: 0 !important;
    transform: translateY(8px) scale(.995);
    filter: blur(2px);
    pointer-events: none;
}
[class*="st-key-nx-page-"] {
    animation: nxPageEnter 420ms cubic-bezier(.2,.75,.25,1) both;
}
@media (prefers-reduced-motion: reduce) {
    [data-testid="stMainBlockContainer"], [class*="st-key-nx-page-"] {
        animation: none !important; transition: none !important; filter: none !important;
    }
}

h1, h2, h3, h4 { color: #f4f9ff !important; letter-spacing: -.01em; }
p, label, li { color: var(--nx-body); }
[data-testid="stCaptionContainer"], small { color: var(--nx-muted) !important; }
hr { border-color: var(--nx-border-soft); margin: 1.6rem 0; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1b31 0%, #050e1a 100%);
    border-right: 1px solid var(--nx-border-soft);
}
.nx-brand {
    display: flex; align-items: center; gap: 12px;
    padding: 6px 4px 18px 4px; margin-bottom: 16px;
    border-bottom: 1px solid var(--nx-border-soft);
}
.nx-brand-logo {
    width: 38px; height: 38px; border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #0e8fa6, #14b8a6);
    color: #ffffff; font-weight: 700; font-size: 1.1rem;
}
.nx-brand-name { font-weight: 700; font-size: 1.1rem; color: #ffffff; line-height: 1.1; }
.nx-brand-sub { font-size: .78rem; color: var(--nx-muted); margin-top: 2px; }
.nx-nav-label {
    font-size: .7rem; font-weight: 600; letter-spacing: .14em;
    color: #5f87ab; margin: 4px 4px 8px 4px; text-transform: uppercase;
}
[data-testid="stSidebar"] div[role="radiogroup"] { gap: 4px; }
[data-testid="stSidebar"] div[role="radiogroup"] > label {
    width: 100%; padding: 10px 14px; border-radius: 8px;
    border-left: 3px solid transparent;
    transition: background .15s ease;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: rgba(34, 211, 238, .07); }
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: rgba(34, 211, 238, .12);
    border-left-color: var(--nx-cyan);
}
[data-testid="stSidebar"] div[role="radiogroup"] label p {
    color: var(--nx-body); font-weight: 500; font-size: .95rem;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
    color: #ffffff; font-weight: 600;
}

/* Page header */
.nx-hero {
    background: linear-gradient(120deg, #0e2744 0%, #0a1d35 100%);
    border: 1px solid var(--nx-border);
    border-left: 4px solid var(--nx-cyan);
    border-radius: 10px;
    padding: 22px 28px;
    margin-bottom: 22px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, .30);
}
.nx-kicker {
    color: var(--nx-cyan); font-size: .72rem; font-weight: 600;
    letter-spacing: .16em; text-transform: uppercase;
}
.nx-title {
    color: #ffffff; font-size: 1.85rem; font-weight: 700;
    letter-spacing: -.01em; margin: 6px 0 4px 0; line-height: 1.2;
}
.nx-sub { color: var(--nx-muted); font-size: .95rem; }

/* Section heading */
.nx-section { display: flex; align-items: center; gap: 12px; margin: 6px 0 14px 0; }
.nx-bar { width: 3px; height: 28px; border-radius: 2px; background: linear-gradient(180deg, var(--nx-cyan), var(--nx-teal)); }
.nx-section-title { color: #f4f9ff; font-size: 1.1rem; font-weight: 600; line-height: 1.2; }
.nx-section-sub { color: var(--nx-muted); font-size: .85rem; margin-top: 2px; }

/* Metric cards */
.nx-metric {
    display: flex; flex-direction: column; justify-content: center;
    min-height: 96px; padding: 18px 22px; box-sizing: border-box;
    background: linear-gradient(145deg, #0f2641, #0b1d33);
    border: 1px solid var(--nx-border);
    border-radius: 10px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, .28);
}
.nx-metric-label {
    color: var(--nx-muted); font-size: .74rem; font-weight: 600;
    letter-spacing: .09em; text-transform: uppercase;
}
.nx-metric-value { color: var(--nx-cyan); font-size: 1.65rem; font-weight: 700; line-height: 1.25; margin-top: 4px; }
.nx-metric-sub { color: var(--nx-muted); font-size: .78rem; margin-top: 2px; }

/* Workflow and category cards */
.nx-step {
    height: 150px; padding: 20px 22px; box-sizing: border-box;
    background: linear-gradient(160deg, #0f2641, #0b1d33);
    border: 1px solid var(--nx-border);
    border-top: 3px solid var(--nx-cyan);
    border-radius: 10px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, .28);
    margin-bottom: 6px;
}
.nx-step-top { margin-bottom: 8px; }
.nx-step-num { color: var(--nx-cyan); font-weight: 700; font-size: .9rem; letter-spacing: .08em; }
.nx-step-title { color: #f4f9ff; font-weight: 600; font-size: 1.02rem; margin-bottom: 5px; }
.nx-step-desc { color: var(--nx-muted); font-size: .88rem; line-height: 1.5; }
.nx-cat {
    min-height: 84px; padding: 18px 20px; box-sizing: border-box;
    background: linear-gradient(160deg, #0f2641, #0b1d33);
    border: 1px solid var(--nx-border);
    border-radius: 10px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, .28);
    display: flex; align-items: center; gap: 14px;
}
.nx-cat-dot { width: 10px; height: 10px; border-radius: 50%; flex: 0 0 auto; }
.nx-cat-name { color: #f4f9ff; font-weight: 600; font-size: 1.04rem; }
.nx-cat-sub { color: var(--nx-muted); font-size: .8rem; }

/* Image labels, info rows, statistics line */
.nx-img-label {
    font-size: .72rem; font-weight: 600; letter-spacing: .1em;
    text-transform: uppercase; color: #7fd6ee; margin-bottom: 8px;
}
[data-testid="stImage"] img { border-radius: 8px; border: 1px solid var(--nx-border); }
.nx-info-row {
    display: flex; justify-content: space-between; gap: 12px;
    padding: 10px 0; border-bottom: 1px solid var(--nx-border-soft); font-size: .9rem;
}
.nx-info-row span:first-child { color: var(--nx-muted); }
.nx-info-row span:last-child { color: var(--nx-text); font-weight: 500; text-align: right; word-break: break-all; }
.nx-stats {
    margin-top: 14px; padding: 11px 16px; border-radius: 8px;
    background: rgba(13, 32, 56, .85); border: 1px solid var(--nx-border-soft);
    color: var(--nx-body); font-size: .86rem;
}

/* Tables */
.nx-table-wrap {
    background: var(--nx-panel-alt);
    border: 1px solid var(--nx-border);
    border-radius: 10px; overflow: auto; max-height: 460px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, .28);
}
.nx-table { width: 100%; border-collapse: collapse; font-size: .92rem; }
.nx-table th {
    text-align: left; padding: 12px 16px; color: #7fd6ee;
    font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; font-weight: 600;
    background: #0d2240; position: sticky; top: 0;
    border-bottom: 1px solid var(--nx-border);
}
.nx-table td { padding: 12px 16px; color: var(--nx-body); border-bottom: 1px solid var(--nx-border-soft); }
.nx-table tr:last-child td { border-bottom: 0; }
.nx-table tbody tr:hover td { background: rgba(34, 211, 238, .05); }
.nx-table td.num, .nx-table th.num { text-align: right; }
.nx-badge {
    display: inline-block; padding: 3px 11px; border-radius: 999px;
    font-size: .78rem; font-weight: 600;
}
.nx-badge-ok { background: rgba(20,184,166,.16); color: #5eead4; border: 1px solid rgba(20,184,166,.4); }
.nx-badge-bad { background: rgba(245,158,11,.14); color: #fcd34d; border: 1px solid rgba(245,158,11,.4); }
.nx-badge-na { background: rgba(147,174,203,.10); color: #93aecb; border: 1px solid rgba(147,174,203,.25); }

/* Widgets */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(90deg, #0d9488, #0891b2);
    color: #ffffff; border: 1px solid rgba(34, 211, 238, .35);
    border-radius: 8px; padding: .6rem 1.2rem; font-weight: 600;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: #67e8f9; color: #ffffff;
    background: linear-gradient(90deg, #14b8a6, #06b6d4);
}
.stButton > button p, .stDownloadButton > button p { color: #ffffff; }
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] {
    background: rgba(10, 26, 46, .9);
    border: 1px dashed rgba(34, 211, 238, .45);
    border-radius: 10px;
}
[data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span { color: var(--nx-muted); }
[data-testid="stFileUploader"] button {
    background: rgba(20,184,166,.16); color: var(--nx-text);
    border: 1px solid rgba(34,211,238,.4);
}
div[data-baseweb="select"] > div {
    background: var(--nx-panel); border-color: var(--nx-border); color: var(--nx-text);
}
div[data-baseweb="select"] span, div[data-baseweb="select"] div { color: var(--nx-text); }
div[data-baseweb="popover"] ul { background: var(--nx-panel); }
div[data-baseweb="popover"] li { color: var(--nx-text); }
[data-testid="stExpander"] {
    background: rgba(10, 26, 46, .85);
    border: 1px solid var(--nx-border-soft);
    border-radius: 8px;
}
[data-testid="stExpander"] summary p { color: var(--nx-body); font-size: .9rem; font-weight: 500; }
[data-testid="stAlert"] { border-radius: 10px; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--nx-border-soft); }
.stTabs [data-baseweb="tab"] { color: var(--nx-muted); font-weight: 500; padding: 10px 18px; }
.stTabs [aria-selected="true"] { color: var(--nx-cyan); font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--nx-cyan); }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: #060f1d; }
::-webkit-scrollbar-thumb { background: #1b3a5c; border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: #26507a; }
::selection { background: rgba(34,211,238,.35); color: #ffffff; }

@media (max-width: 768px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .nx-title { font-size: 1.45rem; }
    .nx-hero { padding: 18px; }
    .nx-step { height: auto; min-height: 0; }
}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)


# =========================================================
# GENERIC UI HELPERS
# =========================================================
def esc(value):
    """HTML-escape text (and neutralise '$' so Streamlit does not read it as LaTeX)."""
    return html_lib.escape(str(value)).replace("$", "&#36;")


def render_html(markup):
    """Render HTML without markdown treating indented lines as code blocks."""
    flattened = " ".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(flattened, unsafe_allow_html=True)


def page_header(kicker, title, subtitle=None):
    sub = f'<div class="nx-sub">{esc(subtitle)}</div>' if subtitle else ""
    render_html(
        f"""<div class="nx-hero">
        <div class="nx-kicker">{esc(kicker)}</div>
        <div class="nx-title">{esc(title)}</div>
        {sub}
        </div>"""
    )


def section(title, subtitle=None):
    sub = f'<div class="nx-section-sub">{esc(subtitle)}</div>' if subtitle else ""
    render_html(
        f"""<div class="nx-section"><span class="nx-bar"></span>
        <div><div class="nx-section-title">{esc(title)}</div>{sub}</div></div>"""
    )


def metric_card(label, value, sub=None):
    sub_html = f'<div class="nx-metric-sub">{esc(sub)}</div>' if sub else ""
    render_html(
        f"""<div class="nx-metric">
        <div class="nx-metric-label">{esc(label)}</div>
        <div class="nx-metric-value">{esc(value)}</div>{sub_html}</div>"""
    )


def image_label(text):
    render_html(f'<div class="nx-img-label">{esc(text)}</div>')


def show_image(image, caption=None):
    return stretch_call(st.image, image, caption=caption)


def badge(value):
    text = str(value)
    if text == "Match":
        css = "nx-badge-ok"
    elif text == "Mismatch":
        css = "nx-badge-bad"
    else:
        css = "nx-badge-na"
    return f'<span class="nx-badge {css}">{esc(text)}</span>'


def render_table(df, badge_col=None, numeric_cols=()):
    head = "".join(
        f'<th class="num">{esc(c)}</th>' if c in numeric_cols else f"<th>{esc(c)}</th>"
        for c in df.columns
    )
    body = []
    for _, row in df.iterrows():
        cells = []
        for column in df.columns:
            value = row[column]
            text = badge(value) if column == badge_col else esc(value)
            css = ' class="num"' if column in numeric_cols else ""
            cells.append(f"<td{css}>{text}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    render_html(
        '<div class="nx-table-wrap"><table class="nx-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


# =========================================================
# CHARTS
# =========================================================
def style_chart(chart, height):
    return (
        chart.properties(height=height, background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#a9c4df",
            titleColor="#a9c4df",
            gridColor="rgba(120,170,220,0.14)",
            domainColor="rgba(120,170,220,0.30)",
            tickColor="rgba(120,170,220,0.30)",
            labelFontSize=12,
            titleFontSize=12,
            labelFont="Inter",
            titleFont="Inter",
        )
    )


def capsule_chart(scores, predicted_class):
    df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Score": [float(s) for s in scores],
        "Top": [name == predicted_class for name in CLASS_NAMES],
    })
    base = alt.Chart(df).encode(
        y=alt.Y("Class:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220))
    )
    bars = base.mark_bar(
        cornerRadiusTopRight=6, cornerRadiusBottomRight=6, size=26
    ).encode(
        x=alt.X(
            "Score:Q",
            title="Capsule score",
            scale=alt.Scale(domain=[0, 1.1], nice=False),
            axis=alt.Axis(values=[0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        ),
        color=alt.condition("datum.Top", alt.value(COLOR_CYAN), alt.value(COLOR_MUTED_BAR)),
        tooltip=[alt.Tooltip("Class:N"), alt.Tooltip("Score:Q", format=".4f")],
    )
    labels = base.mark_text(
        align="left", dx=6, color="#e8f2fc", fontWeight=600, fontSize=12
    ).encode(
        x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1.1], nice=False)),
        text=alt.Text("Score:Q", format=".4f"),
    )
    return style_chart(alt.layer(bars, labels), height=230)


def metric_bar_chart(metric, color):
    df = REPORT_DF[["Class", metric]].rename(columns={metric: "Value"})
    base = alt.Chart(df).encode(
        x=alt.X(
            "Class:N",
            sort=CLASS_NAMES,
            title=None,
            axis=alt.Axis(labelAngle=0, labelOverlap="greedy"),
        )
    )
    bars = base.mark_bar(
        cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color=color, size=34
    ).encode(
        y=alt.Y(
            "Value:Q",
            title=None,
            scale=alt.Scale(domain=[0, 1.1], nice=False),
            axis=alt.Axis(values=[0, 0.25, 0.5, 0.75, 1.0], format=".2f"),
        ),
        tooltip=[alt.Tooltip("Class:N"), alt.Tooltip("Value:Q", format=".2f", title=metric)],
    )
    labels = base.mark_text(
        dy=-8, color="#e8f2fc", fontWeight=700, fontSize=12
    ).encode(
        y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 1.1], nice=False)),
        text=alt.Text("Value:Q", format=".2f"),
    )
    return style_chart(alt.layer(bars, labels), height=240)


def show_chart(chart):
    return stretch_call(st.altair_chart, chart, theme=None)


# =========================================================
# FILE / CHECKPOINT HELPERS (unchanged pipeline)
# =========================================================
def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_weights(model, path):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    state = extract_state_dict(checkpoint)
    # Remove common DataParallel prefix if present.
    if isinstance(state, dict) and state:
        state = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state.items()
        }
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    missing = []
    resolved = {}
    for name, paths in CHECKPOINT_CANDIDATES.items():
        resolved[name] = first_existing(paths)
        if resolved[name] is None:
            missing.append(name)

    if missing:
        expected = {
            "mdunet": "models/mdunet_figshare_best.pth",
            "swinct": "models/swinct_classifier_best.pth",
            "hcac": "models/hcac_classifier_corrected.pth",
        }
        details = "\n".join(f"- {name}: {expected[name]}" for name in missing)
        raise FileNotFoundError(
            "Missing model checkpoint(s). Place the required files in the models folder:\n"
            + details
        )

    mdunet = load_weights(MDUNet(), resolved["mdunet"])
    swinct = load_weights(SwinCTClassifier(num_classes=4), resolved["swinct"])
    hcac = load_weights(HCACClassifier(), resolved["hcac"])
    return mdunet, swinct, hcac


@st.cache_data
def load_metadata():
    path = first_existing(METADATA_CANDIDATES)
    if path is None:
        raise FileNotFoundError(
            "hcac_metadata.json was not found. Expected it inside app_artifacts/."
        )
    with open(path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    required = ["scaler_mean", "scaler_scale", "selected_feature_indices"]
    missing = [key for key in required if key not in metadata]
    if missing:
        raise KeyError(f"Metadata is missing: {', '.join(missing)}")
    return metadata


@st.cache_resource(show_spinner="Loading models...")
def load_pipeline():
    """Models, metadata and the differentiable score function, built once per server process.

    Only reusable model objects are cached here. Predictions and explanations are never cached.
    """
    models = load_models()
    metadata = load_metadata()
    return models, metadata, build_score_function(models, metadata)


def xai_package_status():
    """Which explanation packages are importable by the interpreter running this app."""
    import importlib.util

    return {
        name: importlib.util.find_spec(name) is not None
        for name in ("lime", "shap", "skimage", "scipy", "matplotlib")
    }


# =========================================================
# MRI INFERENCE (unchanged pipeline)
# =========================================================
def analyze_mri(image, score_function):
    """Preprocess the MRI and classify it. One forward pass, no gradients.

    Uses the same steps as training: Resize(224, 224) -> ToTensor -> SwinCT features ->
    scaler -> OOA-selected features -> HCAC capsule scores. MDU-Net runs separately in
    segment_mri() so the prediction can be shown before segmentation starts.
    """
    image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        predictions = score_function(image_tensor)
        capsule_scores = predictions[0].detach().cpu().numpy()

    if len(capsule_scores) != len(CLASS_NAMES) or not np.isfinite(capsule_scores).all():
        raise ValueError("HCAC returned invalid class scores. Check the classifier checkpoint.")

    predicted_index = int(np.argmax(capsule_scores))
    return {
        "prediction": CLASS_NAMES[predicted_index],
        "prediction_index": predicted_index,
        "prediction_strength": float(capsule_scores[predicted_index]),
        "capsule_scores": capsule_scores,
        "mask": None,
    }


def segment_mri(image, models):
    """MDU-Net segmentation probability map for the uploaded MRI (no gradients)."""
    mdunet = models[0]
    image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        output = mdunet(image_tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return torch.sigmoid(output)[0, 0].detach().cpu().numpy()


# =========================================================
# SEGMENTATION AND EXPLAINABILITY
# Everything below is computed from the MRI that was just analysed.
# No saved example images are read or displayed.
# =========================================================
# Explanation detail presets. LIME and SHAP evaluate the full SwinCT -> OOA -> HCAC pipeline
# once per perturbed image, so these numbers are the main driver of analysis time.
# Fixed lightweight settings: no user-facing settings panel. These are deliberately
# labelled as quick, exploratory explanations rather than clinically validated maps.
XAI_PRESETS = {
    "Fast": {"lime_samples": 80, "lime_segments": 30, "shap_evals": 64},
}
DEFAULT_XAI_PRESET = "Fast"
LIME_TOP_REGIONS = 5
XAI_BATCH_SIZE = 32 if DEVICE.type == "cuda" else 16

# Grad-CAM display settings.
GRADCAM_LAYER_NAME = None   # None = auto-detect; or set e.g. "7.conv2" (name inside cnn_branch.cnn)
GRADCAM_MAX_SIDE = 1024     # longest side of the displayed MRI / heatmap
GRADCAM_THRESHOLD = 0.30    # normalised responses below this stay fully transparent
GRADCAM_MAX_ALPHA = 0.65    # opacity of the strongest response
GRADCAM_COLORMAP = "turbo"
_GRADCAM_LOCK = threading.Lock()

STAGE_LABELS = [
    "Preprocessing MRI and HCAC classification",
    "MDU-Net segmentation",
    "Grad-CAM",
    "LIME superpixel explanation",
    "SHAP explanation",
]
PANEL_KEYS = ("segmentation", "gradcam", "lime", "shap")
PANEL_TITLES = ("Segmentation", "Grad-CAM", "LIME", "SHAP")

_COLORMAP_STOPS = np.array([0.0, 0.30, 0.60, 0.85, 1.0])
_COLORMAP_RGB = np.array([
    [8, 20, 42],
    [10, 92, 140],
    [20, 184, 166],
    [103, 232, 249],
    [255, 255, 255],
], dtype=np.float32)


def apply_colormap(normalized):
    channels = [
        np.interp(normalized, _COLORMAP_STOPS, _COLORMAP_RGB[:, c]) for c in range(3)
    ]
    return np.stack(channels, axis=-1).astype(np.float32)


def colorize(values, colormap_name):
    """Map values in [0, 1] to RGB uint8 with a matplotlib colormap."""
    from matplotlib import colormaps

    return (colormaps[colormap_name](np.clip(values, 0.0, 1.0))[..., :3] * 255).astype(np.uint8)


def display_array(image, max_side=GRADCAM_MAX_SIDE):
    """The uploaded MRI at (or capped to) its original resolution, as RGB uint8."""
    rgb = image.convert("RGB")
    scale = max(rgb.size) / float(max_side)
    if scale > 1.0:
        rgb = rgb.resize((round(rgb.width / scale), round(rgb.height / scale)), Image.LANCZOS)
    return np.asarray(rgb, dtype=np.uint8)


def gradcam_overlay(base, heat, colormap_name=GRADCAM_COLORMAP):
    """Blend a Grad-CAM map in [0, 1] over the MRI without hiding anatomy.

    Responses below GRADCAM_THRESHOLD are fully transparent, so most of the original MRI is
    untouched. Above it, opacity rises smoothly with the response.
    """
    heat = np.clip(np.asarray(heat, dtype=np.float32), 0.0, 1.0)
    colored = colorize(heat, colormap_name).astype(np.float32)
    ramp = np.clip((heat - GRADCAM_THRESHOLD) / (1.0 - GRADCAM_THRESHOLD), 0.0, 1.0)
    alpha = (GRADCAM_MAX_ALPHA * ramp * ramp * (3.0 - 2.0 * ramp))[..., None]  # smoothstep
    blended = base.astype(np.float32) * (1.0 - alpha) + colored * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def gradcam_legend_html(colormap_name=GRADCAM_COLORMAP):
    from matplotlib import colormaps

    cmap = colormaps[colormap_name]
    stops = ", ".join(
        "rgb({},{},{}) {:.0f}%".format(
            *(int(c * 255) for c in cmap(t)[:3]), (t - GRADCAM_THRESHOLD) / (1 - GRADCAM_THRESHOLD) * 100
        )
        for t in np.linspace(GRADCAM_THRESHOLD, 1.0, 8)
    )
    return (
        '<div style="margin-top:6px">'
        f'<div style="height:8px;border-radius:4px;background:linear-gradient(90deg,{stops})"></div>'
        '<div style="display:flex;justify-content:space-between;font-size:0.75rem;opacity:0.8">'
        "<span>Moderate influence</span><span>Strongest influence</span></div></div>"
    )


def overlay_signed(base, values):
    """Blend signed attributions: red raises the predicted-class score, blue lowers it."""
    values = np.asarray(values, dtype=np.float32)
    limit = float(np.abs(values).max())
    if not np.isfinite(limit) or limit <= 0:
        return base
    scaled = np.clip(values / limit, -1.0, 1.0)
    colored = colorize((scaled + 1.0) / 2.0, "RdBu_r").astype(np.float32)
    alpha = (0.65 * np.abs(scaled) ** 0.7)[..., None]
    blended = base.astype(np.float32) * (1.0 - alpha) + colored * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------
# Model input and differentiable classification pipeline
# ---------------------------------------------------------
def model_input_array(image):
    """224 x 224 RGB uint8 array holding exactly the pixels the models receive."""
    resized = transforms.Resize((224, 224))(image.convert("RGB"))
    return np.asarray(resized, dtype=np.uint8)


def array_to_tensor(batch):
    """NHWC pixels (0-255) to an NCHW tensor in [0, 1], the same scaling as ToTensor()."""
    tensor = torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32))
    return (tensor.permute(0, 3, 1, 2) / 255.0).to(DEVICE)


def build_score_function(models, metadata):
    """Image tensor -> HCAC capsule scores, using the same steps as analyze_mri()."""
    _, swinct, hcac = models
    scaler_mean = torch.as_tensor(
        metadata["scaler_mean"], dtype=torch.float32, device=DEVICE
    ).reshape(1, -1)
    scaler_scale = torch.as_tensor(
        metadata["scaler_scale"], dtype=torch.float32, device=DEVICE
    ).reshape(1, -1)
    selected_indices = torch.as_tensor(
        metadata["selected_feature_indices"], dtype=torch.long, device=DEVICE
    )

    def score_function(image_tensor):
        features = swinct.feature_extractor(image_tensor)
        if isinstance(features, (tuple, list)):
            features = features[0]
        features = features.reshape(features.shape[0], -1)
        standardized = (features - scaler_mean) / torch.clamp(scaler_scale, min=1e-8)
        predictions, _, _ = hcac(standardized[:, selected_indices])
        return predictions

    return score_function


def build_predict_function(score_function, on_batch=None):
    """NHWC pixel batches -> NumPy capsule scores, for LIME and SHAP (no gradients needed).

    on_batch(n) is called after each chunk so the UI can report real progress.
    """

    def predict(batch):
        batch = np.asarray(batch)
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(batch), XAI_BATCH_SIZE):
                chunk = array_to_tensor(batch[start:start + XAI_BATCH_SIZE])
                outputs.append(score_function(chunk).detach().cpu().numpy())
                if on_batch is not None:
                    on_batch(len(chunk))
        return np.concatenate(outputs, axis=0)

    return predict


# ---------------------------------------------------------
# MDU-Net segmentation output
# ---------------------------------------------------------
def inspect_segmentation(probability):
    """Summarise the raw MDU-Net output and decide whether a binary mask is meaningful.

    The mask is only shown when the output is confident (few pixels near the decision
    boundary), covers a plausible area and forms one compact region. These are sanity
    checks on the output, not a clinical validation of the mask.
    """
    probability = np.squeeze(np.asarray(probability, dtype=np.float32))
    binary = probability >= 0.5
    positive_fraction = float(binary.mean())
    uncertain_fraction = float(((probability > 0.35) & (probability < 0.65)).mean())
    peak = float(probability.max())

    largest_share = 0.0
    if binary.any():
        from scipy import ndimage

        labelled, _ = ndimage.label(binary)
        sizes = np.bincount(labelled.ravel())[1:]
        largest_share = float(sizes.max() / sizes.sum())

    reason = None
    if peak < 0.5:
        reason = "no pixel reaches the 0.50 threshold"
    elif positive_fraction < 0.002:
        reason = "the thresholded region is negligibly small"
    elif positive_fraction > 0.5:
        reason = "the thresholded region covers most of the image"
    elif uncertain_fraction > 0.15:
        reason = "too many pixels lie close to the decision boundary"
    elif largest_share < 0.6:
        reason = "the thresholded region is fragmented"

    return {
        "min": float(probability.min()),
        "max": peak,
        "mean": float(probability.mean()),
        "positive_fraction": positive_fraction,
        "mask_valid": reason is None,
        "reason": reason,
    }


def segmentation_views(base, probability, info):
    """Return (base, probability map, overlay, map label, overlay label) for display."""
    probability = np.squeeze(np.asarray(probability, dtype=np.float32))
    height, width = probability.shape
    if base.shape[:2] != (height, width):
        base = np.asarray(Image.fromarray(base).resize((width, height), Image.BILINEAR))

    span = info["max"] - info["min"]
    stretched = span < 0.25
    if stretched:
        normalized = (probability - info["min"]) / max(span, 1e-6)
    else:
        normalized = np.clip(probability, 0.0, 1.0)
    response = apply_colormap(normalized).astype(np.uint8)

    if info["mask_valid"]:
        from scipy import ndimage

        binary = probability >= 0.5
        overlay = base.astype(np.float32)
        overlay[binary] = overlay[binary] * 0.65 + np.array([34, 211, 238], dtype=np.float32) * 0.35
        contour = binary & ~ndimage.binary_erosion(binary, iterations=2)
        overlay[contour] = [103, 232, 249]
        overlay = overlay.astype(np.uint8)
        overlay_label = "Mask Overlay (threshold 0.50)"
    else:
        alpha = (0.10 + 0.60 * normalized)[..., None]
        overlay = np.clip(
            base.astype(np.float32) * (1.0 - alpha) + response.astype(np.float32) * alpha, 0, 255
        ).astype(np.uint8)
        overlay_label = "Response Overlay"

    map_label = "Response Map (contrast-stretched)" if stretched else "Segmentation Probability Map"
    return base, response, overlay, map_label, overlay_label


def estimate_brain_mask(image_array):
    """Estimate the visible MRI foreground to suppress black canvas/background artifacts.

    This is a display/perturbation mask only; it is not a tumor segmentation and must not
    be interpreted as an anatomical or clinical contour.
    """
    from scipy import ndimage

    rgb = np.asarray(image_array, dtype=np.float32)
    gray = rgb.mean(axis=2) if rgb.ndim == 3 else rgb
    positive = gray[gray > 0]
    if positive.size < 16 or float(gray.max()) <= 0:
        return np.ones(gray.shape, dtype=bool)
    threshold = max(6.0, float(np.percentile(positive, 35)) * 0.35, float(gray.max()) * 0.025)
    candidate = gray > threshold
    candidate = ndimage.binary_closing(candidate, iterations=2)
    labels, count = ndimage.label(candidate)
    if count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        candidate = labels == int(np.argmax(sizes))
        candidate = ndimage.binary_fill_holes(candidate)
        candidate = ndimage.binary_dilation(candidate, iterations=2)
    fraction = float(candidate.mean())
    # If thresholding is implausible, fail open rather than erase most of the image.
    if fraction < 0.01 or fraction > 0.85:
        return np.ones(gray.shape, dtype=bool)
    return candidate.astype(bool)


# ---------------------------------------------------------
# Grad-CAM (SwinCT CNN branch, predicted-class capsule score)
# ---------------------------------------------------------
def _gradcam_search_root(swinct):
    """The CNN branch of the SwinCT feature extractor (the part Grad-CAM can explain)."""
    extractor = swinct.feature_extractor
    branch = getattr(extractor, "cnn_branch", None)
    if branch is None:
        raise RuntimeError("SwinCT has no `cnn_branch`; Grad-CAM cannot locate a convolutional layer.")
    cnn = getattr(branch, "cnn", None)
    return cnn if cnn is not None else branch


def compute_gradcam(image_array, score_function, swinct, target_index, output_size, expected_scores=None):
    """Grad-CAM for the predicted class on the MRI that was just classified.

    * Every leaf module of the SwinCT CNN branch is hooked, and the 4-D feature maps it produces
      are recorded in execution order. Candidates are taken from the END of the branch, so the
      first candidate is the final post-activation feature map before pooling (a raw Conv2d output
      is pre-BatchNorm/ReLU and can be negative everywhere, which gave an empty map before).
    * One backward pass gives gradients for several candidate layers; the last layer that shows
      positive evidence is used, and its name is reported.
    * Gradients are taken from the predicted class's capsule score through the same SwinCT ->
      scaler -> OOA selection -> HCAC path used for classification. Never run under inference_mode.
    """
    import torch.nn.functional as F

    root = _gradcam_search_root(swinct)
    leaves = [(name, module) for name, module in root.named_modules() if not list(module.children())]
    if GRADCAM_LAYER_NAME:
        leaves = [(name, module) for name, module in leaves if name == GRADCAM_LAYER_NAME]
        if not leaves:
            raise ValueError(f"GRADCAM_LAYER_NAME '{GRADCAM_LAYER_NAME}' was not found in the CNN branch.")
    if not leaves:
        raise RuntimeError("The SwinCT CNN branch has no layers Grad-CAM can hook.")

    owner = threading.get_ident()
    executed = []

    def make_hook(name):
        def hook(module, inputs, output):
            if threading.get_ident() == owner and torch.is_tensor(output) and output.dim() == 4:
                executed.append((name or type(module).__name__, output))
        return hook

    handles = [module.register_forward_hook(make_hook(name)) for name, module in leaves]
    try:
        with _GRADCAM_LOCK, torch.enable_grad():
            tensor = array_to_tensor(image_array[None]).requires_grad_(True)
            scores = score_function(tensor)
            if expected_scores is not None:
                reproduced = scores.detach().cpu().numpy()[0]
                if not np.allclose(reproduced, expected_scores, atol=5e-3):
                    raise ValueError("the Grad-CAM forward pass did not reproduce the model prediction")

            candidates, seen = [], set()
            for name, output in reversed(executed):
                height, width = output.shape[-2:]
                if height * width < 4 or not output.requires_grad or id(output) in seen:
                    continue
                seen.add(id(output))
                candidates.append((name, output))
                if len(candidates) == 6:
                    break
            if not candidates:
                raise RuntimeError(
                    "No spatial feature map (at least 2 x 2) from the CNN branch is connected to the score."
                )
            gradients = torch.autograd.grad(
                scores[0, target_index], [output for _, output in candidates],
                allow_unused=True, retain_graph=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    if all(g is None or float(g.abs().max()) == 0.0 for g in gradients):
        return {
            "status": "unsupported",
            "message": (
                "Grad-CAM is not reliable for this model: no gradient reaches the SwinCT CNN branch "
                "from the predicted HCAC score. This happens when the OOA-selected features do not "
                "depend on the CNN branch. LIME and SHAP, which only use the input pixels, remain valid."
            ),
        }

    notes, chosen = [], None
    for (name, output), gradient in zip(candidates, gradients):
        height, width = (int(v) for v in output.shape[-2:])
        if gradient is None:
            notes.append(f"{name} ({width}x{height}): no gradient")
            continue
        activation = output.detach()[0]
        weights = gradient.detach()[0].mean(dim=(1, 2), keepdim=True)   # channel importance
        cam = torch.relu((weights * activation).sum(dim=0))              # positive evidence only
        peak = float(cam.max()) if torch.isfinite(cam).all() else float("nan")
        notes.append(f"{name} ({width}x{height}): max {peak:.3e}")
        if np.isfinite(peak) and peak > 0:
            chosen = (name, cam)
            break

    if chosen is None:
        return {
            "status": "empty",
            "message": (
                "Grad-CAM found no positive evidence for the predicted class in this image, "
                "on any of the last feature maps of the CNN branch."
            ),
            "detail": "Layers checked (last first):\n" + "\n".join(notes),
        }

    layer_name, cam = chosen
    map_shape = tuple(int(v) for v in cam.shape)
    cam = F.interpolate(cam[None, None], size=tuple(output_size), mode="bicubic", align_corners=False)[0, 0]
    cam = cam.clamp_min(0).cpu().numpy().astype(np.float32)
    brain_mask = estimate_brain_mask(image_array)
    if brain_mask.shape != cam.shape:
        brain_mask = np.asarray(
            Image.fromarray(brain_mask.astype(np.uint8) * 255).resize(
                (cam.shape[1], cam.shape[0]), Image.NEAREST
            ), dtype=np.uint8
        ) > 0
    inside = cam[brain_mask]
    high = float(np.percentile(inside, 99.5)) if inside.size else 0.0
    if not np.isfinite(high) or high <= 1e-12:
        return {
            "status": "empty",
            "message": "Grad-CAM produced no measurable positive response inside the estimated MRI foreground. The map is not displayed as a reliable explanation.",
        }
    heat = np.clip(cam / high, 0.0, 1.0)
    heat[~brain_mask] = 0.0
    return {
        "status": "ok",
        "heatmap": heat,
        "layer": layer_name,
        "map_shape": map_shape,
        "coverage": float((heat >= 0.5).mean()),
        "detail": "Layers checked (last first):\n" + "\n".join(notes),
    }


# ---------------------------------------------------------
# LIME
# ---------------------------------------------------------
def compute_lime(image_array, predict_function, target_index, num_samples, num_segments):
    import inspect

    from lime import lime_image
    from skimage.segmentation import slic

    brain_mask = estimate_brain_mask(image_array)

    def segment(image):
        # Keep the black canvas as one fixed background superpixel (label 0); only
        # subdivide the estimated MRI foreground. This prevents LIME selecting borders.
        labels = slic(
            image, n_segments=num_segments, compactness=10, sigma=1,
            start_label=1, mask=brain_mask, channel_axis=-1,
        )
        labels = np.asarray(labels, dtype=np.int32)
        labels[~brain_mask] = 0
        return labels

    explainer = lime_image.LimeImageExplainer(random_state=0)
    options = dict(
        labels=(target_index,),
        top_labels=None,
        hide_color=0,
        num_samples=num_samples,
        batch_size=XAI_BATCH_SIZE,
        segmentation_fn=segment,
        random_seed=0,
    )
    if "progress_bar" in inspect.signature(explainer.explain_instance).parameters:
        options["progress_bar"] = False
    explanation = explainer.explain_instance(image_array, predict_function, **options)

    segments = explanation.segments
    weights = explanation.local_exp[target_index]
    weight_map = np.zeros(segments.shape, dtype=np.float32)
    for segment_id, weight in weights:
        if int(segment_id) > 0:
            weight_map[segments == segment_id] = weight

    supporting = sorted(
        (item for item in weights if int(item[0]) > 0 and item[1] > 0),
        key=lambda item: item[1], reverse=True
    )[:LIME_TOP_REGIONS]
    if not supporting:
        return {
            "status": "empty",
            "message": "LIME found no regions that support the predicted class in this image.",
        }

    score = explanation.score
    if isinstance(score, dict):
        score = score.get(target_index)
    fit = float(score) if score is not None and np.isfinite(score) else None
    return {
        "status": "ok",
        "weight_map": weight_map,
        "support_mask": np.isin(segments, [segment_id for segment_id, _ in supporting]) & brain_mask,
        "fit": fit,
        "samples": int(num_samples),
        "segments": int(len(np.unique(segments))),
    }


def lime_support_overlay(base, support_mask):
    from scipy import ndimage

    overlay = base.astype(np.float32)
    overlay[support_mask] = overlay[support_mask] * 0.65 + np.array([20, 184, 166], dtype=np.float32) * 0.35
    edge = support_mask & ~ndimage.binary_erosion(support_mask, iterations=1)
    overlay[edge] = [103, 232, 249]
    return overlay.astype(np.uint8)


# ---------------------------------------------------------
# SHAP (Partition explainer, black masking, full classification pipeline)
# ---------------------------------------------------------
def compute_shap(image_array, predict_function, target_index, max_evals, expected_score=None):
    """Partition SHAP over image regions for the full pixels -> SwinCT -> OOA -> HCAC pipeline.

    The explained function is the real classifier, so attributions describe how image regions
    change the predicted capsule score. Masked regions are replaced by black (value 0).
    """
    import shap

    masker = shap.maskers.Image(0, image_array.shape)
    explainer = shap.Explainer(
        predict_function, masker, algorithm="partition", output_names=CLASS_NAMES
    )
    explanation = explainer(
        image_array[None].astype(np.float32),
        max_evals=max_evals,
        batch_size=XAI_BATCH_SIZE,
        silent=True,
    )
    values = np.asarray(explanation.values)[0]
    class_values = values[..., target_index]
    attribution = class_values.sum(axis=-1).astype(np.float32)
    brain_mask = estimate_brain_mask(image_array)
    if brain_mask.shape != attribution.shape:
        brain_mask = np.asarray(
            Image.fromarray(brain_mask.astype(np.uint8) * 255).resize(
                (attribution.shape[1], attribution.shape[0]), Image.NEAREST
            ), dtype=np.uint8
        ) > 0
    attribution[~brain_mask] = 0.0
    if not np.isfinite(attribution).all() or float(np.abs(attribution).max()) <= 0:
        return {
            "status": "empty",
            "message": "SHAP did not find any measurable attribution for this image.",
        }

    gap = None
    try:
        base_value = float(np.asarray(explanation.base_values)[0][target_index])
        if expected_score is not None:
            gap = abs(base_value + float(class_values.sum()) - float(expected_score))
    except Exception:
        gap = None
    return {"status": "ok", "values": attribution, "evals": int(max_evals), "additivity_gap": gap}


# ---------------------------------------------------------
# Orchestration: runs once per analysis, for that exact image
# ---------------------------------------------------------
_OPTIONAL_PACKAGES = {"lime": "lime", "shap": "shap", "skimage": "scikit-image", "scipy": "scipy"}


def _failure(method, error):
    text = str(error)
    if isinstance(error, ImportError):
        package = _OPTIONAL_PACKAGES.get(
            getattr(error, "name", "") or "", getattr(error, "name", "") or method.lower()
        )
        message = (
            f"{method} needs the `{package}` package, which is not installed for this Python "
            f"interpreter ({sys.executable}). Run: \"{sys.executable}\" -m pip install {package}  "
            "then restart the app."
        )
    elif "Inference tensors" in text:
        message = (
            f"{method} could not be generated: the model cached tensors while running under "
            "torch.inference_mode(), and autograd cannot reuse them. In analyze_mri(), replace "
            "torch.inference_mode() with torch.no_grad() to fix this."
        )
    else:
        message = f"{method} could not be generated for this image: {error}"
    return {"status": "error", "message": message}


# ---------------------------------------------------------
# Staged orchestration: classify -> show prediction -> segmentation -> Grad-CAM -> LIME -> SHAP
# Each stage stores its result in st.session_state as soon as it finishes, so a Streamlit rerun
# never repeats finished work and never loses the prediction.
# ---------------------------------------------------------
class JobCancelled(Exception):
    """Raised inside a background job when a newer analysis replaces it."""


def _batch_counter(job, total):
    state = {"done": 0}

    def on_batch(count):
        if job.get("cancel"):
            raise JobCancelled()
        state["done"] += count
        job["fraction"] = min(state["done"] / max(total, 1), 1.0)

    return on_batch


def pending_stages(latest, preset_name, retry_errors=False):
    """Stages that still have to run for the stored analysis under the selected settings."""
    if latest is None or latest.get("result") is None:
        return list(PANEL_KEYS)
    pending = []
    if latest["result"].get("mask") is None:
        pending.append("segmentation")
    xai = latest.get("xai") or {}
    for key in ("gradcam", "lime", "shap"):
        entry = xai.get(key)
        if entry is None:
            pending.append(key)
        elif retry_errors and entry.get("status") == "error":
            pending.append(key)
        elif key in ("lime", "shap") and entry.get("preset") != preset_name:
            pending.append(key)
    return pending


def _job_worker(job, latest, image, models, score_function, preset_name, todo):
    """Segmentation, Grad-CAM, LIME and SHAP, in a background thread.

    It never touches st.* or st.session_state. Streamlit only interrupts the script thread, so
    switching pages or rerunning the script cannot freeze or kill this work.
    """
    preset = XAI_PRESETS[preset_name]
    try:
        target_index = int(latest["result"]["prediction_index"])
        capsule_scores = latest["result"]["capsule_scores"]
        image_array = model_input_array(image)
        display_base = display_array(image)

        def begin(index):
            job["stage"] = index
            job["fraction"] = 0.0

        if "segmentation" in todo and not job["cancel"]:
            begin(1)
            started = time.perf_counter()
            try:
                latest["result"]["mask"] = segment_mri(image, models)
                latest.pop("segmentation_error", None)
            except Exception as error:
                latest["segmentation_error"] = str(error)
            latest["timings"]["segmentation"] = time.perf_counter() - started

        package_status = xai_package_status()
        needed_packages = {"lime": ("lime", "skimage", "scipy"), "shap": ("shap",)}
        jobs = [
            ("gradcam", 2, "Grad-CAM", lambda: compute_gradcam(
                image_array, score_function, models[1], target_index,
                output_size=display_base.shape[:2], expected_scores=capsule_scores)),
            ("lime", 3, "LIME", lambda: compute_lime(
                image_array,
                build_predict_function(score_function, _batch_counter(job, preset["lime_samples"])),
                target_index, preset["lime_samples"], preset["lime_segments"])),
            ("shap", 4, "SHAP", lambda: compute_shap(
                image_array,
                build_predict_function(score_function, _batch_counter(job, preset["shap_evals"])),
                target_index, preset["shap_evals"], float(capsule_scores[target_index]))),
        ]
        for key, stage_index, label, run in jobs:
            if key not in todo:
                continue
            if job["cancel"]:
                break
            begin(stage_index)
            started = time.perf_counter()
            try:
                missing = [n for n in needed_packages.get(key, ()) if not package_status.get(n, False)]
                if missing:
                    raise ImportError(f"{missing[0]} is not installed", name=missing[0])
                entry = run()
            except JobCancelled:
                break
            except Exception as error:
                entry = _failure(label, error)
                entry["detail"] = traceback.format_exc()
            entry["preset"] = preset_name
            latest["xai"][key] = entry
            latest["timings"][key] = time.perf_counter() - started
    except Exception as error:
        job["error"] = str(error)
    finally:
        job["done"] = True


def start_analysis(image, filename, signature, reference, preset_name):
    """Classify now (about one forward pass, shown immediately) and start the rest in the background."""
    existing = st.session_state.get("latest")
    same_image = (
        existing is not None
        and existing.get("signature") == signature
        and existing.get("result") is not None
    )
    if same_image and _job_running(signature):
        st.info("This image is already being analysed. The results will appear below as each step finishes.")
        return
    if same_image and not pending_stages(existing, preset_name) and not any(
        (existing.get("xai") or {}).get(k, {}).get("status") == "error" for k in ("gradcam", "lime", "shap")
    ):
        st.info("Results for this image are already computed, so nothing was re-run.")
        return

    previous = st.session_state.get("job")
    if previous and not previous.get("done"):
        previous["cancel"] = True          # a different image replaces the running analysis

    try:
        with st.spinner("Preprocessing MRI and running HCAC classification..."):
            models, metadata, score_function = load_pipeline()
            if same_image:
                latest = existing
            else:
                started = time.perf_counter()
                result = analyze_mri(image, score_function)
                latest = {
                    "signature": signature,
                    "filename": filename,
                    "image": image,
                    "result": result,
                    "xai": {},
                    "reference": reference,
                    "timings": {"classification": time.perf_counter() - started},
                }
                st.session_state["latest"] = latest
                if reference is None:
                    match_status = "Not evaluated"
                else:
                    match_status = "Match" if reference == result["prediction"] else "Mismatch"
                st.session_state.setdefault("prediction_history", []).append({
                    "Filename": filename,
                    "Predicted Class": result["prediction"],
                    "Prediction Strength (%)": round(result["prediction_strength"] * 100, 2),
                    "Reference Class": reference if reference is not None else "Not specified",
                    "Match Status": match_status,
                })
    except Exception as error:
        st.error(f"Could not analyze this image: {error}")
        st.info(
            "Verify that the model checkpoints, hcac_metadata.json and "
            "model_architectures_combined.py are present in the project folders."
        )
        return

    todo = pending_stages(latest, preset_name, retry_errors=True)
    job = {
        "signature": signature, "done": False, "cancel": False,
        "stage": 1, "fraction": 0.0, "error": None, "todo": todo,
    }
    st.session_state["job"] = job
    if not todo:
        job["done"] = True
        return
    threading.Thread(
        target=_job_worker,
        args=(job, latest, image, models, score_function, preset_name, set(todo)),
        daemon=True,
    ).start()


def _completed_count(latest):
    xai = latest.get("xai") or {}
    done = 1 if latest["result"].get("mask") is not None else 0
    return done + sum(1 for key in ("gradcam", "lime", "shap") if key in xai)


def _poller():
    fragment = getattr(st, "fragment", None)
    if fragment is None:
        return None
    try:
        return fragment(run_every=1.5)
    except TypeError:          # Streamlit older than 1.37: no run_every
        return None


_POLL = _poller()


def _draw_progress(signature):
    job = st.session_state.get("job")
    latest = st.session_state.get("latest")
    if not job or latest is None or job.get("signature") != signature:
        return
    if job.get("error"):
        st.error(f"The background analysis stopped: {job['error']}")
    if job["done"] or _completed_count(latest) != st.session_state.get("_seen_count"):
        st.rerun()             # a stage finished: redraw the panels with the new result
    stage = int(job.get("stage", 1))
    overall = min((stage + job.get("fraction", 0.0)) / len(STAGE_LABELS), 1.0)
    text = f"Step {stage + 1} of {len(STAGE_LABELS)}: {STAGE_LABELS[stage]}"
    if 0.0 < job.get("fraction", 0.0) < 1.0:
        text += f" - {job['fraction'] * 100:.0f}%"
    st.progress(overall, text=text)


if _POLL is not None:
    _draw_progress = _POLL(_draw_progress)


def progress_area(signature):
    """Progress bar for a running background job; polls without blocking page switches."""
    latest = st.session_state.get("latest")
    if latest is None or not _job_running(signature):
        return
    st.session_state["_seen_count"] = _completed_count(latest)
    _draw_progress(signature)
    if _POLL is None:
        st.caption("Your Streamlit version cannot auto-refresh (needs 1.37+).")
        stretch_call(st.button, "Refresh progress")


# ---------------------------------------------------------
# Rendering
# ---------------------------------------------------------
def show_unavailable(entry):
    st.info(entry.get("message", "This explanation is not available for the current image."))
    if entry.get("detail"):
        with st.expander("Technical details"):
            st.code(entry["detail"])


def _job_running(signature=None):
    job = st.session_state.get("job")
    if not job or job.get("done"):
        return False
    return signature is None or job.get("signature") == signature


def _missing_message(label):
    latest = st.session_state.get("latest")
    if latest is not None and _job_running(latest.get("signature")):
        return f"{label} is queued or running. It will appear here automatically when it finishes."
    return f"{label} has not been generated yet. Click Analyze MRI to run it."


def render_segmentation(latest):
    probability = latest["result"].get("mask")
    if probability is None:
        if latest.get("segmentation_error"):
            st.info(f"Segmentation could not be generated for this image: {latest['segmentation_error']}")
        else:
            st.info(_missing_message("Segmentation"))
        return
    info = inspect_segmentation(probability)
    base, response, overlay, map_label, overlay_label = segmentation_views(
        model_input_array(latest["image"]), probability, info
    )
    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        image_label("Original MRI")
        show_image(base)
    with col2:
        image_label(map_label)
        show_image(response)
    with col3:
        image_label(overlay_label)
        show_image(overlay)
    render_html(
        f'<div class="nx-stats">Output range {info["min"]:.3f} to {info["max"]:.3f}'
        f' &nbsp;|&nbsp; Mean {info["mean"]:.3f}'
        f' &nbsp;|&nbsp; Pixels above 0.50: {info["positive_fraction"] * 100:.2f}%</div>'
    )
    if not info["mask_valid"]:
        st.caption(
            f"A binary mask is not shown because {info['reason']}. "
            "The maps above display the raw model output."
        )


def render_gradcam(latest):
    entry = (latest.get("xai") or {}).get("gradcam")
    if entry is None:
        st.info(_missing_message("Grad-CAM"))
        return
    if entry["status"] != "ok":
        show_unavailable(entry)
        return
    base = display_array(latest["image"])
    heat = entry["heatmap"]
    if heat.shape != base.shape[:2]:
        heat = np.asarray(
            Image.fromarray((heat * 255).astype(np.uint8)).resize(
                (base.shape[1], base.shape[0]), Image.BICUBIC
            ),
            dtype=np.float32,
        ) / 255.0
    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        image_label("Original MRI")
        show_image(base)
    with col2:
        image_label("Grad-CAM Heatmap")
        heat_rgb = colorize(heat, GRADCAM_COLORMAP)
        heat_rgb[heat <= 0.0] = 0  # keep the non-responsive canvas black, not falsely highlighted
        show_image(heat_rgb)
    with col3:
        image_label(f"Grad-CAM Overlay: {latest['result']['prediction']}")
        show_image(gradcam_overlay(base, heat))
    render_html(gradcam_legend_html())
    rows, cols = entry["map_shape"]
    note = (
        " The response is spread over a large part of the image, so treat it as a broad "
        "indication rather than a precise localisation."
        if entry["coverage"] > 0.5 else ""
    )
    st.caption(
        "Exploratory model-attention visualisation only; it is not a tumor boundary or diagnosis. "
        f"Grad-CAM for the {latest['result']['prediction']} capsule score, from layer "
        f"`{entry['layer']}` of the SwinCT CNN branch ({cols} x {rows} feature map, upsampled "
        f"to the {base.shape[1]} x {base.shape[0]} MRI). Warmer colours contributed more; responses "
        f"below {GRADCAM_THRESHOLD:.0%} of the peak are left transparent.{note}"
    )


def render_lime(latest):
    entry = (latest.get("xai") or {}).get("lime")
    if entry is None:
        st.info(_missing_message("LIME"))
        return
    if entry["status"] != "ok":
        show_unavailable(entry)
        return
    base = model_input_array(latest["image"])
    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        image_label("Original MRI")
        show_image(base)
    with col2:
        image_label("Supporting Regions")
        show_image(lime_support_overlay(base, entry["support_mask"]))
    with col3:
        image_label("Region Weights")
        show_image(overlay_signed(base, entry["weight_map"]))
    fit_text = f" Local surrogate fit (R\u00b2): {entry['fit']:.2f}." if entry.get("fit") is not None else ""
    st.caption(
        "Exploratory local surrogate; highlighted superpixels are not confirmed tumor regions. "
        f"Highlighted regions support the {latest['result']['prediction']} prediction. "
        f"In the weight map, red regions support it and blue regions oppose it. "
        f"{entry.get('samples', '')} perturbed samples over about {entry.get('segments', '')} "
        f"superpixels, scored by the full HCAC pipeline.{fit_text}"
    )


def render_shap(latest):
    entry = (latest.get("xai") or {}).get("shap")
    if entry is None:
        st.info(_missing_message("SHAP"))
        return
    if entry["status"] != "ok":
        show_unavailable(entry)
        return
    base = model_input_array(latest["image"])
    left, right = st.columns(2, gap="medium")
    with left:
        image_label("Original MRI")
        show_image(base)
    with right:
        image_label(f"SHAP Attribution: {latest['result']['prediction']}")
        show_image(overlay_signed(base, entry["values"]))
    gap = entry.get("additivity_gap")
    gap_text = f" Baseline plus attributions reproduces the model score within {gap:.4f}." if gap is not None else ""
    st.caption(
        "Exploratory feature attribution; colored blocks are not tumor segmentation. "
        f"Red regions increase the {latest['result']['prediction']} score and blue regions decrease it. "
        f"Partition SHAP ({entry.get('evals', '')} evaluations) on the full pixels-to-HCAC pipeline "
        f"with black masking. It explains image regions, not the 229 selected features.{gap_text}"
    )


def render_panel(latest, key):
    {
        "segmentation": render_segmentation,
        "gradcam": render_gradcam,
        "lime": render_lime,
        "shap": render_shap,
    }[key](latest)


def render_explanations(latest):
    if latest.get("xai") is None:
        st.info("Explanations are not available for this analysis.")
        return
    tabs = st.tabs(list(PANEL_TITLES))
    for tab, key in zip(tabs, PANEL_KEYS):
        with tab:
            render_panel(latest, key)


def render_timings(latest):
    labels = {
        "classification": "classification", "segmentation": "segmentation",
        "gradcam": "Grad-CAM", "lime": "LIME", "shap": "SHAP",
    }
    parts = [
        f"{labels[key]} {latest['timings'][key]:.1f}s"
        for key in labels if key in (latest.get("timings") or {})
    ]
    if parts:
        st.caption("Stage timings for this analysis: " + " | ".join(parts))


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
def install_page_transition():
    """Fade the current page out before a sidebar navigation rerun."""
    transition_script = r"""<script>
    (() => {
      try {
        const doc = window.parent.document;
        if (doc.__neuroScanTransitionInstalled) return;
        doc.__neuroScanTransitionInstalled = true;
        let replaying = false;
        doc.addEventListener('click', (event) => {
          if (replaying) return;
          const label = event.target && event.target.closest
            ? event.target.closest('[data-testid="stSidebar"] label') : null;
          if (!label) return;
          const radio = label.querySelector('input[type="radio"]');
          if (!radio || radio.checked) return;
          const main = doc.querySelector('[data-testid="stMainBlockContainer"]');
          if (!main) return;
          event.preventDefault();
          event.stopPropagation();
          event.stopImmediatePropagation();
          main.classList.add('nx-page-leaving');
          window.setTimeout(() => {
            replaying = true;
            label.click();
            replaying = false;
          }, 245);
        }, true);
      } catch (error) {
        /* CSS entrance animation remains as a graceful fallback. */
      }
    })();
    </script>"""
    components.html(transition_script, height=0, width=0, scrolling=False)


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
with st.sidebar:
    render_html(
        f"""<div class="nx-brand"><div class="nx-brand-logo">N</div>
        <div><div class="nx-brand-name">{APP_NAME}</div>
        <div class="nx-brand-sub">Brain MRI Analysis</div></div></div>"""
    )
    render_html('<div class="nx-nav-label">Navigation</div>')
    page = st.radio(
        "Navigation",
        PAGES,
        label_visibility="collapsed",
    )
    install_page_transition()


# =========================================================
# PAGES
# =========================================================
def page_dashboard():
    page_header(
        "AI-ASSISTED NEURO-IMAGING",
        APP_TITLE,
        "Segmentation, feature selection, classification and explainability for brain MRI.",
    )

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        metric_card("Test Accuracy", f"{TEST_ACCURACY:.2f}%")
    with c2:
        metric_card("Test Images", "1,600")
    with c3:
        metric_card("Tumor Categories", "4")
    with c4:
        metric_card("Selected Features", "229")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    section("Analysis Workflow")
    steps = [
        ("01", "MRI Input", "Upload a brain MRI scan for analysis."),
        ("02", "MDU-Net Segmentation", "Produces a segmentation response for the scan."),
        ("03", "SwinCT Feature Extraction", "Extracts deep image features from the MRI."),
        ("04", "OOA Feature Selection", "Selects the optimized 229-feature subset."),
        ("05", "HCAC Classification", "Classifies the scan into one of four categories."),
        ("06", "Explainable AI", "Grad-CAM, LIME and SHAP visual explanations."),
    ]
    for row_start in (0, 3):
        cols = st.columns(3, gap="medium")
        for col, (number, title, description) in zip(cols, steps[row_start:row_start + 3]):
            with col:
                render_html(
                    f"""<div class="nx-step">
                    <div class="nx-step-top"><span class="nx-step-num">{number}</span>
                    </div>
                    <div class="nx-step-title">{esc(title)}</div>
                    <div class="nx-step-desc">{esc(description)}</div></div>"""
                )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    section("Classification Categories")
    dot_colors = [COLOR_CYAN, COLOR_TEAL, COLOR_BLUE, "#818cf8"]
    cols = st.columns(4, gap="medium")
    for col, name, color in zip(cols, CLASS_NAMES, dot_colors):
        with col:
            render_html(
                f"""<div class="nx-cat"><span class="nx-cat-dot" style="background:{color}"></span>
                <div><div class="nx-cat-name">{esc(name)}</div>
                <div class="nx-cat-sub">400 test images</div></div></div>"""
            )


def render_prediction(latest):
    result = latest["result"]
    reference = latest["reference"]

    section("Classification Result")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        metric_card("Predicted Class", result["prediction"])
    with c2:
        metric_card(
            "Prediction Strength",
            f'{result["prediction_strength"] * 100:.2f}%',
            "Maximum raw capsule score",
        )

    if reference is not None:
        if reference == result["prediction"]:
            st.success(f"Matches the reference class: {reference}.")
        else:
            st.warning(
                f"Reference class ({reference}) differs from the model prediction "
                f"({result['prediction']})."
            )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    section("Class-wise Capsule Scores")
    show_chart(capsule_chart(result["capsule_scores"], result["prediction"]))


def render_results(latest):
    render_prediction(latest)
    st.markdown("---")
    section("Segmentation and Explainability", f"Generated for {latest['filename']}")
    st.caption("Grad-CAM, LIME and SHAP describe model behavior. They are not clinically validated tumor-localization tools.")
    render_explanations(latest)
    render_timings(latest)


def page_analysis():
    page_header(
        "IMAGE CLASSIFICATION",
        "MRI Analysis",
        "Upload a brain MRI to run the trained analysis pipeline.",
    )

    uploaded_file = st.file_uploader(
        "Upload MRI image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        key="mri_uploader",
    )
    if uploaded_file is None:
        st.info("Upload an MRI image to begin.")
        return

    file_bytes = uploaded_file.getvalue()
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as error:
        st.error(f"Could not read this image: {error}")
        return

    # The signature includes a hash of the pixels, so a different file never reuses old results.
    signature = f"{uploaded_file.name}:{hashlib.md5(file_bytes).hexdigest()}"
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        image_label("Uploaded MRI")
        show_image(image)
    with right:
        image_label("Scan Details")
        render_html(
            f"""<div class="nx-info-row"><span>File</span><span>{esc(uploaded_file.name)}</span></div>
            <div class="nx-info-row"><span>Dimensions</span><span>{image.width} \u00d7 {image.height} px</span></div>
            <div class="nx-info-row"><span>Size</span><span>{uploaded_file.size / 1024:.1f} KB</span></div>"""
        )
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        with st.expander("Reference class (optional, for testing)"):
            known_label = st.selectbox(
                "Reference class",
                ["Not specified"] + CLASS_NAMES,
                key="known_label",
                label_visibility="collapsed",
                help="Select the dataset label only if you know it. It does not affect the prediction.",
            )
        preset_name = DEFAULT_XAI_PRESET  # fixed quick preset; settings stay internal
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        run_analysis = stretch_call(st.button, "Analyze MRI", type="primary")

    if run_analysis:
        reference = None if known_label == "Not specified" else known_label
        start_analysis(image, uploaded_file.name, signature, reference, preset_name)

    latest = st.session_state.get("latest")
    if latest is not None and latest.get("signature") == signature:
        st.markdown("---")
        render_prediction(latest)
        st.markdown("---")
        section("Segmentation and Explainability", f"Generated for {latest['filename']}")
        progress_area(signature)
        render_explanations(latest)
        render_timings(latest)
        if not _job_running(signature) and pending_stages(latest, preset_name):
            st.info(
                "Some stages have not been run for the selected settings. "
                "Click Analyze MRI to run only the missing ones."
            )


def page_xai():
    page_header(
        "EXPLAINABLE ANALYSIS",
        "Segmentation & XAI",
        "Segmentation output and explanations generated for the latest analyzed MRI.",
    )

    latest = st.session_state.get("latest")
    if latest is None:
        st.info("Analyze an MRI on the MRI Analysis page to see its segmentation and explanations here.")
        return

    section(
        "Analyzed MRI",
        f"{latest['filename']}  |  Predicted class: {latest['result']['prediction']}",
    )
    progress_area(latest["signature"])
    st.caption("Grad-CAM, LIME and SHAP describe model behavior. They are not clinically validated tumor-localization tools.")
    render_explanations(latest)
    render_timings(latest)


def page_performance():
    """Model Performance page.

    Everything shown here is read from the project's saved result files
    (normally inside the `results/` folder next to app.py):

      swinct_classification_history.csv    epoch, train_loss, val_loss, train_accuracy, val_accuracy
      mdunet_segmentation_history.csv      epoch, train_loss, val_loss, train_dice, val_dice
      hcac_corrected_training_history.csv  epoch, train_loss, val_loss, train_accuracy, val_accuracy
      hcac_final_confusion_matrix.png
      hcac_final_test_metrics.csv
      dmo_optimization_curve.png

    Nothing is generated from random or placeholder data. A missing or malformed
    file produces a clear message instead of a graph.
    """
    results_dir = BASE_DIR / "results"
    search_dirs = [
        results_dir,
        BASE_DIR / "app_artifacts" / "results",
        BASE_DIR / "evaluation",
        BASE_DIR / "training_results",
        BASE_DIR,
    ]

    def spacer(pixels=14):
        st.markdown(f"<div style='height:{pixels}px'></div>", unsafe_allow_html=True)

    def locate(filename):
        for folder in search_dirs:
            candidate = folder / filename
            if candidate.is_file():
                return candidate
        return None

    def missing_file_message(filename, purpose):
        st.info(f"{purpose} is not available.")

    # -----------------------------------------------------
    # Cached data loading and figure rendering
    # (cache keys include the file's modified time, so re-running training
    # and overwriting a file refreshes the page automatically)
    # -----------------------------------------------------
    @st.cache_data(show_spinner=False)
    def read_history(csv_path, modified):
        frame = pd.read_csv(csv_path)
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        return frame

    @st.cache_data(show_spinner=False)
    def curve_png(csv_path, modified, train_col, val_col, ylabel, higher_is_better):
        from matplotlib.figure import Figure
        from matplotlib.ticker import MaxNLocator

        frame = pd.read_csv(csv_path)
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        train = pd.to_numeric(frame[train_col], errors="coerce").to_numpy(dtype=float)
        val = pd.to_numeric(frame[val_col], errors="coerce").to_numpy(dtype=float)
        if "epoch" in frame.columns:
            epochs = pd.to_numeric(frame["epoch"], errors="coerce").to_numpy(dtype=float)
        else:
            epochs = np.arange(1, len(frame) + 1, dtype=float)
        if not np.isfinite(epochs).all():
            epochs = np.arange(1, len(frame) + 1, dtype=float)

        figure = Figure(figsize=(6.6, 4.0), dpi=200, facecolor="#0b1e33")
        ax = figure.subplots()
        ax.set_facecolor("#0e2238")
        ax.plot(epochs, train, color="#22d3ee", linewidth=2.2, label="Train")
        ax.plot(epochs, val, color="#fbbf24", linewidth=2.2, label="Validation")

        if np.isfinite(val).any():
            best = int(np.nanargmax(val) if higher_is_better else np.nanargmin(val))
            digits = 2 if np.nanmax(np.abs(val)) > 1.5 else 4
            ax.scatter(
                [epochs[best]], [val[best]], s=70, color="#fbbf24",
                edgecolors="#ffffff", linewidths=1.2, zorder=5,
            )
            ax.annotate(
                f"Best {val[best]:.{digits}f} @ epoch {int(epochs[best])}",
                xy=(epochs[best], val[best]),
                xytext=(0, 14),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color="#fde68a",
                bbox=dict(boxstyle="round,pad=0.3", fc="#0b1e33", ec="#fbbf24", lw=0.8),
            )

        ax.margins(x=0.03, y=0.16)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for spine in ax.spines.values():
            spine.set_color("#2a4a6b")
        ax.tick_params(colors="#a9c4df", labelsize=9)
        ax.grid(True, color="#2a4a6b", alpha=0.35, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("Epoch", color="#a9c4df", fontsize=10)
        ax.set_ylabel(ylabel, color="#a9c4df", fontsize=10)
        legend = ax.legend(loc="best", frameon=True, fontsize=9)
        legend.get_frame().set_facecolor("#0b1e33")
        legend.get_frame().set_edgecolor("#2a4a6b")
        for text in legend.get_texts():
            text.set_color("#e8f2fc")

        figure.tight_layout()
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", facecolor=figure.get_facecolor())
        return buffer.getvalue()

    @st.cache_data(show_spinner=False)
    def report_png(classes, precision, recall, f1_score):
        from matplotlib.figure import Figure

        positions = np.arange(len(classes))
        width = 0.26
        figure = Figure(figsize=(10.5, 4.0), dpi=200, facecolor="#0b1e33")
        ax = figure.subplots()
        ax.set_facecolor("#0e2238")
        series = [
            ("Precision", precision, "#14b8a6"),
            ("Recall", recall, "#22d3ee"),
            ("F1-score", f1_score, "#60a5fa"),
        ]
        for offset, (label, values, color) in zip((-width, 0.0, width), series):
            bars = ax.bar(positions + offset, values, width, label=label, color=color)
            for rect, value in zip(bars, values):
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    value + 0.012,
                    f"{value:.2f}",
                    ha="center", va="bottom", fontsize=8, color="#e8f2fc",
                )
        ax.set_xticks(positions)
        ax.set_xticklabels(classes)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score", color="#a9c4df", fontsize=10)
        ax.tick_params(colors="#a9c4df", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#2a4a6b")
        ax.grid(True, axis="y", color="#2a4a6b", alpha=0.35, linewidth=0.8)
        ax.set_axisbelow(True)
        legend = ax.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=10
        )
        for text in legend.get_texts():
            text.set_color("#e8f2fc")
        figure.tight_layout()
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", facecolor=figure.get_facecolor())
        return buffer.getvalue()

    # -----------------------------------------------------
    # Classification report loader (hcac_final_test_metrics.csv)
    # -----------------------------------------------------
    def load_report():
        """Return (report_dataframe, status). status: csv | missing | format."""
        recorded = REPORT_DF.copy()
        path = locate("hcac_final_test_metrics.csv")
        if path is None:
            return recorded, "missing"
        try:
            frame = pd.read_csv(path)
            frame.columns = [str(c).strip().lower() for c in frame.columns]
            lookup = {re.sub(r"[^a-z0-9]", "", col): col for col in frame.columns}

            def pick(*names):
                for name in names:
                    if name in lookup:
                        return lookup[name]
                return None

            precision_col = pick("precision")
            recall_col = pick("recall")
            f1_col = pick("f1score", "f1", "fscore")
            support_col = pick("support")
            name_col = pick("class", "classname", "label", "category", "tumortype", "unnamed0")
            if name_col is None and len(frame.columns) > 0:
                first_column = frame.columns[0]
                if not pd.api.types.is_numeric_dtype(frame[first_column]):
                    name_col = first_column
            if name_col is None or not (precision_col and recall_col and f1_col):
                return recorded, "format"

            canonical = {re.sub(r"[^a-z0-9]", "", n.lower()): n for n in CLASS_NAMES}
            index_names = {str(i): n for i, n in enumerate(CLASS_NAMES)}
            rows = {}
            for _, row in frame.iterrows():
                key = re.sub(r"[^a-z0-9]", "", str(row[name_col]).lower())
                class_name = canonical.get(key) or index_names.get(key)
                if class_name is not None:
                    rows[class_name] = row
            if len(rows) != len(CLASS_NAMES):
                return recorded, "format"

            def numbers(column):
                return [float(pd.to_numeric(rows[name][column], errors="coerce")) for name in CLASS_NAMES]

            precision = numbers(precision_col)
            recall = numbers(recall_col)
            f1_values = numbers(f1_col)
            if support_col:
                support = numbers(support_col)
            else:
                support = [float(v) for v in recorded["Support"]]
            everything = precision + recall + f1_values + support
            if not np.isfinite(everything).all():
                return recorded, "format"
            if max(precision + recall + f1_values) > 1.5:
                precision = [v / 100.0 for v in precision]
                recall = [v / 100.0 for v in recall]
                f1_values = [v / 100.0 for v in f1_values]
            report = pd.DataFrame({
                "Class": CLASS_NAMES,
                "Precision": precision,
                "Recall": recall,
                "F1-score": f1_values,
                "Support": [int(round(v)) for v in support],
            })
            return report, "csv"
        except Exception:
            return recorded, "format"

    # -----------------------------------------------------
    # One training-history tab
    # -----------------------------------------------------
    def render_history(spec):
        filename = spec["file"]
        path = locate(filename)
        if path is None:
            missing_file_message(filename, f"The {spec['name']} training history")
            return
        try:
            frame = read_history(str(path), path.stat().st_mtime)
        except Exception as error:
            st.error(f"Could not read `{filename}`: {error}")
            return

        required = ["train_loss", "val_loss", spec["train"], spec["val"]]
        missing_columns = [c for c in required if c not in frame.columns]
        if missing_columns:
            st.error(
                f"`{filename}` is missing column(s): {', '.join(missing_columns)}. "
                f"Columns found: {', '.join(frame.columns)}."
            )
            return
        if frame.empty:
            st.warning(f"`{filename}` contains no rows.")
            return

        try:
            train_metric = pd.to_numeric(frame[spec["train"]], errors="coerce")
            val_metric = pd.to_numeric(frame[spec["val"]], errors="coerce")
            val_loss = pd.to_numeric(frame["val_loss"], errors="coerce")
            if "epoch" in frame.columns:
                epoch_values = pd.to_numeric(frame["epoch"], errors="coerce")
            else:
                epoch_values = pd.Series(np.arange(1, len(frame) + 1), index=frame.index)
            epoch_values = epoch_values.fillna(pd.Series(np.arange(1, len(frame) + 1), index=frame.index))

            is_accuracy = spec["kind"] == "accuracy"
            overall_max = max(float(train_metric.max()), float(val_metric.max()))
            is_fraction = is_accuracy and overall_max <= 1.0

            def fmt(value):
                if is_accuracy:
                    return f"{value * 100:.2f}%" if is_fraction else f"{value:.2f}%"
                return f"{value:.4f}"

            best_idx = val_metric.idxmax()
            loss_idx = val_loss.idxmin()
            final_idx = frame.index[-1]

            c1, c2, c3, c4 = st.columns(4, gap="medium")
            with c1:
                metric_card("Epochs Trained", f"{len(frame)}")
            with c2:
                metric_card(f"Best Val {spec['metric']}",
                    fmt(float(val_metric[best_idx])),
                    f"Epoch {int(epoch_values[best_idx])}",
                )
            with c3:
                metric_card(f"Final Val {spec['metric']}",
                    fmt(float(val_metric[final_idx])),
                    f"Train {fmt(float(train_metric[final_idx]))}",
                )
            with c4:
                metric_card("Lowest Val Loss",
                    f"{float(val_loss[loss_idx]):.4f}",
                    f"Epoch {int(epoch_values[loss_idx])}",
                )

            spacer(16)
            if is_accuracy:
                metric_axis = "Accuracy" if is_fraction else "Accuracy (%)"
            else:
                metric_axis = spec["metric"]

            modified = path.stat().st_mtime
            loss_image = curve_png(str(path), modified, "train_loss", "val_loss", "Loss", False)
            metric_image = curve_png(
                str(path), modified, spec["train"], spec["val"], metric_axis, True
            )
            left, right = st.columns(2, gap="large")
            with left:
                image_label(f"{spec['name']} – Training vs Validation Loss")
                show_image(loss_image)
            with right:
                image_label(f"{spec['name']} – Training vs Validation {spec['metric']}")
                show_image(metric_image)
        except Exception as error:
            st.error(f"Could not build the {spec['name']} graphs from `{filename}`: {error}")

    # -----------------------------------------------------
    # PAGE
    # -----------------------------------------------------
    page_header(
        "EVALUATION",
        "Model Performance",
        "Test-set results and training behavior of every model in the pipeline.",
    )

    m1, m2, m3, m4 = st.columns(4, gap="medium")
    with m1:
        metric_card("Test Accuracy", f"{TEST_ACCURACY:.2f}%")
    with m2:
        metric_card("Test Images", "1,600")
    with m3:
        metric_card("Classes", "4")
    with m4:
        metric_card("Images per Class", "400")

    # ---- Classification report ----
    spacer(16)
    section("Classification Report", "Per-class precision, recall, F1-score and support on the test set.")
    report_df, _ = load_report()
    report_display = report_df.copy()
    for column in ("Precision", "Recall", "F1-score"):
        report_display[column] = report_display[column].map(lambda v: f"{v:.2f}")
    report_display["Support"] = report_display["Support"].map(lambda v: f"{int(v):,}")
    render_table(report_display, numeric_cols=("Precision", "Recall", "F1-score", "Support"))

    spacer(18)
    image_label("Precision, Recall and F1-score by Class")
    show_image(
        report_png(
            tuple(report_df["Class"]),
            tuple(float(v) for v in report_df["Precision"]),
            tuple(float(v) for v in report_df["Recall"]),
            tuple(float(v) for v in report_df["F1-score"]),
        )
    )

    # ---- Confusion matrix ----
    st.markdown("---")
    section("Confusion Matrix", "HCAC classifier evaluated on the 1,600-image test set.")
    confusion_path = locate("hcac_final_confusion_matrix.png")
    if confusion_path is None:
        missing_file_message("hcac_final_confusion_matrix.png", "The HCAC confusion matrix")
    else:
        _, middle, _ = st.columns([1, 2, 1])
        with middle:
            show_image(str(confusion_path))

    # ---- Training curves ----
    st.markdown("---")
    section("Training Curves", "Training and validation history for each model, drawn from the saved CSV logs.")
    history_specs = [
        {
            "tab": "SwinCT Classifier",
            "name": "SwinCT",
            "file": "swinct_classification_history.csv",
            "train": "train_accuracy",
            "val": "val_accuracy",
            "metric": "Accuracy",
            "kind": "accuracy",
        },
        {
            "tab": "MDU-Net Segmentation",
            "name": "MDU-Net",
            "file": "mdunet_segmentation_history.csv",
            "train": "train_dice",
            "val": "val_dice",
            "metric": "Dice Score",
            "kind": "dice",
        },
        {
            "tab": "HCAC Classifier",
            "name": "HCAC",
            "file": "hcac_corrected_training_history.csv",
            "train": "train_accuracy",
            "val": "val_accuracy",
            "metric": "Accuracy",
            "kind": "accuracy",
        },
    ]
    tabs = st.tabs([spec["tab"] for spec in history_specs])
    for tab, spec in zip(tabs, history_specs):
        with tab:
            spacer(6)
            render_history(spec)

    # ---- Optimization curve ----
    st.markdown("---")
    section("Feature Selection Optimization", "Convergence curve saved from the optimization run.")
    optimization_path = locate("dmo_optimization_curve.png")
    if optimization_path is None:
        missing_file_message("dmo_optimization_curve.png", "The optimization curve")
    else:
        _, middle, _ = st.columns([1, 2, 1])
        with middle:
            show_image(str(optimization_path))


def page_history():
    page_header(
        "RECENT ACTIVITY",
        "Prediction History",
        "Predictions made during this session.",
    )

    history = st.session_state.get("prediction_history", [])
    if not history:
        st.info("No predictions have been recorded in this session yet.")
        return

    history_df = pd.DataFrame(history)
    display_df = history_df.iloc[::-1].copy()
    display_df["Prediction Strength"] = display_df["Prediction Strength (%)"].map(
        lambda v: f"{v:.2f}%"
    )
    display_df = display_df[
        ["Filename", "Predicted Class", "Prediction Strength", "Reference Class", "Match Status"]
    ]
    render_table(display_df, badge_col="Match Status", numeric_cols=("Prediction Strength",))

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    b1, b2, _ = st.columns([1, 1, 2.2], gap="medium")
    with b1:
        stretch_call(
            st.download_button,
            "Download CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="neuroscan_prediction_history.csv",
            mime="text/csv",
        )
    with b2:
        if stretch_call(st.button, "Clear History"):
            st.session_state["prediction_history"] = []
            st.rerun()


PAGE_FUNCTIONS = {
    "Dashboard": page_dashboard,
    "MRI Analysis": page_analysis,
    "Segmentation & XAI": page_xai,
    "Model Performance": page_performance,
    "Prediction History": page_history,
}


def page_container(name):
    """Keyed container so every incoming page gets a consistent entrance animation."""
    key = "nx-page-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    try:
        return st.container(key=key)
    except TypeError:
        return st.container()


with page_container(page):
    PAGE_FUNCTIONS[page]()
