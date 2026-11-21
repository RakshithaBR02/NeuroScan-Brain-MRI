import matplotlib.pyplot as plt
import html as html_lib
import io
import json
import os
import re
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
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
    page_icon="\U0001F9E0",
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
PAGE_ICONS = {
    "Dashboard": "\U0001F4CA",
    "MRI Analysis": "\U0001F9E0",
    "Segmentation & XAI": "\U0001F52C",
    "Model Performance": "\U0001F4C8",
    "Prediction History": "\U0001F5C2\uFE0F",
}

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
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --nx-bg: #06101f;
    --nx-panel: #0d1d33;
    --nx-border: rgba(64, 170, 220, .22);
    --nx-teal: #14b8a6;
    --nx-cyan: #22d3ee;
    --nx-text: #e8f2fc;
    --nx-muted: #93aecb;
}

html, body, [class*="css"], .stApp, button, input, select, textarea {
    font-family: 'Manrope', 'Inter', sans-serif !important;
}
.stApp {
    background:
        radial-gradient(circle at 12% 0%, rgba(20, 184, 166, .13), transparent 30%),
        radial-gradient(circle at 92% 8%, rgba(34, 211, 238, .09), transparent 30%),
        #06101f;
    color: var(--nx-text);
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container {
    padding-top: 2rem;
    padding-bottom: 2.5rem;
    max-width: 1240px;
}

/* One shared page transition (180 ms) for every page */
@keyframes nxPageFade {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
[class*="st-key-nx-page-"] {
    animation: nxPageFade 180ms ease-out both;
}

h1, h2, h3, h4 { color: #f4f9ff !important; letter-spacing: -.01em; }
p, label, li { color: #cfe0f3; }
[data-testid="stCaptionContainer"], small { color: var(--nx-muted) !important; }
hr { border-color: rgba(96, 163, 220, .16); margin: 1.6rem 0; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1a2f 0%, #06101f 100%);
    border-right: 1px solid rgba(64, 170, 220, .18);
}
.nx-brand { display: flex; align-items: center; gap: 12px; padding: 6px 4px 14px 4px; }
.nx-brand-logo {
    width: 44px; height: 44px; border-radius: 13px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem;
    background: linear-gradient(135deg, rgba(20,184,166,.35), rgba(34,211,238,.18));
    border: 1px solid rgba(34, 211, 238, .45);
}
.nx-brand-name { font-weight: 800; font-size: 1.2rem; color: #fff; line-height: 1.1; }
.nx-brand-sub { font-size: .78rem; color: var(--nx-muted); margin-top: 2px; }
.nx-nav-label {
    font-size: .7rem; font-weight: 800; letter-spacing: .16em;
    color: #5f87ab; margin: 8px 4px 8px 4px; text-transform: uppercase;
}
[data-testid="stSidebar"] div[role="radiogroup"] { gap: 6px; }
[data-testid="stSidebar"] div[role="radiogroup"] > label {
    width: 100%;
    padding: 10px 14px;
    border-radius: 12px;
    border: 1px solid transparent;
    transition: background .15s ease, border-color .15s ease;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: rgba(34, 211, 238, .08);
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(90deg, rgba(20,184,166,.24), rgba(34,211,238,.10));
    border-color: rgba(34, 211, 238, .45);
}
[data-testid="stSidebar"] div[role="radiogroup"] label p {
    color: #e8f2fc; font-weight: 600; font-size: .95rem;
}
.nx-side-note {
    font-size: .74rem; color: #6f8fb0; line-height: 1.5;
    padding: 12px 6px 0 6px; border-top: 1px solid rgba(96,163,220,.14); margin-top: 18px;
}

/* Hero */
.nx-hero {
    background: linear-gradient(120deg, rgba(14,56,86,.97), rgba(10,28,52,.96) 60%, rgba(12,64,78,.92));
    border: 1px solid rgba(34, 211, 238, .28);
    border-radius: 20px;
    padding: 26px 30px;
    margin-bottom: 22px;
    box-shadow: 0 14px 36px rgba(0,0,0,.25);
}
.nx-kicker {
    color: var(--nx-cyan); font-size: .74rem; font-weight: 800;
    letter-spacing: .18em; text-transform: uppercase;
}
.nx-title {
    color: #ffffff; font-size: 2rem; font-weight: 800;
    letter-spacing: -.02em; margin: 8px 0 4px 0; line-height: 1.15;
}
.nx-sub { color: #a9c4df; font-size: .98rem; }

/* Section heading */
.nx-section { display: flex; align-items: center; gap: 12px; margin: 6px 0 14px 0; }
.nx-bar {
    width: 4px; height: 30px; border-radius: 4px;
    background: linear-gradient(180deg, var(--nx-cyan), var(--nx-teal));
}
.nx-section-title { color: #f4f9ff; font-size: 1.18rem; font-weight: 800; line-height: 1.2; }
.nx-section-sub { color: var(--nx-muted); font-size: .85rem; margin-top: 2px; }

/* Metric cards */
.nx-metric {
    display: flex; align-items: center; gap: 14px;
    min-height: 100px; padding: 18px 20px; box-sizing: border-box;
    background: linear-gradient(145deg, rgba(18,44,72,.95), rgba(10,26,46,.96));
    border: 1px solid var(--nx-border);
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0,0,0,.22);
}
.nx-metric-icon {
    flex: 0 0 auto; width: 46px; height: 46px; border-radius: 13px;
    display: flex; align-items: center; justify-content: center; font-size: 1.3rem;
    background: linear-gradient(135deg, rgba(20,184,166,.30), rgba(34,211,238,.14));
    border: 1px solid rgba(34, 211, 238, .35);
}
.nx-metric-label {
    color: var(--nx-muted); font-size: .76rem; font-weight: 700;
    letter-spacing: .08em; text-transform: uppercase;
}
.nx-metric-value { color: var(--nx-cyan); font-size: 1.7rem; font-weight: 800; line-height: 1.2; }
.nx-metric-sub { color: #7f9ebf; font-size: .76rem; margin-top: 2px; }

/* Workflow and category cards */
.nx-step {
    height: 172px; padding: 20px; box-sizing: border-box;
    background: rgba(13, 34, 57, .80);
    border: 1px solid rgba(64, 170, 220, .20);
    border-radius: 16px;
    box-shadow: 0 6px 20px rgba(0,0,0,.16);
    display: flex; flex-direction: column; justify-content: flex-start;
    margin-bottom: 6px;
}
.nx-step-top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
.nx-step-num { color: var(--nx-cyan); font-weight: 800; font-size: 1.05rem; letter-spacing: .06em; }
.nx-step-icon {
    width: 38px; height: 38px; border-radius: 11px; font-size: 1.1rem;
    display: flex; align-items: center; justify-content: center;
    background: rgba(20,184,166,.16); border: 1px solid rgba(34,211,238,.28);
}
.nx-step-title { color: #f4f9ff; font-weight: 800; font-size: 1.04rem; margin-bottom: 5px; }
.nx-step-desc { color: #a9c4df; font-size: .88rem; line-height: 1.45; }
.nx-cat {
    min-height: 96px; padding: 18px 20px; box-sizing: border-box;
    background: rgba(13, 34, 57, .80);
    border: 1px solid rgba(64, 170, 220, .20);
    border-radius: 16px;
    display: flex; align-items: center; gap: 14px;
}
.nx-cat-dot { width: 12px; height: 12px; border-radius: 50%; flex: 0 0 auto; }
.nx-cat-name { color: #f4f9ff; font-weight: 800; font-size: 1.06rem; }
.nx-cat-sub { color: var(--nx-muted); font-size: .8rem; }

/* Image labels */
.nx-img-label {
    font-size: .74rem; font-weight: 800; letter-spacing: .1em;
    text-transform: uppercase; color: #7fd6ee; margin-bottom: 8px;
}
[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid rgba(64, 170, 220, .25);
}
.nx-note {
    background: rgba(20, 184, 166, .08);
    border: 1px solid rgba(20, 184, 166, .28);
    border-radius: 12px; padding: 11px 16px;
    color: #b6dbe3; font-size: .86rem; margin-top: 12px;
}
.nx-info-row { display: flex; justify-content: space-between; gap: 12px;
    padding: 9px 0; border-bottom: 1px solid rgba(96,163,220,.14); font-size: .9rem; }
.nx-info-row span:first-child { color: var(--nx-muted); }
.nx-info-row span:last-child { color: #eaf3ff; font-weight: 600; text-align: right; word-break: break-all; }

/* Tables */
.nx-table-wrap {
    background: rgba(13, 34, 57, .80);
    border: 1px solid rgba(64, 170, 220, .22);
    border-radius: 14px; overflow: auto; max-height: 460px;
}
.nx-table { width: 100%; border-collapse: collapse; font-size: .92rem; }
.nx-table th {
    text-align: left; padding: 12px 16px; color: #7fd6ee;
    font-size: .74rem; letter-spacing: .1em; text-transform: uppercase; font-weight: 800;
    background: rgba(10, 30, 52, .95); position: sticky; top: 0;
    border-bottom: 1px solid rgba(64,170,220,.25);
}
.nx-table td { padding: 11px 16px; color: #dbe9f8; border-bottom: 1px solid rgba(96,163,220,.10); }
.nx-table tr:last-child td { border-bottom: 0; }
.nx-table tbody tr:hover td { background: rgba(34, 211, 238, .05); }
.nx-table td.num, .nx-table th.num { text-align: right; }
.nx-badge {
    display: inline-block; padding: 3px 11px; border-radius: 999px;
    font-size: .78rem; font-weight: 700;
}
.nx-badge-ok { background: rgba(20,184,166,.18); color: #5eead4; border: 1px solid rgba(20,184,166,.4); }
.nx-badge-bad { background: rgba(245,158,11,.16); color: #fcd34d; border: 1px solid rgba(245,158,11,.4); }
.nx-badge-na { background: rgba(147,174,203,.12); color: #93aecb; border: 1px solid rgba(147,174,203,.25); }

/* Widgets */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(90deg, #0d9488, #0891b2);
    color: #ffffff; border: 1px solid rgba(34,211,238,.35);
    border-radius: 11px; padding: .62rem 1.2rem; font-weight: 700;
    box-shadow: 0 6px 18px rgba(8, 145, 178, .22);
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: #67e8f9; color: #ffffff;
    background: linear-gradient(90deg, #14b8a6, #06b6d4);
}
.stButton > button p, .stDownloadButton > button p { color: #ffffff; }
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] {
    background: rgba(13, 34, 57, .75);
    border: 1px dashed rgba(34, 211, 238, .5);
    border-radius: 14px;
}
[data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span { color: #a9c4df; }
[data-testid="stFileUploader"] button {
    background: rgba(20,184,166,.18); color: #e8f2fc;
    border: 1px solid rgba(34,211,238,.4);
}
div[data-baseweb="select"] > div {
    background: #0f2540; border-color: rgba(64,170,220,.35); color: #eaf3ff;
}
div[data-baseweb="select"] span, div[data-baseweb="select"] div { color: #eaf3ff; }
div[data-baseweb="popover"] ul { background: #0f2540; }
div[data-baseweb="popover"] li { color: #eaf3ff; }
[data-testid="stExpander"] {
    background: rgba(13, 34, 57, .55);
    border: 1px solid rgba(64,170,220,.18);
    border-radius: 12px;
}
[data-testid="stExpander"] summary p { color: #a9c4df; font-size: .88rem; font-weight: 600; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid rgba(96,163,220,.18); }
.stTabs [data-baseweb="tab"] { color: #93aecb; font-weight: 700; padding: 10px 18px; }
.stTabs [aria-selected="true"] { color: #2dd4bf; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #22d3ee; }

.nx-footer { text-align: center; color: #6f8fb0; font-size: .78rem; padding: 6px 0 0 0; }

@media (max-width: 768px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .nx-title { font-size: 1.5rem; }
    .nx-hero { padding: 20px; }
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


def metric_card(icon, label, value, sub=None):
    sub_html = f'<div class="nx-metric-sub">{esc(sub)}</div>' if sub else ""
    render_html(
        f"""<div class="nx-metric"><div class="nx-metric-icon">{icon}</div>
        <div><div class="nx-metric-label">{esc(label)}</div>
        <div class="nx-metric-value">{esc(value)}</div>{sub_html}</div></div>"""
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
            gridColor="rgba(120,170,220,0.12)",
            domainColor="rgba(120,170,220,0.25)",
            tickColor="rgba(120,170,220,0.25)",
            labelFontSize=12,
            titleFontSize=12,
            labelFont="Manrope",
            titleFont="Manrope",
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
            title="Raw capsule score",
            scale=alt.Scale(domain=[0, 1.1], nice=False),
            axis=alt.Axis(values=[0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        ),
        color=alt.condition("datum.Top", alt.value(COLOR_CYAN), alt.value(COLOR_MUTED_BAR)),
        tooltip=[alt.Tooltip("Class:N"), alt.Tooltip("Score:Q", format=".4f")],
    )
    labels = base.mark_text(
        align="left", dx=6, color="#eaf3ff", fontWeight=600, fontSize=12
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
        dy=-8, color="#eaf3ff", fontWeight=700, fontSize=12
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


# =========================================================
# MRI INFERENCE (unchanged pipeline)
# =========================================================
def analyze_mri(image, metadata, models):
    mdunet, swinct, hcac = models
    image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        # MDU-Net segmentation output
        segmentation_output = mdunet(image_tensor)
        if isinstance(segmentation_output, (tuple, list)):
            segmentation_output = segmentation_output[0]
        segmentation_mask = torch.sigmoid(segmentation_output)[0, 0].detach().cpu().numpy()

        # SwinCT feature extraction
        features = swinct.feature_extractor(image_tensor)
        if isinstance(features, (tuple, list)):
            features = features[0]
        features = features.reshape(features.shape[0], -1)

        scaler_mean = torch.as_tensor(
            metadata["scaler_mean"], dtype=features.dtype, device=DEVICE
        ).reshape(1, -1)
        scaler_scale = torch.as_tensor(
            metadata["scaler_scale"], dtype=features.dtype, device=DEVICE
        ).reshape(1, -1)

        standardized_features = (features - scaler_mean) / torch.clamp(
            scaler_scale, min=1e-8
        )
        selected_indices = torch.as_tensor(
            metadata["selected_feature_indices"], dtype=torch.long, device=DEVICE
        )
        selected_features = standardized_features[:, selected_indices]

        # HCAC returns capsule lengths/scores, not calibrated probabilities.
        predictions, reconstruction, latent = hcac(selected_features)
        capsule_scores = predictions[0].detach().cpu().numpy()

    if len(capsule_scores) != len(CLASS_NAMES) or not np.isfinite(capsule_scores).all():
        raise ValueError("HCAC returned invalid class scores. Check the classifier checkpoint.")

    predicted_index = int(np.argmax(capsule_scores))
    return {
        "prediction": CLASS_NAMES[predicted_index],
        "prediction_index": predicted_index,
        "prediction_strength": float(capsule_scores[predicted_index]),
        "capsule_scores": capsule_scores,
        "mask": segmentation_mask,
    }


# =========================================================
# MDU-NET RESPONSE VISUALISATION (uses the raw model output only)
# =========================================================
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


def prepare_response_maps(image, mask):
    """Return (original, contrast-enhanced response map, overlay) as uint8 RGB arrays."""
    mask = np.squeeze(np.asarray(mask, dtype=np.float32))
    height, width = mask.shape
    base = np.asarray(
        image.convert("RGB").resize((width, height), Image.BILINEAR), dtype=np.float32
    )

    low, high = np.percentile(mask, [1, 99.5])
    if not high > low:
        low, high = float(mask.min()), float(mask.max())
    if high > low:
        normalized = np.clip((mask - low) / (high - low), 0.0, 1.0)
    else:
        normalized = np.zeros_like(mask)

    response = apply_colormap(normalized)
    alpha = (0.10 + 0.60 * normalized)[..., None]
    overlay = base * (1.0 - alpha) + response * alpha
    return (
        base.astype(np.uint8),
        response.astype(np.uint8),
        np.clip(overlay, 0, 255).astype(np.uint8),
    )


def render_response_maps(image, mask):
    original, response, overlay = prepare_response_maps(image, mask)
    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        image_label("Original MRI")
        show_image(original)
    with col2:
        image_label("MDU-Net Response Map")
        show_image(response)
    with col3:
        image_label("MRI Overlay")
        show_image(overlay)
    render_html(
        '<div class="nx-note">Contrast-enhanced visualization of the raw MDU-Net output. '
        "It shows where the model responds and is <b>not</b> a validated tumor mask.</div>"
    )


# =========================================================
# RESULT IMAGE DISCOVERY (recursive, flexible, cached)
# =========================================================
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SKIP_DIRS = {
    ".git", "__pycache__", "venv", ".venv", "env", "node_modules",
    "site-packages", ".ipynb_checkpoints", ".streamlit",
}
RESULT_DIR_HINTS = (
    "result", "train", "eval", "plot", "figure", "output", "artifact",
    "xai", "visual", "metric", "explain", "graph",
)
FOLDER_RANK = {"training_results": 0, "evaluation": 1, "results": 2, "app_artifacts": 3}
KEYWORD_STOPWORDS = {
    "shape", "shapes", "shaped", "gloss", "glossary", "bloss", "timeline", "sublime",
}
SEARCHED_FOLDERS_TEXT = "training_results, evaluation, results, app_artifacts (including subfolders)"


@st.cache_data(show_spinner=False)
def index_result_images(base_dir):
    """Scan result-like project folders once and cache every image found."""
    base = Path(base_dir)
    entries = []

    def add(path):
        try:
            relative = path.relative_to(base)
        except ValueError:
            return
        parts = relative.parts
        rank = FOLDER_RANK.get(parts[0].lower(), 4) if len(parts) > 1 else 5
        stem = re.sub(r"([a-z])([A-Z])", r"\1 \2", path.stem).lower()
        tokens = tuple(t for t in re.split(r"[^a-z0-9]+", stem) if t)
        entries.append((str(path), relative.as_posix(), rank, tokens, "".join(tokens)))

    try:
        children = sorted(base.iterdir())
    except OSError:
        return []

    for child in children:
        try:
            if child.is_file():
                if child.suffix.lower() in IMAGE_EXTENSIONS:
                    add(child)
            elif (
                child.is_dir()
                and child.name not in SKIP_DIRS
                and not child.name.startswith(".")
                and any(hint in child.name.lower() for hint in RESULT_DIR_HINTS)
            ):
                for root, dirs, files in os.walk(child):
                    dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
                    depth = len(Path(root).relative_to(base).parts)
                    if depth >= 8:
                        dirs[:] = []
                    for name in sorted(files):
                        candidate = Path(root) / name
                        if candidate.suffix.lower() in IMAGE_EXTENSIONS:
                            add(candidate)
                    if len(entries) >= 3000:
                        break
        except OSError:
            continue
    return entries


def get_image_index():
    return index_result_images(str(BASE_DIR))


def _keyword_matches(keyword, tokens, joined):
    if keyword in tokens:
        return True
    if len(keyword) >= 6 and keyword in joined:
        return True
    for token in tokens:
        if token in KEYWORD_STOPWORDS:
            continue
        if len(token) > len(keyword) and (token.startswith(keyword) or token.endswith(keyword)):
            return True
    return False


def find_images(required, excluded=(), preferred=()):
    """Return matching image paths, best first.

    required: list of groups; each group is a tuple of alternative keywords
    (every group must be satisfied by at least one alternative).
    """
    matches = []
    for path, relative, rank, tokens, joined in get_image_index():
        if not all(
            any(_keyword_matches(alt, tokens, joined) for alt in group) for group in required
        ):
            continue
        if any(_keyword_matches(word, tokens, joined) for word in excluded):
            continue
        preference = sum(1 for word in preferred if _keyword_matches(word, tokens, joined))
        matches.append((-preference, rank, len(relative), path))
    matches.sort()
    return [Path(item[3]) for item in matches]


def find_first(*queries):
    """Try several (required, excluded, preferred) queries in order."""
    for required, excluded, preferred in queries:
        found = find_images(required, excluded, preferred)
        if found:
            return found[0]
    return None


def find_confusion_matrix():
    return find_first(
        ([("confusion",)], ("mdunet", "swinct"), ("hcac", "test", "final")),
        ([("confusion",)], (), ("hcac", "test", "final")),
    )


def find_swinct_accuracy():
    return find_first(
        ([("swinct", "swin"), ("accuracy", "acc")], ("loss", "confusion", "mdunet", "segmentation"), ()),
        ([("accuracy", "acc")], ("loss", "confusion", "mdunet", "segmentation", "hcac"), ("swinct", "swin")),
    )


def find_swinct_loss():
    return find_first(
        ([("swinct", "swin"), ("loss",)], ("mdunet", "unet", "segmentation", "accuracy"), ()),
        ([("loss",)], ("mdunet", "unet", "segmentation", "seg", "hcac"), ("swinct", "swin")),
    )


def find_mdunet_loss():
    return find_first(
        ([("mdunet", "unet", "segmentation", "seg"), ("loss",)], ("swinct", "swin"), ("mdunet", "segmentation")),
    )


XAI_QUERIES = {
    "MDU-Net": (
        [("mdunet", "unet", "segmentation")],
        ("loss", "curve", "accuracy", "confusion", "gradcam", "lime", "shap"),
        ("example", "segmentation", "output", "result"),
        "mdunet_segmentation_example.png",
    ),
    "Grad-CAM": (
        [("gradcam",)],
        ("loss", "accuracy", "confusion"),
        ("hcac", "example"),
        "gradcam_hcac_example.png",
    ),
    "LIME": (
        [("lime",)],
        ("loss", "accuracy", "confusion"),
        ("hcac", "example"),
        "lime_hcac_example.png",
    ),
    "SHAP": (
        [("shap",)],
        ("loss", "accuracy", "confusion"),
        ("hcac", "example"),
        "shap_hcac_example.png",
    ),
}


def missing_file_note(description, example_name):
    st.info(
        f"{description} was not found. Expected a file such as `{example_name}` "
        f"inside: {SEARCHED_FOLDERS_TEXT}."
    )


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
with st.sidebar:
    render_html(
        f"""<div class="nx-brand"><div class="nx-brand-logo">\U0001F9E0</div>
        <div><div class="nx-brand-name">{APP_NAME}</div>
        <div class="nx-brand-sub">Brain MRI Analysis</div></div></div>"""
    )
    render_html('<div class="nx-nav-label">Navigation</div>')
    page = st.radio(
        "Navigation",
        PAGES,
        format_func=lambda name: f"{PAGE_ICONS[name]}  {name}",
        label_visibility="collapsed",
    )
    render_html(
        '<div class="nx-side-note">Research prototype for academic use. '
        "Not intended for clinical diagnosis.</div>"
    )


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
        metric_card("\U0001F3AF", "Test Accuracy", f"{TEST_ACCURACY:.2f}%")
    with c2:
        metric_card("\U0001F5BC\uFE0F", "Test Images", "1,600")
    with c3:
        metric_card("\U0001F9EC", "Tumor Categories", "4")
    with c4:
        metric_card("\U0001F9EA", "Selected Features", "229")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    section("Analysis Workflow")
    steps = [
        ("01", "\U0001F4E4", "MRI Input", "Upload a brain MRI scan for analysis."),
        ("02", "\U0001F9E9", "MDU-Net Segmentation", "Produces a segmentation response for the scan."),
        ("03", "\U0001F52C", "SwinCT Feature Extraction", "Extracts deep image features from the MRI."),
        ("04", "\U0001F39B\uFE0F", "OOA Feature Selection", "Selects the optimized 229-feature subset."),
        ("05", "\U0001F9E0", "HCAC Classification", "Classifies the scan into one of four categories."),
        ("06", "\U0001F4A1", "Explainable AI", "Grad-CAM, LIME and SHAP visual explanations."),
    ]
    for row_start in (0, 3):
        cols = st.columns(3, gap="medium")
        for col, (number, icon, title, description) in zip(cols, steps[row_start:row_start + 3]):
            with col:
                render_html(
                    f"""<div class="nx-step">
                    <div class="nx-step-top"><span class="nx-step-num">{number}</span>
                    <span class="nx-step-icon">{icon}</span></div>
                    <div class="nx-step-title">{esc(title)}</div>
                    <div class="nx-step-desc">{esc(description)}</div></div>"""
                )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    section("Classification Categories")
    dot_colors = [COLOR_CYAN, COLOR_TEAL, COLOR_BLUE, "#a78bfa"]
    cols = st.columns(4, gap="medium")
    for col, name, color in zip(cols, CLASS_NAMES, dot_colors):
        with col:
            render_html(
                f"""<div class="nx-cat"><span class="nx-cat-dot" style="background:{color}"></span>
                <div><div class="nx-cat-name">{esc(name)}</div>
                <div class="nx-cat-sub">400 test images</div></div></div>"""
            )


def render_results(latest):
    result = latest["result"]
    reference = latest["reference"]

    section("Classification Result")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        metric_card("\U0001F3AF", "Predicted Class", result["prediction"])
    with c2:
        metric_card(
            "\U0001F4F6",
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
    st.caption("Raw capsule scores are model outputs, not calibrated probabilities.")

    st.markdown("---")
    section("MDU-Net Response Visualization")
    render_response_maps(latest["image"], result["mask"])


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

    try:
        image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    except Exception as error:
        st.error(f"Could not read this image: {error}")
        return

    signature = f"{uploaded_file.name}:{uploaded_file.size}"
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
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        run_analysis = stretch_call(st.button, "Analyze MRI", type="primary")

    if run_analysis:
        try:
            with st.spinner("Analyzing MRI..."):
                metadata = load_metadata()
                models = load_models()
                result = analyze_mri(image, metadata, models)
        except Exception as error:
            st.error(f"Could not analyze this image: {error}")
            st.info(
                "Verify that the model checkpoints, hcac_metadata.json and "
                "model_architectures_combined.py are present in the project folders."
            )
            return

        reference = None if known_label == "Not specified" else known_label
        st.session_state["latest"] = {
            "signature": signature,
            "filename": uploaded_file.name,
            "image": image,
            "result": result,
            "reference": reference,
        }
        if reference is None:
            match_status = "Not evaluated"
        else:
            match_status = "Match" if reference == result["prediction"] else "Mismatch"
        st.session_state.setdefault("prediction_history", []).append({
            "Filename": uploaded_file.name,
            "Predicted Class": result["prediction"],
            "Prediction Strength (%)": round(result["prediction_strength"] * 100, 2),
            "Reference Class": reference if reference is not None else "Not specified",
            "Match Status": match_status,
        })

    latest = st.session_state.get("latest")
    if latest is not None and latest["signature"] == signature:
        st.markdown("---")
        render_results(latest)


def page_xai():
    page_header(
        "EXPLAINABLE ANALYSIS",
        "Segmentation & XAI",
        "Live MDU-Net response and saved explainability visualizations.",
    )

    latest = st.session_state.get("latest")
    if latest is not None:
        section("Latest Uploaded MRI", latest["filename"])
        render_response_maps(latest["image"], latest["result"]["mask"])
        st.markdown("---")

    section("Saved Explainability Results")
    tabs = st.tabs(list(XAI_QUERIES.keys()))
    for tab, (name, (required, excluded, preferred, example)) in zip(tabs, XAI_QUERIES.items()):
        with tab:
            matches = find_images(required, excluded, preferred)
            if matches:
                show_image(str(matches[0]), caption=matches[0].name)
                if len(matches) > 1:
                    with st.expander(f"More {name} images ({len(matches) - 1})"):
                        for extra in matches[1:6]:
                            show_image(str(extra), caption=extra.name)
            else:
                missing_file_note(f"No saved {name} visualization", example)


def page_performance():
    page_header(
        "EVALUATION",
        "Model Performance",
        "Test-set results and training curves saved by the project.",
    )

    m1, m2, m3, m4 = st.columns(4, gap="medium")
    with m1:
        metric_card("\U0001F3AF", "Test Accuracy", f"{TEST_ACCURACY:.2f}%")
    with m2:
        metric_card("\U0001F5BC\uFE0F", "Test Images", "1,600")
    with m3:
        metric_card("\U0001F9EC", "Classes", "4")
    with m4:
        metric_card("\u2696\uFE0F", "Images per Class", "400")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    section("Classification Report")
    report_display = REPORT_DF.copy()
    for column in ("Precision", "Recall", "F1-score"):
        report_display[column] = report_display[column].map(lambda v: f"{v:.2f}")
    render_table(report_display, numeric_cols=("Precision", "Recall", "F1-score", "Support"))

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    chart_cols = st.columns(3, gap="medium")
    for col, (metric, color) in zip(
        chart_cols,
        [("Precision", COLOR_TEAL), ("Recall", COLOR_CYAN), ("F1-score", COLOR_BLUE)],
    ):
        with col:
            image_label(f"{metric} by Class")
            show_chart(metric_bar_chart(metric, color))

    st.markdown("---")
    section("Confusion Matrix")
    confusion_path = find_confusion_matrix()
    if confusion_path is not None:
        _, middle, _ = st.columns([1, 2, 1])
        with middle:
            show_image(str(confusion_path), caption="HCAC test-set confusion matrix")
    else:
        missing_file_note("The saved HCAC confusion matrix", "hcac_confusion_matrix.png")

    st.markdown("---")
    section("Training Curves")
    accuracy_path = find_swinct_accuracy()
    loss_path = find_swinct_loss()
    mdunet_path = find_mdunet_loss()

    left, right = st.columns(2, gap="large")
    with left:
        image_label("SwinCT \u2013 Accuracy")
        if accuracy_path is not None:
            show_image(str(accuracy_path))
        else:
            missing_file_note("The SwinCT accuracy plot", "swinct_accuracy.png")
    with right:
        image_label("SwinCT \u2013 Loss")
        if loss_path is not None:
            show_image(str(loss_path))
        else:
            missing_file_note("The SwinCT loss plot", "swinct_loss.png")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    _, middle, _ = st.columns([1, 2, 1])
    with middle:
        image_label("MDU-Net \u2013 Segmentation Loss")
        if mdunet_path is not None:
            show_image(str(mdunet_path))
        else:
            missing_file_note("The MDU-Net segmentation loss plot", "mdunet_segmentation_loss.png")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    if st.button("Rescan result folders", key="rescan_results"):
        index_result_images.clear()
        st.rerun()


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
    """Keyed container so every page change replays the same 180 ms fade."""
    key = "nx-page-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    try:
        return st.container(key=key)
    except TypeError:
        return st.container()


with page_container(page):
    PAGE_FUNCTIONS[page]()

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
render_html(f'<div class="nx-footer">{esc(APP_TITLE)} \u00b7 Research prototype, not for clinical use</div>')
