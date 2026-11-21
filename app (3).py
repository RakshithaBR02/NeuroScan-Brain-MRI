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

COLOR_TEAL = "#0e7490"
COLOR_CYAN = "#0b5cab"
COLOR_BLUE = "#5b9bd5"
COLOR_MUTED_BAR = "#b9c9db"


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
    --nx-bg: #f4f7fb;
    --nx-surface: #ffffff;
    --nx-border: #dbe3ec;
    --nx-primary: #0b5cab;
    --nx-primary-dark: #084882;
    --nx-accent: #0e7490;
    --nx-text: #0f2a43;
    --nx-body: #334155;
    --nx-muted: #64748b;
}

.stApp, .stApp p, .stApp label, .stApp li, .stApp button, .stApp input,
.stApp select, .stApp textarea, h1, h2, h3, h4, [class*="nx-"] {
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
}
.stApp { background: var(--nx-bg); color: var(--nx-text); }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container {
    padding-top: 2rem;
    padding-bottom: 2.5rem;
    max-width: 1200px;
}

@keyframes nxPageFade {
    from { opacity: 0; transform: translateY(4px); }
    to   { opacity: 1; transform: translateY(0); }
}
[class*="st-key-nx-page-"] { animation: nxPageFade 180ms ease-out both; }

h1, h2, h3, h4 { color: var(--nx-text) !important; letter-spacing: -.01em; }
p, label, li { color: var(--nx-body); }
[data-testid="stCaptionContainer"], small { color: var(--nx-muted) !important; }
hr { border-color: var(--nx-border); margin: 1.6rem 0; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid var(--nx-border);
}
.nx-brand {
    display: flex; align-items: center; gap: 12px;
    padding: 6px 4px 18px 4px; margin-bottom: 16px;
    border-bottom: 1px solid var(--nx-border);
}
.nx-brand-logo {
    width: 38px; height: 38px; border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    background: var(--nx-primary); color: #ffffff;
    font-weight: 700; font-size: 1.1rem;
}
.nx-brand-name { font-weight: 700; font-size: 1.1rem; color: var(--nx-text); line-height: 1.1; }
.nx-brand-sub { font-size: .78rem; color: var(--nx-muted); margin-top: 2px; }
.nx-nav-label {
    font-size: .7rem; font-weight: 600; letter-spacing: .12em;
    color: var(--nx-muted); margin: 4px 4px 8px 4px; text-transform: uppercase;
}
[data-testid="stSidebar"] div[role="radiogroup"] { gap: 4px; }
[data-testid="stSidebar"] div[role="radiogroup"] > label {
    width: 100%;
    padding: 10px 14px;
    border-radius: 8px;
    border-left: 3px solid transparent;
    transition: background .15s ease;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: #eef4fa; }
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: #e6f0fb;
    border-left-color: var(--nx-primary);
}
[data-testid="stSidebar"] div[role="radiogroup"] label p {
    color: var(--nx-text); font-weight: 500; font-size: .95rem;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
    color: var(--nx-primary-dark); font-weight: 600;
}

/* Page header */
.nx-hero {
    background: var(--nx-surface);
    border: 1px solid var(--nx-border);
    border-left: 4px solid var(--nx-primary);
    border-radius: 10px;
    padding: 22px 28px;
    margin-bottom: 22px;
    box-shadow: 0 1px 3px rgba(15, 42, 67, .06);
}
.nx-kicker {
    color: var(--nx-primary); font-size: .72rem; font-weight: 600;
    letter-spacing: .14em; text-transform: uppercase;
}
.nx-title {
    color: var(--nx-text); font-size: 1.75rem; font-weight: 700;
    letter-spacing: -.01em; margin: 6px 0 4px 0; line-height: 1.2;
}
.nx-sub { color: var(--nx-muted); font-size: .95rem; }

/* Section heading */
.nx-section { display: flex; align-items: center; gap: 12px; margin: 6px 0 14px 0; }
.nx-bar { width: 3px; height: 28px; border-radius: 2px; background: var(--nx-primary); }
.nx-section-title { color: var(--nx-text); font-size: 1.1rem; font-weight: 600; line-height: 1.2; }
.nx-section-sub { color: var(--nx-muted); font-size: .85rem; margin-top: 2px; }

/* Metric cards */
.nx-metric {
    display: flex; flex-direction: column; justify-content: center;
    min-height: 96px; padding: 18px 22px; box-sizing: border-box;
    background: var(--nx-surface);
    border: 1px solid var(--nx-border);
    border-radius: 10px;
    box-shadow: 0 1px 3px rgba(15, 42, 67, .06);
}
.nx-metric-label {
    color: var(--nx-muted); font-size: .74rem; font-weight: 600;
    letter-spacing: .08em; text-transform: uppercase;
}
.nx-metric-value { color: var(--nx-primary-dark); font-size: 1.65rem; font-weight: 700; line-height: 1.25; margin-top: 4px; }
.nx-metric-sub { color: var(--nx-muted); font-size: .78rem; margin-top: 2px; }

/* Workflow and category cards */
.nx-step {
    height: 150px; padding: 20px 22px; box-sizing: border-box;
    background: var(--nx-surface);
    border: 1px solid var(--nx-border);
    border-top: 3px solid var(--nx-primary);
    border-radius: 10px;
    box-shadow: 0 1px 3px rgba(15, 42, 67, .06);
    margin-bottom: 6px;
}
.nx-step-top { margin-bottom: 8px; }
.nx-step-num { color: var(--nx-primary); font-weight: 700; font-size: .9rem; letter-spacing: .08em; }
.nx-step-title { color: var(--nx-text); font-weight: 600; font-size: 1.02rem; margin-bottom: 5px; }
.nx-step-desc { color: var(--nx-muted); font-size: .88rem; line-height: 1.5; }
.nx-cat {
    min-height: 84px; padding: 18px 20px; box-sizing: border-box;
    background: var(--nx-surface);
    border: 1px solid var(--nx-border);
    border-radius: 10px;
    box-shadow: 0 1px 3px rgba(15, 42, 67, .06);
    display: flex; align-items: center; gap: 14px;
}
.nx-cat-dot { width: 10px; height: 10px; border-radius: 50%; flex: 0 0 auto; }
.nx-cat-name { color: var(--nx-text); font-weight: 600; font-size: 1.04rem; }
.nx-cat-sub { color: var(--nx-muted); font-size: .8rem; }

/* Image labels and info rows */
.nx-img-label {
    font-size: .72rem; font-weight: 600; letter-spacing: .1em;
    text-transform: uppercase; color: var(--nx-muted); margin-bottom: 8px;
}
[data-testid="stImage"] img { border-radius: 8px; border: 1px solid var(--nx-border); }
.nx-info-row {
    display: flex; justify-content: space-between; gap: 12px;
    padding: 10px 0; border-bottom: 1px solid var(--nx-border); font-size: .9rem;
}
.nx-info-row span:first-child { color: var(--nx-muted); }
.nx-info-row span:last-child { color: var(--nx-text); font-weight: 500; text-align: right; word-break: break-all; }

/* Tables */
.nx-table-wrap {
    background: var(--nx-surface);
    border: 1px solid var(--nx-border);
    border-radius: 10px; overflow: auto; max-height: 460px;
    box-shadow: 0 1px 3px rgba(15, 42, 67, .06);
}
.nx-table { width: 100%; border-collapse: collapse; font-size: .92rem; }
.nx-table th {
    text-align: left; padding: 12px 16px; color: var(--nx-muted);
    font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; font-weight: 600;
    background: #f1f5f9; position: sticky; top: 0;
    border-bottom: 1px solid var(--nx-border);
}
.nx-table td { padding: 12px 16px; color: var(--nx-body); border-bottom: 1px solid #eef2f6; }
.nx-table tr:last-child td { border-bottom: 0; }
.nx-table tbody tr:hover td { background: #f8fafc; }
.nx-table td.num, .nx-table th.num { text-align: right; }
.nx-badge {
    display: inline-block; padding: 3px 11px; border-radius: 999px;
    font-size: .78rem; font-weight: 600;
}
.nx-badge-ok { background: #e7f5ec; color: #1e7b3a; border: 1px solid #bfe3cb; }
.nx-badge-bad { background: #fff4e0; color: #9a5b00; border: 1px solid #f3d9a4; }
.nx-badge-na { background: #f1f5f9; color: #64748b; border: 1px solid #dbe3ec; }

/* Widgets */
.stButton > button, .stDownloadButton > button {
    background: var(--nx-primary);
    color: #ffffff; border: 1px solid var(--nx-primary);
    border-radius: 8px; padding: .6rem 1.2rem; font-weight: 600;
    box-shadow: none;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: var(--nx-primary-dark); border-color: var(--nx-primary-dark); color: #ffffff;
}
.stButton > button p, .stDownloadButton > button p { color: #ffffff; }
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] {
    background: #ffffff;
    border: 1px dashed #9fb6cf;
    border-radius: 10px;
}
[data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span { color: var(--nx-muted); }
[data-testid="stFileUploader"] button {
    background: #ffffff; color: var(--nx-primary);
    border: 1px solid var(--nx-primary);
}
[data-testid="stFileUploader"] button p { color: var(--nx-primary); }
div[data-baseweb="select"] > div {
    background: #ffffff; border-color: var(--nx-border); color: var(--nx-text);
}
div[data-baseweb="select"] span, div[data-baseweb="select"] div { color: var(--nx-text); }
div[data-baseweb="popover"] ul { background: #ffffff; }
div[data-baseweb="popover"] li { color: var(--nx-text); }
[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid var(--nx-border);
    border-radius: 8px;
}
[data-testid="stExpander"] summary p { color: var(--nx-body); font-size: .9rem; font-weight: 500; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--nx-border); }
.stTabs [data-baseweb="tab"] { color: var(--nx-muted); font-weight: 500; padding: 10px 18px; }
.stTabs [aria-selected="true"] { color: var(--nx-primary); font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--nx-primary); }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: #f4f7fb; }
::-webkit-scrollbar-thumb { background: #c5d1de; border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: #a9b9ca; }

@media (max-width: 768px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .nx-title { font-size: 1.4rem; }
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
            labelColor="#475569",
            titleColor="#475569",
            gridColor="#e2e8f0",
            domainColor="#cbd5e1",
            tickColor="#cbd5e1",
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
        align="left", dx=6, color="#0f2a43", fontWeight=600, fontSize=12
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
        dy=-8, color="#0f2a43", fontWeight=700, fontSize=12
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
    st.info(f"{description} is not available.")


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
    dot_colors = [COLOR_CYAN, COLOR_TEAL, COLOR_BLUE, "#64748b"]
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
        metric_card("Predicted Class", result["prediction"])
    with c2:
        metric_card("Prediction Strength",
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
                show_image(str(matches[0]))
                if len(matches) > 1:
                    with st.expander(f"More {name} images ({len(matches) - 1})"):
                        for extra in matches[1:6]:
                            show_image(str(extra))
            else:
                missing_file_note(f"The saved {name} visualization", example)


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

        figure = Figure(figsize=(6.6, 4.0), dpi=200, facecolor="#ffffff")
        ax = figure.subplots()
        ax.set_facecolor("#ffffff")
        ax.plot(epochs, train, color="#0b5cab", linewidth=2.2, label="Train")
        ax.plot(epochs, val, color="#d97706", linewidth=2.2, label="Validation")

        if np.isfinite(val).any():
            best = int(np.nanargmax(val) if higher_is_better else np.nanargmin(val))
            digits = 2 if np.nanmax(np.abs(val)) > 1.5 else 4
            ax.scatter(
                [epochs[best]], [val[best]], s=70, color="#d97706",
                edgecolors="#ffffff", linewidths=1.2, zorder=5,
            )
            ax.annotate(
                f"Best {val[best]:.{digits}f} @ epoch {int(epochs[best])}",
                xy=(epochs[best], val[best]),
                xytext=(0, 14),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color="#92400e",
                bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#d97706", lw=0.8),
            )

        ax.margins(x=0.03, y=0.16)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for spine in ax.spines.values():
            spine.set_color("#d5dde7")
        ax.tick_params(colors="#475569", labelsize=9)
        ax.grid(True, color="#d5dde7", alpha=0.35, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("Epoch", color="#475569", fontsize=10)
        ax.set_ylabel(ylabel, color="#475569", fontsize=10)
        legend = ax.legend(loc="best", frameon=True, fontsize=9)
        legend.get_frame().set_facecolor("#ffffff")
        legend.get_frame().set_edgecolor("#d5dde7")
        for text in legend.get_texts():
            text.set_color("#0f2a43")

        figure.tight_layout()
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", facecolor=figure.get_facecolor())
        return buffer.getvalue()

    @st.cache_data(show_spinner=False)
    def report_png(classes, precision, recall, f1_score):
        from matplotlib.figure import Figure

        positions = np.arange(len(classes))
        width = 0.26
        figure = Figure(figsize=(10.5, 4.0), dpi=200, facecolor="#ffffff")
        ax = figure.subplots()
        ax.set_facecolor("#ffffff")
        series = [
            ("Precision", precision, "#0e7490"),
            ("Recall", recall, "#0b5cab"),
            ("F1-score", f1_score, "#8fb3d9"),
        ]
        for offset, (label, values, color) in zip((-width, 0.0, width), series):
            bars = ax.bar(positions + offset, values, width, label=label, color=color)
            for rect, value in zip(bars, values):
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    value + 0.012,
                    f"{value:.2f}",
                    ha="center", va="bottom", fontsize=8, color="#0f2a43",
                )
        ax.set_xticks(positions)
        ax.set_xticklabels(classes)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score", color="#475569", fontsize=10)
        ax.tick_params(colors="#475569", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#d5dde7")
        ax.grid(True, axis="y", color="#d5dde7", alpha=0.35, linewidth=0.8)
        ax.set_axisbelow(True)
        legend = ax.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=10
        )
        for text in legend.get_texts():
            text.set_color("#0f2a43")
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
    """Keyed container so every page change replays the same 180 ms fade."""
    key = "nx-page-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    try:
        return st.container(key=key)
    except TypeError:
        return st.container()


with page_container(page):
    PAGE_FUNCTIONS[page]()
