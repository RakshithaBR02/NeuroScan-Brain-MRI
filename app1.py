import json
from pathlib import Path

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
st.set_page_config(
    page_title="Brain MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"
TRAINING_DIR = BASE_DIR / "training_results"
EVALUATION_DIR = BASE_DIR / "evaluation"
ARTIFACT_DIR = BASE_DIR / "app_artifacts"
ARTIFACT_MODEL_DIR = ARTIFACT_DIR / "models"
ARTIFACT_RESULT_DIR = ARTIFACT_DIR / "results"

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

# =========================================================
# STYLING
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background:
      radial-gradient(circle at 15% 5%, rgba(0, 191, 255, .10), transparent 28%),
      radial-gradient(circle at 90% 15%, rgba(45, 100, 220, .12), transparent 30%),
      #07111f;
    color: #eaf3ff;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1b30 0%, #07111f 100%);
    border-right: 1px solid rgba(74, 170, 255, .18);
}
.block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; }
/* Consistent short transition on every page change */
@keyframes pageFadeIn {
    from { opacity: 0; transform: translateY(7px); }
    to { opacity: 1; transform: translateY(0); }
}
[data-testid="stAppViewContainer"] .main .block-container {
    animation: pageFadeIn .18s ease-out both;
}

h1, h2, h3 { color: #f2f8ff !important; }
p, label, span { color: #d4e3f5; }
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(19, 43, 70, .94), rgba(10, 26, 45, .96));
    border: 1px solid rgba(65, 174, 255, .24);
    border-radius: 15px;
    padding: 17px 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,.16);
}
div[data-testid="stMetricLabel"] { color: #9db9d6 !important; }
div[data-testid="stMetricValue"] { color: #55c8ff !important; }
.stButton > button {
    background: linear-gradient(90deg, #087fbd, #1766d1);
    color: white;
    border: 0;
    border-radius: 10px;
    padding: .65rem 1.2rem;
    font-weight: 700;
}
.stButton > button:hover {
    border: 1px solid #68d4ff;
    color: white;
    background: linear-gradient(90deg, #0996d8, #2879ed);
}
div[data-testid="stFileUploader"] {
    background: rgba(15, 39, 64, .65);
    border: 1px dashed rgba(81, 190, 255, .55);
    border-radius: 14px;
    padding: 12px;
}
.hcac-hero {
    background: linear-gradient(115deg, rgba(12,44,75,.98), rgba(12,29,54,.94));
    border: 1px solid rgba(70,177,255,.30);
    border-radius: 20px;
    padding: 26px 30px;
    margin-bottom: 20px;
    box-shadow: 0 12px 35px rgba(0,0,0,.20);
}
.hcac-kicker {
    color: #55c8ff;
    font-size: .78rem;
    font-weight: 800;
    letter-spacing: .16em;
    text-transform: uppercase;
}
.hcac-sub { color: #a9c4df; margin-top: 6px; }
.hcac-card {
    background: rgba(13, 34, 57, .78);
    border: 1px solid rgba(75, 165, 235, .20);
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 12px;
    min-height: 178px;
    min-height: 178px;
    height: auto;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-sizing: border-box;
}
.hcac-card h3 { margin: 7px 0 10px 0 !important; font-size: 1.18rem; }
.hcac-card p { margin-bottom: 0; }
hr { border-color: rgba(96, 163, 220, .18); }
small { color: #94afca !important; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# FILE / CHECKPOINT HELPERS
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


@st.cache_resource(show_spinner="Loading HCAC models...")
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
# MRI INFERENCE
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

    score_order = np.argsort(capsule_scores)[::-1]
    predicted_index = int(score_order[0])
    second_index = int(score_order[1])
    top_score = float(capsule_scores[predicted_index])
    second_score = float(capsule_scores[second_index])
    score_margin = top_score - second_score

    return {
        "prediction": CLASS_NAMES[predicted_index],
        "prediction_index": predicted_index,
        "prediction_strength": top_score,
        "second_prediction": CLASS_NAMES[second_index],
        "second_score": second_score,
        "score_margin": float(score_margin),
        "capsule_scores": capsule_scores,
        "mask": segmentation_mask,
    }


# =========================================================
# DISPLAY HELPERS
# =========================================================
def section_title(title, subtitle=None):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def show_first_image(candidates, title=None, width=None):
    path = first_existing([BASE_DIR / item for item in candidates])
    if path:
        if title:
            st.markdown(f"**{title}**")
        st.image(str(path), use_container_width=(width is None), width=width)
        return True
    return False


def show_image_from_patterns(patterns, title):
    found = []
    for folder in [TRAINING_DIR, EVALUATION_DIR, RESULT_DIR, ARTIFACT_RESULT_DIR]:
        if folder.exists():
            for pattern in patterns:
                found.extend(folder.glob(pattern))
    # Keep unique files and show the first matching image.
    unique = list(dict.fromkeys(found))
    if unique:
        st.markdown(f"**{title}**")
        st.image(str(unique[0]), use_container_width=True)
        return True
    return False


def load_csv_if_exists(paths):
    path = first_existing([BASE_DIR / p for p in paths])
    if path:
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
with st.sidebar:
    st.markdown("## 🧠 Brain MRI Analysis")
    st.caption("Tumor Classification & Explainability")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "MRI Analysis",
            "Segmentation & XAI",
            "Model Performance",
            "Prediction History",
        ],
        label_visibility="collapsed",
    )


# =========================================================
# DASHBOARD
# =========================================================
if page == "Dashboard":
    st.markdown("""
    <div class="hcac-hero">
      <div class="hcac-kicker">HYBRID ATTENTIVE FRAMEWORK</div>
      <h1 style="margin:8px 0 0 0;">Brain Tumor MRI Analysis</h1>
      <div class="hcac-sub">An integrated workflow for MRI segmentation, feature extraction,
      feature selection, classification and explainable analysis.</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test Accuracy", f"{TEST_ACCURACY:.2f}%")
    c2.metric("Test Images", "1,600")
    c3.metric("Tumor Categories", "4")
    c4.metric("Feature Count", "229")

    st.markdown("---")
    section_title("Analysis Workflow", "The stages used by the application.")
    steps = [
        ("01", "MRI Input", "Upload a brain MRI image."),
        ("02", "MDU-Net", "Generate a segmentation output."),
        ("03", "SwinCT", "Extract image features."),
        ("04", "OOA", "Select the optimized feature subset."),
        ("05", "HCAC", "Predict one of four classes."),
        ("06", "XAI", "Review the available explanation outputs."),
    ]
    cols = st.columns(3)
    for i, (number, title, description) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(
                f"""<div class="hcac-card">
                <div style="color:#55c8ff;font-size:1.4rem;font-weight:800">{number}</div>
                <h3 style="margin:4px 0">{title}</h3>
                <div style="color:#a9c4df">{description}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    section_title("Classification Categories")
    category_cols = st.columns(4)
    for col, name in zip(category_cols, CLASS_NAMES):
        with col:
            st.markdown(
                f'<div class="hcac-card"><h3>{name}</h3></div>',
                unsafe_allow_html=True,
            )


# =========================================================
# MRI ANALYSIS
# =========================================================
elif page == "MRI Analysis":
    st.markdown("""
    <div class="hcac-hero">
      <div class="hcac-kicker">IMAGE CLASSIFICATION</div>
      <h1 style="margin:8px 0 0 0;">MRI Analysis</h1>
      <div class="hcac-sub">Upload an MRI image to run the trained analysis pipeline.</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload MRI image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        key="mri_uploader",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            left, right = st.columns([1, 1.15], gap="large")
            with left:
                st.markdown("### Uploaded MRI")
                st.image(image, use_container_width=True)
            with right:
                st.markdown("### Image Evaluation")
                known_label = st.selectbox(
                    "Known class (optional – for testing only)",
                    ["Not specified"] + CLASS_NAMES,
                    key="known_label",
                    help="Select the dataset label only when you know it. This does not change the model prediction.",
                )
                run_analysis = st.button(
                    "Analyze MRI",
                    type="primary",
                    use_container_width=True,
                )

            if run_analysis:
                with st.spinner("Running MRI analysis..."):
                    metadata = load_metadata()
                    models = load_models()
                    result = analyze_mri(image, metadata, models)

                st.session_state["latest_result"] = result
                st.session_state["latest_image"] = image
                st.session_state["latest_filename"] = uploaded_file.name
                st.session_state["latest_known_label"] = known_label

                correct = (
                    known_label == "Not specified"
                    or known_label == result["prediction"]
                )
                history_item = {
                    "Filename": uploaded_file.name,
                    "Known Class": known_label,
                    "Predicted Class": result["prediction"],
                    "Top Capsule Score (%)": round(result["prediction_strength"] * 100, 2),
                    "Second Class": result["second_prediction"],
                    "Second Capsule Score (%)": round(result["second_score"] * 100, 2),
                    "Correct": (
                        "Not evaluated" if known_label == "Not specified"
                        else ("Yes" if correct else "No")
                    ),
                }
                st.session_state.setdefault("prediction_history", []).append(history_item)

                st.success("Analysis completed.")
                st.markdown("---")
                section_title("Classification Result")
                c1, c2 = st.columns(2)
                c1.metric("Predicted Class", result["prediction"])
                c2.metric(
                    "Top Capsule Score",
                    f'{result["prediction_strength"] * 100:.2f}%',
                )

                st.caption(
                    "Capsule scores are model outputs, not calibrated probabilities. "
                    "A high score alone does not guarantee a correct classification."
                )

                if known_label != "Not specified":
                    if correct:
                        st.success(f"Prediction matches the supplied test label: {known_label}.")
                    else:
                        st.warning(
                            f"Misclassification on this test image: known label is "
                            f"{known_label}, while the model predicted {result['prediction']}. "
                            "The model output is shown unchanged; the app does not force the expected label."
                        )

                score_values = np.asarray(result["capsule_scores"], dtype=float)
                score_order = np.argsort(score_values)[::-1]
                score_df = pd.DataFrame({
                    "Class": [CLASS_NAMES[i] for i in score_order],
                    "Capsule Score": score_values[score_order],
                })
                st.markdown("### Class-wise Capsule Scores")
                st.bar_chart(score_df.set_index("Class"), horizontal=True)

                margin = result["score_margin"]
                top_two_col1, top_two_col2, top_two_col3 = st.columns(3)
                top_two_col1.metric("Top Class", result["prediction"])
                top_two_col2.metric("Second Class", result["second_prediction"])
                top_two_col3.metric("Score Difference", f"{margin:.4f}")

                if margin < 0.08:
                    st.warning(
                        "The two highest capsule scores are close. Treat this result as "
                        "ambiguous rather than relying only on the top-ranked class."
                    )

                st.markdown("---")
                section_title("MDU-Net Segmentation Output")
                mask = np.asarray(result["mask"], dtype=np.float32)
                binary_mask = (mask > 0.5).astype(np.uint8)
                low, high = float(mask.min()), float(mask.max())
                display_map = (
                    np.clip((mask - low) / (high - low), 0, 1)
                    if high > low else np.zeros_like(mask)
                )

                seg1, seg2 = st.columns(2, gap="large")
                with seg1:
                    st.markdown("**MDU-Net Probability Output**")
                    st.image(
                        display_map,
                        clamp=True,
                        caption="Contrast-adjusted display of the raw model output",
                        use_container_width=True,
                    )
                with seg2:
                    st.markdown("**Thresholded Output**")
                    st.image(
                        binary_mask * 255,
                        clamp=True,
                        caption="Binary output at threshold 0.5",
                        use_container_width=True,
                    )

                s1, s2, s3 = st.columns(3)
                s1.metric("Minimum Output", f"{low:.4f}")
                s2.metric("Maximum Output", f"{high:.4f}")
                s3.metric("Pixels Above 0.5", f"{binary_mask.mean() * 100:.2f}%")
                if binary_mask.sum() == 0:
                    st.caption("No pixels crossed the 0.5 threshold for this image.")
        except Exception as error:
            st.error(f"Could not analyze this image: {error}")
            st.info(
                "Verify that the model checkpoints, hcac_metadata.json and "
                "model_architectures_combined.py are present in the project folders."
            )
    else:
        st.info("Upload an MRI image to begin.")


# =========================================================
# SEGMENTATION & XAI
# =========================================================
elif page == "Segmentation & XAI":
    st.markdown("""
    <div class="hcac-hero">
      <div class="hcac-kicker">EXPLAINABLE ANALYSIS</div>
      <h1 style="margin:8px 0 0 0;">Segmentation & XAI</h1>
      <div class="hcac-sub">Visual outputs generated and saved during project evaluation.</div>
    </div>
    """, unsafe_allow_html=True)

    latest = st.session_state.get("latest_result")
    latest_image = st.session_state.get("latest_image")

    if latest is not None and latest_image is not None:
        section_title("Latest Uploaded MRI")
        col1, col2 = st.columns(2)
        with col1:
            st.image(latest_image, caption="Input MRI", use_container_width=True)
        with col2:
            latest_mask = np.asarray(latest["mask"], dtype=np.float32)
            lo, hi = float(latest_mask.min()), float(latest_mask.max())
            latest_display = (
                np.clip((latest_mask - lo) / (hi - lo), 0, 1)
                if hi > lo else np.zeros_like(latest_mask)
            )
            st.image(
                latest_display,
                clamp=True,
                caption="Contrast-enhanced view of MDU-Net output",
                use_container_width=True,
            )
        st.caption("Display contrast is enhanced for visibility; it is not a validated tumor mask.")
        st.markdown("---")

    section_title("Saved Explainability Results")
    tabs = st.tabs(["MDU-Net", "Grad-CAM", "LIME", "SHAP"])

    image_candidates = {
        "MDU-Net": [
            "mdunet_segmentation_example.png",
            "mdunet_segmentation.png",
            "mdunet_output.png",
            "segmentation_result.png",
        ],
        "Grad-CAM": [
            "gradcam_hcac_example.png",
            "gradcam_example.png",
            "gradcam.png",
            "*gradcam*.png",
        ],
        "LIME": [
            "lime_hcac_example.png",
            "lime_example.png",
            "*lime*.png",
        ],
        "SHAP": [
            "shap_hcac_example.png",
            "shap_example.png",
            "*shap*.png",
        ],
    }

    for tab, (name, patterns) in zip(tabs, image_candidates.items()):
        with tab:
            candidates = []
            for folder in [
                TRAINING_DIR,
                EVALUATION_DIR,
                RESULT_DIR,
                ARTIFACT_RESULT_DIR,
            ]:
                for pattern in patterns:
                    candidates.extend(folder.glob(pattern) if folder.exists() else [])
            candidates = list(dict.fromkeys(candidates))
            if candidates:
                st.image(str(candidates[0]), use_container_width=True)
            else:
                st.info(f"No saved {name} visualization was found in the results folders.")


# =========================================================
# MODEL PERFORMANCE
# =========================================================
elif page == "Model Performance":
    st.markdown("""
    <div class="hcac-hero">
      <div class="hcac-kicker">EVALUATION</div>
      <h1 style="margin:8px 0 0 0;">Model Performance</h1>
      <div class="hcac-sub">Test-set metrics and training curves saved by the project.</div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test Accuracy", "92.75%")
    m2.metric("Test Images", "1,600")
    m3.metric("Classes", "4")
    m4.metric("Images per Class", "400")

    st.markdown("---")
    section_title("Evaluation Metrics")
    metrics_df = load_csv_if_exists([
        "app_artifacts/results/hcac_final_test_metrics.csv",
        "evaluation/hcac_final_test_metrics.csv",
        "results/hcac_final_test_metrics.csv",
    ])
    if metrics_df is not None:
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    else:
        st.write(
            "The final evaluation on 1,600 test images achieved 92.75% accuracy."
        )
        report_df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Precision": [0.98, 0.91, 0.89, 0.95],
            "Recall": [0.79, 0.93, 1.00, 0.99],
            "F1-score": [0.87, 0.92, 0.94, 0.97],
            "Support": [400, 400, 400, 400],
        })
        st.dataframe(report_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    section_title("Confusion Matrix")

    result_folders = [ARTIFACT_RESULT_DIR, EVALUATION_DIR, RESULT_DIR, TRAINING_DIR]
    def find_saved_plot(keywords):
        matches = []
        for folder in result_folders:
            if folder.exists():
                for ext in ("*.png", "*.jpg", "*.jpeg"):
                    for candidate in folder.glob(ext):
                        name = candidate.stem.lower()
                        if all(word in name for word in keywords):
                            matches.append(candidate)
        return matches[0] if matches else None

    confusion_plot = (
        find_saved_plot(["hcac", "confusion"])
        or find_saved_plot(["confusion"])
    )
    if confusion_plot:
        st.image(
            str(confusion_plot),
            caption="HCAC test-set confusion matrix",
            use_container_width=True,
        )
    else:
        st.info("Confusion matrix image not found in the configured results folders.")

    st.markdown("---")
    section_title("Training Curves")

    accuracy_plot = (
        find_saved_plot(["swinct", "accuracy"])
        or find_saved_plot(["accuracy"])
    )
    swinct_loss_plot = find_saved_plot(["swinct", "loss"])
    mdunet_loss_plot = (
        find_saved_plot(["mdunet", "loss"])
        or find_saved_plot(["segmentation", "loss"])
    )

    curve_left, curve_right = st.columns(2, gap="large")
    with curve_left:
        st.markdown("### SwinCT – Accuracy")
        if accuracy_plot:
            st.image(str(accuracy_plot), use_container_width=True)
        else:
            st.info("SwinCT accuracy plot not found.")
    with curve_right:
        st.markdown("### SwinCT – Loss")
        if swinct_loss_plot:
            st.image(str(swinct_loss_plot), use_container_width=True)
        else:
            st.info("SwinCT loss plot not found.")

    if mdunet_loss_plot:
        st.markdown("### MDU-Net – Segmentation Loss")
        st.image(str(mdunet_loss_plot), use_container_width=True)


# =========================================================
# PREDICTION HISTORY
# =========================================================
elif page == "Prediction History":
    st.markdown("""
    <div class="hcac-hero">
      <div class="hcac-kicker">RECENT ACTIVITY</div>
      <h1 style="margin:8px 0 0 0;">Prediction History</h1>
      <div class="hcac-sub">Predictions made during this application session.</div>
    </div>
    """, unsafe_allow_html=True)

    history = st.session_state.get("prediction_history", [])
    if history:
        history_df = pd.DataFrame(history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Prediction History",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="hcac_prediction_history.csv",
            mime="text/csv",
        )
        if st.button("Clear History"):
            st.session_state["prediction_history"] = []
            st.rerun()
    else:
        st.info("No predictions have been recorded in this session yet.")


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#7894b2;font-size:.8rem;'>"
    "Brain MRI Analysis System"
    "</div>",
    unsafe_allow_html=True,
)
