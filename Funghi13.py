"""
Streamlit app: Fungi Classifier + Multi‑XAI Dashboard
----------------------------------------------------
* Classifies 48 mushroom species (ResNet50 backbone stored on Dropbox)
* Explanation methods selectable in sidebar:
  ‑ Grad‑CAM (baseline)
  ‑ Grad‑CAM++
  ‑ Integrated Gradients
  ‑ Occlusion Sensitivity
* UX extras: TOP‑k bar‑chart, heat‑map alpha slider, safety banner, logging CSV
* Professor‑mode flags: simulate misclassification, switch backbone (future)

Install requirements (if missing):
    pip install streamlit tensorflow tf‑keras‑vis pillow numpy opencv-python
"""

from __future__ import annotations
import os, csv, datetime, pathlib, requests

import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import streamlit as st

# ⇢ Optional XAI libs (Grad‑CAM++, Integrated Gradients)
try:
    from tf_keras_vis.gradcam import Gradcam
    from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
    from tf_keras_vis.integrated_gradients import IntegratedGradients
    from tf_keras_vis.utils.scores import CategoricalScore
    _XAI_AVAILABLE = True
except ModuleNotFoundError:
    _XAI_AVAILABLE = False

######################################################################
# 1. DATA & MODELS ###################################################
######################################################################
MODEL_URL = (
    "https://www.dropbox.com/scl/fi/437k0jr5hvzzyfyrp50z2/"
    "fungi_classifier_model.h5?rlkey=2tar5m1btexq24y6cf2inosnf&dl=1"
)
MODEL_PATH = "fungi_classifier_model.h5"
LABEL_PATH = "class_labels.txt"
LOG_PATH = "predictions_log.csv"

@st.cache_resource(show_spinner=True, ttl=60 * 60 * 24)
def load_model() -> tf.keras.Model:
    """Download (if necessary) and cache Keras model."""
    if not os.path.isfile(MODEL_PATH):
        with st.spinner("📥 Scaricamento del modello (~80 MB)…"):
            r = requests.get(MODEL_URL, stream=True, timeout=60)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

@st.cache_resource
def load_labels() -> list[str]:
    with open(LABEL_PATH) as f:
        return [ln.strip() for ln in f]

model = load_model()
SPECIES = load_labels()
NUM_CLASSES = len(SPECIES)

######################################################################
# 2. PRE/POST‑PROCESSING #############################################
######################################################################

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    arr = cv2.resize(np.array(img), (128, 128)) / 255.0
    return arr[None, ...].astype("float32")


def predict(img_arr: np.ndarray) -> tuple[str, float, np.ndarray]:
    preds = model.predict(img_arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return SPECIES[idx], float(preds[idx]), preds

######################################################################
# 3. XAI UTILITIES ###################################################
######################################################################

def _norm_heatmap(hm: np.ndarray) -> np.ndarray:
    hm = np.maximum(hm, 0)
    if hm.max() > 0:
        hm /= hm.max()
    return cv2.resize(hm, (128, 128))


def gradcam_heatmap(img_arr: np.ndarray, idx: int) -> np.ndarray:
    """Vanilla Grad‑CAM using tf‑keras‑vis."""
    if not _XAI_AVAILABLE:
        st.error("Installa tf‑keras‑vis per usare Grad‑CAM")
        return np.zeros((128, 128))
    score = CategoricalScore([idx])
    cam = Gradcam(model)
    heatmap = cam(score, img_arr)[0]
    return _norm_heatmap(heatmap)


def gradcampp_heatmap(img_arr: np.ndarray, idx: int) -> np.ndarray:
    if not _XAI_AVAILABLE:
        st.error("Grad‑CAM++ richiede tf‑keras‑vis")
        return np.zeros((128, 128))
    score = CategoricalScore([idx])
    campp = GradcamPlusPlus(model)
    heatmap = campp(score, img_arr)[0]
    return _norm_heatmap(heatmap)


def integrated_gradients(img_arr: np.ndarray, idx: int) -> np.ndarray:
    if not _XAI_AVAILABLE:
        st.error("Integrated Gradients richiede tf‑keras‑vis")
        return np.zeros((128, 128))
    score = CategoricalScore([idx])
    ig = IntegratedGradients(model)
    attributions = ig(score, img_arr, steps=24)[0].mean(axis=-1)
    return _norm_heatmap(attributions)


def occlusion_map(img: Image.Image, patch: int, stride: int, idx: int) -> np.ndarray:
    img_np = cv2.resize(np.array(img.convert("RGB")), (128, 128))
    h, w, _ = img_np.shape
    base_pred = model.predict(img_np[None]/255.0, verbose=0)[0][idx]
    heat = np.zeros((h, w), dtype="float32")
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            occl = img_np.copy()
            occl[y:y+patch, x:x+patch] = 0
            p = model.predict(occl[None]/255.0, verbose=0)[0][idx]
            heat[y:y+patch, x:x+patch] = base_pred - p  # drop in prob
    return _norm_heatmap(heat)


def overlay(image: Image.Image, heatmap: np.ndarray, alpha: float) -> np.ndarray:
    img = cv2.resize(np.array(image.convert("RGB")), (128, 128))
    cmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(cmap, alpha, img, 1 - alpha, 0)
    return blended

######################################################################
# 4. UI ##############################################################
######################################################################

st.set_page_config("Fungi Classifier XAI", layout="centered")
st.title("🍄 Classificatore di Funghi con Explainability")
st.info("⚠️ **Non consumare funghi basandoti solo su questo modello** — può sbagliare!")

# ─ Sidebar controls ───────────────────────────────────────────────┐
with st.sidebar:
    st.header("🔧 Parametri")
    xai_method = st.selectbox(
        "Metodo di spiegazione",
        ["Grad‑CAM", "Grad‑CAM++", "Integrated Gradients", "Occlusion Sensitivity"],
    )
    alpha = st.slider("Trasparenza heat‑map", 0.1, 0.8, 0.4, 0.05)
    if xai_method == "Occlusion Sensitivity":
        patch = st.slider("Dimensione patch", 4, 32, 16, 2)
        stride = st.slider("Stride", 4, 32, 8, 2)
    else:
        patch = stride = None
    topk = st.slider("Mostra TOP‑k specie", 3, 10, 5)
    simulate_error = st.checkbox("Simula misclassificazione (20 % prob.)")
# ─ End sidebar ────────────────────────────────────────────────────┘

uploaded = st.file_uploader("Carica foto del fungo", ["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="📷 Immagine caricata", use_column_width=True)

    arr = preprocess(img)

    pred_label, conf, preds = predict(arr)

    # Optionally flip prediction to 2nd best to simulate error
    if simulate_error and np.random.rand() < 0.2:
        alt_idx = int(np.argsort(preds)[-2])
        pred_label = SPECIES[alt_idx]
        conf = preds[alt_idx]

    st.subheader("🔎 Risultato")
    st.markdown(f"**Specie predetta:** `{pred_label}`  ")
    st.markdown(f"**Confidenza:** `{conf*100:.1f}%`")

    # TOP‑k bar chart
    top_indices = np.argsort(preds)[::-1][:topk]
    top_scores = {SPECIES[i]: float(preds[i]) for i in top_indices}
    st.bar_chart(top_scores)

    # Heat‑map
    idx_pred = int(np.argmax(preds))
    if xai_method == "Grad‑CAM":
        heat = gradcam_heatmap(arr, idx_pred)
    elif xai_method == "Grad‑CAM++":
        heat = gradcampp_heatmap(arr, idx_pred)
    elif xai_method == "Integrated Gradients":
        heat = integrated_gradients(arr, idx_pred)
    else:  # Occlusion
        heat = occlusion_map(img, patch or 16, stride or 8, idx_pred)

    blended = overlay(img, heat, alpha)
    st.image(blended, caption=f"{xai_method} overlay", use_column_width=True)

    # ========== Logging ==========
    try:
        write_header = not pathlib.Path(LOG_PATH).exists()
        with open(LOG_PATH, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(["timestamp", "filename", "predicted", "confidence", "method"])
            writer.writerow([
                datetime.datetime.now().isoformat(timespec="seconds"),
                uploaded.name,
                pred_label,
                f"{conf:.4f}",
                xai_method,
            ])
    except Exception as e:
        st.warning(f"Impossibile salvare log: {e}")

else:
    st.caption("⬆️ Carica un'immagine per iniziare…")

