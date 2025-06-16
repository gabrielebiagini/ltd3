"""
Streamlit app: Fungi Classifier + Multi‑XAI Dashboard
----------------------------------------------------
* Classifies 48 mushroom species (ResNet50 backbone stored on Dropbox)
* Explanation methods selectable in sidebar:
  – Grad‑CAM (baseline)
  – Grad‑CAM++
  – Integrated Gradients
  – Occlusion Sensitivity
* UX extras: TOP‑k bar‑chart, heat‑map alpha slider, safety banner, logging CSV
* Professor‑mode flags: simulate misclassification, switch backbone (future)

Install missing packages:
    pip install streamlit tensorflow tf-keras-vis pillow numpy opencv-python
"""

from __future__ import annotations
import csv
import datetime as _dt
import os
import pathlib
import requests

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st

# ──────────────────────────────── PAGE CONFIG (must be first st.* call)
st.set_page_config(page_title="Fungi Classifier XAI", page_icon="🍄", layout="centered")

# Optional XAI libs (Grad‑CAM++, Integrated Gradients)
try:
    from tf_keras_vis.gradcam import Gradcam
    from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
    from tf_keras_vis.integrated_gradients import IntegratedGradients
    from tf_keras_vis.utils.scores import CategoricalScore
    _XAI_OK = True
except ModuleNotFoundError:
    _XAI_OK = False

########################################################################
# 1. DATA & MODELS #####################################################
########################################################################
_MODEL_URL = (
    "https://www.dropbox.com/scl/fi/437k0jr5hvzzyfyrp50z2/"
    "fungi_classifier_model.h5?rlkey=2tar5m1btexq24y6cf2inosnf&dl=1"
)
_MODEL_PATH = "fungi_classifier_model.h5"
_LABEL_PATH = "class_labels.txt"
_LOG_PATH = "predictions_log.csv"

@st.cache_resource(show_spinner=True, ttl=60 * 60 * 24)
def load_model() -> tf.keras.Model:  # noqa: N802
    """Download (if necessary) and cache Keras model."""
    if not os.path.isfile(_MODEL_PATH):
        with st.spinner("📥 Scaricamento modello (~80 MB)…"):
            r = requests.get(_MODEL_URL, stream=True, timeout=60)
            r.raise_for_status()
            with open(_MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
    return tf.keras.models.load_model(_MODEL_PATH, compile=False)

@st.cache_resource
def load_labels() -> list[str]:
    with open(_LABEL_PATH, encoding="utf-8") as f:
        return [ln.strip() for ln in f]

model = load_model()
SPECIES = load_labels()
NUM_CLASSES = len(SPECIES)

########################################################################
# 2. PRE/POST‑PROCESSING ###############################################
########################################################################

def preprocess(img: Image.Image) -> np.ndarray:
    """Resize→norm→shape (1,128,128,3)."""
    arr = cv2.resize(np.array(img.convert("RGB")), (128, 128)) / 255.0
    return arr[np.newaxis, ...].astype("float32")


def predict(img_arr: np.ndarray) -> tuple[str, float, np.ndarray]:
    preds = model.predict(img_arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return SPECIES[idx], float(preds[idx]), preds

########################################################################
# 3. XAI UTILITIES #####################################################
########################################################################

def _norm(hm: np.ndarray) -> np.ndarray:
    hm = np.maximum(hm, 0)
    hm = hm / hm.max() if hm.max() else hm
    return cv2.resize(hm, (128, 128))


def gradcam_hm(arr: np.ndarray, idx: int) -> np.ndarray:
    if not _XAI_OK:
        st.error("Installa tf‑keras‑vis per Grad‑CAM")
        return np.zeros((128, 128))
    score = CategoricalScore([idx])
    cam = Gradcam(model)
    return _norm(cam(score, arr)[0])


def gradcampp_hm(arr: np.ndarray, idx: int) -> np.ndarray:
    if not _XAI_OK:
        st.error("Grad‑CAM++ richiede tf‑keras‑vis")
        return np.zeros((128, 128))
    score = CategoricalScore([idx])
    campp = GradcamPlusPlus(model)
    return _norm(campp(score, arr)[0])


def integgrads_hm(arr: np.ndarray, idx: int) -> np.ndarray:
    if not _XAI_OK:
        st.error("Integrated Gradients richiede tf‑keras‑vis")
        return np.zeros((128, 128))
    score = CategoricalScore([idx])
    ig = IntegratedGradients(model)
    attr = ig(score, arr, steps=24)[0].mean(-1)
    return _norm(attr)


def occlusion_hm(img: Image.Image, patch: int, stride: int, idx: int) -> np.ndarray:
    base = cv2.resize(np.array(img.convert("RGB")), (128, 128))
    h, w, _ = base.shape
    base_prob = model.predict(base[np.newaxis] / 255.0, verbose=0)[0][idx]
    heat = np.zeros((h, w), dtype="float32")
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            occl = base.copy()
            occl[y:y + patch, x:x + patch] = 0
            p = model.predict(occl[np.newaxis] / 255.0, verbose=0)[0][idx]
            heat[y:y + patch, x:x + patch] = base_prob - p
    return _norm(heat)


def overlay(img: Image.Image, hm: np.ndarray, alpha: float) -> np.ndarray:
    rgb = cv2.resize(np.array(img.convert("RGB")), (128, 128))
    cmap = cv2.applyColorMap(np.uint8(hm * 255), cv2.COLORMAP_JET)
    return cv2.addWeighted(cmap, alpha, rgb, 1 - alpha, 0)

########################################################################
# 4. UI ###############################################################
########################################################################

st.title("🍄 Classificatore di Funghi con Explainability")
st.info("⚠️ **Non consumare funghi basandoti solo su questo modello** — può sbagliare!")

with st.sidebar:
    st.header("🔧 Parametri")
    method = st.selectbox(
        "Metodo di spiegazione",
        ["Grad‑CAM", "Grad‑CAM++", "Integrated Gradients", "Occlusion Sensitivity"],
    )
    alpha = st.slider("Trasparenza heat‑map", 0.1, 0.8, 0.4, 0.05)
    if method == "Occlusion Sensitivity":
        patch = st.slider("Dimensione patch", 4, 32, 16, 2)
        stride = st.slider("Stride", 4, 32, 8, 2)
    else:
        patch = stride = None
    topk = st.slider("Mostra TOP‑k specie", 3, 10, 5)
    simulate_err = st.checkbox("Simula misclassificazione (20 %)")

uploaded = st.file_uploader("Carica foto del fungo", ["jpg", "jpeg", "png"])
if not uploaded:
    st.caption("⬆️ Carica un'immagine per iniziare…")
    st.stop()

img = Image.open(uploaded)
st.image(img, caption="📷 Immagine caricata", use_column_width=True)
arr = preprocess(img)
label, conf, preds = predict(arr)
if simulate_err and np.random.rand() < 0.2:
    alt_idx = int(np.argsort(preds)[-2])
    label, conf = SPECIES[alt_idx], float(preds[alt_idx])

st.subheader("🔎 Risultato")
st.markdown(f"**Specie predetta:** `{label}`  ")
st.markdown(f"**Confidenza:** `{conf*100:.1f}%`")

top_idx = np.argsort(preds)[::-1][:topk]
st.bar_chart({SPECIES[i]: float(preds[i]) for i in top_idx})

pred_idx = int(np.argmax(preds))
if method == "Grad‑CAM":
    hm = gradcam_hm(arr, pred_idx)
elif method == "Grad‑CAM++":
    hm = gradcampp_hm(arr, pred_idx)
elif method == "Integrated Gradients":
    hm = integgrads_hm(arr, pred_idx)
else:
    hm = occlusion_hm(img, patch or 16, stride or 8, pred_idx)

st.image(overlay(img, hm, alpha), caption=method, use_column_width=True)

# Logging
try:
    header = not pathlib.Path(_LOG_PATH).exists()
    with open(_LOG_PATH, "a", newline="") as f:
        wr = csv.writer(f)
        if header:
            wr.writerow(["ts", "file", "pred", "conf", "method"])
        wr.writerow([_dt.datetime.now().isoformat(timespec="seconds"), uploaded.name, label, f"{conf:.4f}", method])
except Exception as _e:
    st.warning(f"Log fallito: {_e}")
