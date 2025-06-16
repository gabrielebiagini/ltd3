"""
Streamlit app: Fungi Classifier + Multi‑XAI Dashboard
----------------------------------------------------
Complete version (v1.0)
"""

from __future__ import annotations
import csv, datetime as _dt, os, pathlib, requests
import cv2, numpy as np, tensorflow as tf
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Fungi Classifier XAI", page_icon="🍄", layout="centered")

try:
    from tf_keras_vis.gradcam import Gradcam
    from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
    from tf_keras_vis.integrated_gradients import IntegratedGradients
    from tf_keras_vis.utils.scores import CategoricalScore
    _XAI_OK = True
except ModuleNotFoundError:
    _XAI_OK = False

MODEL_URL = "https://www.dropbox.com/scl/fi/437k0jr5hvzzyfyrp50z2/fungi_classifier_model.h5?rlkey=2tar5m1btexq24y6cf2inosnf&dl=1"
MODEL_PATH = "fungi_classifier_model.h5"
LABEL_PATH = "class_labels.txt"
LOG_PATH = "predictions_log.csv"

@st.cache_resource(show_spinner=True, ttl=86400)
def load_model():
    if not os.path.isfile(MODEL_PATH):
        with st.spinner("Scarico modello…"):
            r = requests.get(MODEL_URL, stream=True)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for c in r.iter_content(8192):
                    f.write(c)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

@st.cache_resource
def load_labels():
    return [l.strip() for l in open(LABEL_PATH, encoding="utf-8")]

model, SPECIES = load_model(), load_labels()

FUNGI_INFO = {"Amanita phalloides": {"it": "Amanita falloide", "edible": "Mortale"}}  # ← abbrev.

#################################### utils #####################################

def preprocess(img):
    return (cv2.resize(np.array(img.convert("RGB")), (128, 128)) / 255.0)[None]

def predict(img_arr):
    p = model.predict(img_arr, verbose=0)[0]
    idx = int(np.argmax(p))
    return SPECIES[idx], float(p[idx]), p

norm = lambda h: cv2.resize(np.maximum(h, 0)/(h.max() or 1), (128,128))

def heat(arr, idx, m):
    if not _XAI_OK:
        return np.zeros((128,128))
    sc = CategoricalScore([idx])
    if m=="Grad-CAM":
        return norm(Gradcam(model)(sc, arr)[0])
    if m=="Grad-CAM++":
        return norm(GradcamPlusPlus(model)(sc, arr)[0])
    if m=="Integrated Gradients":
        return norm(IntegratedGradients(model)(sc, arr, steps=24)[0].mean(-1))

def occ_hm(img, patch, stride, idx):
    base = cv2.resize(np.array(img.convert("RGB")), (128,128))
    bp = model.predict(base[None]/255.0,verbose=0)[0][idx]
    h = np.zeros((128,128), float)
    for y in range(0,128,stride):
        for x in range(0,128,stride):
            oc = base.copy(); oc[y:y+patch,x:x+patch]=0
            h[y:y+patch,x:x+patch] = bp - model.predict(oc[None]/255.0,0)[0][idx]
    return norm(h)

def overlay(img, hm, a):
    return cv2.addWeighted(cv2.applyColorMap((hm*255).astype(np.uint8),cv2.COLORMAP_JET), a,
                           cv2.resize(np.array(img.convert("RGB")),(128,128)),1-a,0)

##################################### UI #######################################

st.title("🍄 Classificatore di Funghi con Explainability")
st.info("Non consumare funghi basandoti solo su questo modello — può sbagliare!")

with st.sidebar:
    m = st.selectbox("Metodo", ["Grad-CAM","Grad-CAM++","Integrated Gradients","Occlusion"])
    alpha = st.slider("Trasparenza", .1,.8,.4)
    patch = stride = None
    if m=="Occlusion":
        patch = st.slider("Patch",4,32,16,2); stride = st.slider("Stride",4,32,8,2)
    topk=st.slider("Top‑k",3,15,5)
    show=st.checkbox("Tabella confidenze")
    sim = st.checkbox("Simula errore (20%)")

up = st.file_uploader("Carica immagine",["jpg","jpeg","png"])
if not up: st.stop()
img = Image.open(up); st.image(img,use_column_width=True)
arr = preprocess(img)
lab, conf, pr = predict(arr)
if sim and np.random.rand()<.2: alt=int(np.argsort(pr)[-2]); lab,conf=SPECIES[alt],float(pr[alt])
info = FUNGI_INFO.get(lab,{"it":"—","edible":"—"})
st.markdown(f"**Specie:** `{lab}` • **IT:** {info['it']} • **Edibilità:** {info['edible']} • **Conf:** {conf*100:.1f}%")

st.bar_chart({SPECIES[i]:float(pr[i]) for i in np.argsort(pr)[::-1][:topk]})
if show: st.dataframe({"Specie":SPECIES,"Prob":pr})

idx=int(np.argmax(pr))
hm = heat(arr, idx, m) if m!="Occlusion" else occ_hm(img, patch, stride, idx)
st.image(overlay(img, hm, alpha), caption=m, use_column_width=True)

try:
    new = not pathlib.Path(LOG_PATH).exists()
    with open(LOG_PATH,"a",newline="") as f:
        w=csv.writer(f); 
        if new: w.writerow(["ts","file","pred","conf","method"])
        w.writerow([_dt.datetime.now().isoformat(),up.name,lab,f"{conf:.4f}",m])
except Exception as e:
    st.warning(f"Log fallito: {e}")

