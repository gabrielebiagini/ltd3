import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import requests
import os
import io
from lime import lime_image
from skimage.segmentation import mark_boundaries
from serpapi import SerpApiClient # <-- MODIFICATO: Importiamo la classe corretta
import gspread
from gspread_dataframe import set_with_dataframe
import pandas as pd

# --- CONFIGURAZIONE DELLA PAGINA STREAMLIT ---
st.set_page_config(
    page_title="Analisi Funghi con XAI",
    page_icon="🍄",
    layout="wide"
)

# ... (Tutto il dizionario FUNGI_INFO e la lista SPECIES_LIST rimangono invariati) ...
# Caricamento dell'ordine delle classi in modo robusto
try:
    with open('class_labels.txt', 'r') as f:
        SPECIES_LIST = [line.strip() for line in f]
except FileNotFoundError:
    st.error("Errore critico: il file 'class_labels.txt' non è stato trovato.")
    st.stop()

# Incolli qui il suo dizionario completo. Per brevità, è ridotto.
FUNGI_INFO = {
    "Agaricus bisporus": {"nome_italiano": "Prataiolo coltivato", "commestibile": "Commestibile"},
    "Amanita phalloides": {"nome_italiano": "Amanita falloide", "commestibile": "Mortale"},
    "Boletus edulis": {"nome_italiano": "Porcino", "commestibile": "Commestibile"},
    "Agaricus subrufescens": {"nome_italiano": "Prataiolo mandorlato", "commestibile": "Commestibile"},
    "Amanita bisporigera": {"nome_italiano": "Amanita bisporigera", "commestibile": "Velenoso"},
    "Amanita muscaria": {"nome_italiano": "Amanita muscaria", "commestibile": "Velenoso"},
    "Amanita ocreata": {"nome_italiano": "Amanita ocreata", "commestibile": "Velenoso"},
    "Amanita smithiana": {"nome_italiano": "Amanita smithiana", "commestibile": "Velenoso"},
    "Amanita verna": {"nome_italiano": "Amanita verna", "commestibile": "Mortale"},
    "Amanita virosa": {"nome_italiano": "Amanita virosa", "commestibile": "Mortale"},
    "Cantharellus cibarius": {"nome_italiano": "Gallinaccio", "commestibile": "Commestibile"},
    "Galerina marginata": {"nome_italiano": "Galerina marginata", "commestibile": "Mortale"},
    "Lepiota brunneoincarnata": {"nome_italiano": "Lepiota brunneoincarnata", "commestibile": "Mortale"}
}

# --- FUNZIONI (incluso il caricamento del modello e le funzioni XAI esistenti) ---
# ... (Tutte le funzioni da load_model a make_occlusion_sensitivity_map rimangono invariate) ...
@st.cache_resource
def load_model():
    model_url = 'https://www.dropbox.com/scl/fi/437k0jr5hvzzyfyrp50z2/fungi_classifier_model.h5?rlkey=2tar5m1btexq24y6cf2inosnf&dl=1'
    model_path = 'fungi_classifier_model.h5'
    if not os.path.isfile(model_path):
        st.write(f"Modello non trovato, scaricando da URL...")
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            st.success("Download completato.")
        except requests.exceptions.RequestException as e:
            st.error(f"Errore durante il download del modello: {e}")
            return None
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("Modello caricato correttamente.")
        return model
    except Exception as e:
        st.error(f"Errore nel caricamento del file del modello: {e}")
        return None

def preprocess_image(image: Image.Image):
    image = image.convert('RGB')
    img_array = np.array(image, dtype=np.uint8)
    img_array = cv2.resize(img_array, (128, 128))
    img_array_scaled = img_array / 255.0
    return np.expand_dims(img_array_scaled, axis=0), img_array

def predict_fungus(model, image_array):
    predictions = model.predict(image_array)[0]
    predicted_index = np.argmax(predictions)
    predicted_species = SPECIES_LIST[predicted_index]
    confidence = predictions[predicted_index] * 100
    info = FUNGI_INFO.get(predicted_species, {"nome_italiano": "N/A", "commestibile": "Sconosciuta"})
    return predicted_species, info, confidence, predictions * 100

def find_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4 and isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def display_superimposed_heatmap(original_image, heatmap, alpha=0.5):
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap_color, alpha, original_image, 1 - alpha, 0)
    return superimposed_img

@st.cache_data
def make_gradcam_heatmap(_model, img_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model([_model.inputs], [_model.get_layer(last_conv_layer_name).output, _model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

@st.cache_data
def explain_with_lime(_model, preprocessed_image_array):
    explainer = lime_image.LimeImageExplainer()
    prediction_fn = lambda x: _model.predict(x)
    explanation = explainer.explain_instance(preprocessed_image_array[0], prediction_fn, top_labels=1, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    lime_img = mark_boundaries(temp / 2 + 0.5, mask)
    return lime_img

@st.cache_data
def make_occlusion_sensitivity_map(_model, original_image_resized, patch_size=16):
    original_pred = _model.predict(np.expand_dims(original_image_resized / 255.0, axis=0))[0]
    original_pred_class_prob = np.max(original_pred)
    heatmap = np.zeros((original_image_resized.shape[0], original_image_resized.shape[1]), dtype=np.float32)
    for h in range(0, original_image_resized.shape[0], patch_size):
        for w in range(0, original_image_resized.shape[1], patch_size):
            occluded_image = original_image_resized.copy()
            occluded_image[h:h+patch_size, w:w+patch_size, :] = 0
            occluded_array = np.expand_dims(occluded_image / 255.0, axis=0)
            pred = _model.predict(occluded_array)[0]
            heatmap[h:h+patch_size, w:w+patch_size] = original_pred_class_prob - pred[np.argmax(original_pred)]
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-9)
    return heatmap

# --- FUNZIONE PER CERCARE IMMAGINI ONLINE (CORRETTA) ---
@st.cache_data
def fetch_online_images(query: str, num_images: int = 4):
    """Cerca immagini online usando SerpApi e restituisce gli URL."""
    try:
        api_key = st.secrets["SERPAPI_KEY"]
    except KeyError:
        return "Errore: SERPAPI_KEY non trovata nei secrets di Streamlit."

    params = {
        "engine": "google_images",
        "q": query,
        "api_key": api_key,
    }
    
    try:
        client = SerpApiClient(params) # <-- MODIFICATO: Usiamo la nuova classe
        results = client.get_dict()
        image_results = [item['original'] for item in results.get('images_results', [])[:num_images]]
        return image_results
    except Exception as e:
        return f"Errore durante la chiamata API a SerpApi: {e}"


# --- FUNZIONI PER L'ATTIVITÀ ACCADEMICA ---
# ... (La funzione save_experiment_data rimane invariata) ...
def save_data_to_google_sheet(data):
    """Salva i dati dell'esperimento in un Google Sheet."""
    try:
        # Carica le credenziali e il nome del foglio dai secrets
        creds = st.secrets["gcp_service_account"]
        sheet_name = st.secrets["gcp_sheet_name"]
        
        # Autorizza e apre il foglio di calcolo
        gc = gspread.service_account_from_dict(creds)
        spreadsheet = gc.open(sheet_name)
        worksheet = spreadsheet.sheet1 # Accede al primo foglio
        
        # Converte i dati in un DataFrame di pandas per un facile inserimento
        df = pd.DataFrame([data])
        
        # Trova la prima riga vuota e scrive i dati
        # Nota: gspread_dataframe non ha una funzione append diretta e pulita.
        # Un modo semplice è leggere tutto, aggiungere la riga e riscrivere,
        # ma per un log è più efficiente usare l'API di base.
        worksheet.append_row(df.values.flatten().tolist())
        
        return True, None # Successo
    except Exception as e:
        return False, str(e) # Fallimento


# --- INTERFACCIA UTENTE STREAMLIT (invariata) ---
st.title("🍄 Analisi Funghi con AI Spiegabile (XAI)")

model = load_model()
if model is None: st.stop()

# --- SIDEBAR ---
# ... (la sidebar rimane invariata) ...
st.sidebar.header("Impostazioni")
uploaded_file = st.sidebar.file_uploader("1. Carica un'immagine di un fungo...", type=["jpg", "jpeg", "png"])
is_experiment_mode = st.sidebar.checkbox("Attiva Modalità Esperimento")
student_id = st.sidebar.text_input("ID Studente", "studente_01") if is_experiment_mode else "default"
explanation_mode = st.sidebar.radio("Modalità di Spiegazione", ("Completa (XAI)", "Nessuna (Black Box)")) if is_experiment_mode else "Completa (XAI)"


if uploaded_file is not None:
    # ... (tutta la parte di predizione e visualizzazione risultati rimane invariata) ...
    image = Image.open(uploaded_file)
    preprocessed_array, original_resized_array = preprocess_image(image)
    predicted_species, info, confidence, all_confidences = predict_fungus(model, preprocessed_array)

    if is_experiment_mode and uploaded_file.name == "amanita_test_01.jpg":
        st.warning("⚠️ **ATTENZIONE: MODALITÀ ESPERIMENTO ATTIVA**", icon="🔬")
        predicted_species = "Boletus edulis"
        info = FUNGI_INFO.get(predicted_species, {})
        confidence = 88.42

    st.header("Risultati dell'Analisi AI")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption=f"Immagine Caricata: {uploaded_file.name}", use_column_width=True)
    with col2:
        st.subheader(f"Predizione: **{predicted_species}**")
        st.write(f"Nome Italiano: **{info.get('nome_italiano', 'N/A')}**")
        st.write(f"Confidenza AI: **{confidence:.2f}%**")
        commestibilita = info.get('commestibile', 'Sconosciuta')
        if commestibilita == "Commestibile": st.success(f"**Commestibilità: {commestibilita}** ✅", icon="✅")
        elif commestibilita == "Velenoso": st.warning(f"**Commestibilità: {commestibilita}** ⚠️", icon="⚠️")
        elif commestibilita == "Mortale": st.error(f"**Commestibilità: {commestibilita}** ☠️", icon="☠️")
        else: st.info(f"**Commestibilità: {commestibilita}** ❔", icon="❔")

    with st.expander("Mostra tutte le probabilità di classificazione (Classi di Appartenenza)"):
        conf_dict = {SPECIES_LIST[i]: f"{all_confidences[i]:.2f}%" for i in range(len(SPECIES_LIST))}
        st.json(conf_dict)

    st.divider()

    if explanation_mode == "Completa (XAI)":
        st.header("🤖 Spiegazione della Decisione (XAI)")
        
        # --- NUOVA INTERFACCIA A SCHEDE CON LA RICERCA ONLINE ---
        tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Esempi dal Web", "🔥 Grad-CAM", "🧩 LIME", "⬛ Occlusion Sensitivity"])

        with tab1:
            st.markdown(f"Sto cercando immagini di **{predicted_species}** online per darti un riferimento visivo.")
            with st.spinner("Ricerca immagini in corso..."):
                online_images = fetch_online_images(f"{predicted_species} mushroom")
                if isinstance(online_images, list) and online_images:
                    st.image(online_images, width=150, caption=[f"Esempio #{i+1}" for i in range(len(online_images))])
                    st.info("""
                    **Come leggere questa scheda?** Queste immagini **non provengono dal modello**, ma da una ricerca su Google. 
                    Usale come un riferimento esterno per confrontare l'immagine originale con esempi "canonici" della specie predetta.
                    """, icon="💡")
                elif isinstance(online_images, str):
                    st.error(online_images) # Mostra il messaggio di errore
                else:
                    st.warning("Nessuna immagine trovata online per questa specie.")
        
        with tab2:
            # ... (il codice per Grad-CAM rimane invariato) ...
            with st.spinner("Generazione Grad-CAM..."):
                last_conv_layer = find_last_conv_layer_name(model)
                if last_conv_layer:
                    gradcam_heatmap = make_gradcam_heatmap(model, preprocessed_array, last_conv_layer)
                    gradcam_superimposed = display_superimposed_heatmap(original_resized_array, gradcam_heatmap)
                    st.image(gradcam_superimposed, caption="Heatmap Grad-CAM", use_column_width=True)
                    st.markdown(f"**Cosa significa?** Le aree **rosse** indicano le parti dell'immagine che l'AI ha ritenuto più importanti per classificarlo come *{predicted_species}*.")
                else:
                    st.error("Impossibile generare Grad-CAM: nessun layer convoluzionale trovato.")
        
        with tab3:
            # ... (il codice per LIME rimane invariato) ...
            with st.spinner("Generazione LIME..."):
                lime_img = explain_with_lime(model, preprocessed_array)
                st.image(lime_img, caption="Spiegazione LIME", use_column_width=True)
                st.markdown(f"**Cosa significa?** LIME evidenzia i **gruppi di pixel** che hanno contribuito maggiormente alla previsione *{predicted_species}*.")
        
        with tab4:
            # ... (il codice per Occlusion Sensitivity rimane invariato) ...
            with st.spinner("Generazione Occlusion Sensitivity..."):
                occlusion_map = make_occlusion_sensitivity_map(model, original_resized_array)
                occlusion_superimposed = display_superimposed_heatmap(original_resized_array, occlusion_map, alpha=0.6)
                st.image(occlusion_superimposed, caption="Mappa di Occlusion Sensitivity", use_column_width=True)
                st.markdown("**Cosa significa?** Le aree **rosse** indicano le regioni dell'immagine che, se coperte, causano il maggior calo di fiducia nella predizione.")

    elif explanation_mode == "Nessuna (Black Box)":
        st.info("🤖 Modalità Black Box: nessuna spiegazione fornita.", icon="⬛")

    if is_experiment_mode:
        # ... (la parte di raccolta dati per l'esperimento rimane invariata) ...
        st.divider()
        st.header("🔬 La Tua Valutazione (per l'Esperimento)")
        trust_score = st.slider("Quanta fiducia hai nella previsione dell'AI? (1=Nessuna, 5=Massima)", 1, 5, 3)
        final_decision = st.radio("Qual è la tua decisione finale sulla commestibilità?", ("Commestibile", "Non Commestibile / Velenoso", "Non so decidere"), index=None, horizontal=True)
        if st.button("Salva e Invia la mia Decisione"):
         if final_decision and student_id:
            experiment_data = {
                "ID_Studente": student_id,
                "Nome_File": uploaded_file.name,
                "Specie_AI": predicted_species,
                "Commestibilita_AI": commestibilita,
                "Decisione_Studente": final_decision,
                "Fiducia_Studente": trust_score,
                "Modalita_Spiegazione": explanation_mode
            }
            # CHIAMA LA NUOVA FUNZIONE
            success, error_message = save_data_to_google_sheet(experiment_data)
         if success:
                st.success("Decisione registrata con successo sul Google Sheet! Grazie.")
             else:
                st.error(f"Errore durante il salvataggio su Google Sheets: {error_message}")
        else:
            st.error("Per favore, compila l'ID studente e fai una scelta prima di salvare.")
