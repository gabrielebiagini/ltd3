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

# --- CONFIGURAZIONE DELLA PAGINA STREAMLIT ---
st.set_page_config(
    page_title="Analisi Funghi con XAI",
    page_icon="🍄",
    layout="wide"
)

# --- DIZIONARIO DATI E LISTA CLASSI ---
# Caricamento dell'ordine delle classi in modo robusto
try:
    with open('class_labels.txt', 'r') as f:
        SPECIES_LIST = [line.strip() for line in f]
except FileNotFoundError:
    st.error("Errore critico: il file 'class_labels.txt' non è stato trovato. Assicurarsi che sia nella stessa cartella dell'app.")
    st.stop() # Interrompe l'esecuzione se il file delle classi manca

FUNGI_INFO = {
    "Agaricus bisporus": {"nome_italiano": "Prataiolo coltivato", "commestibile": "Commestibile"},
    "Amanita phalloides": {"nome_italiano": "Amanita falloide", "commestibile": "Mortale"},
    "Boletus edulis": {"nome_italiano": "Porcino", "commestibile": "Commestibile"},
    # Aggiungere qui tutte le altre voci del dizionario...
    # Per brevità, ho ridotto il dizionario. Incolli qui il suo dizionario completo.
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


# --- FUNZIONI DI UTILITY E CARICAMENTO MODELLO (CON CACHING) ---

@st.cache_resource
def load_model():
    """Scarica e carica il modello Keras, usando la cache di Streamlit per efficienza."""
    model_url = 'https://www.dropbox.com/scl/fi/437k0jr5hvzzyfyrp50z2/fungi_classifier_model.h5?rlkey=2tar5m1btexq24y6cf2inosnf&dl=1'
    model_path = 'fungi_classifier_model.h5'
    
    if not os.path.isfile(model_path):
        st.write(f"Modello non trovato, scaricando da URL...")
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status()  # Controlla se ci sono errori HTTP
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
    """Preprocessa l'immagine per il modello."""
    image = image.convert('RGB')
    img_array = np.array(image, dtype=np.uint8)
    img_array = cv2.resize(img_array, (128, 128))
    img_array_scaled = img_array / 255.0
    return np.expand_dims(img_array_scaled, axis=0), img_array

def predict_fungus(model, image_array):
    """Esegue la predizione e restituisce i risultati."""
    predictions = model.predict(image_array)[0]
    predicted_index = np.argmax(predictions)
    predicted_species = SPECIES_LIST[predicted_index]
    confidence = predictions[predicted_index] * 100
    info = FUNGI_INFO.get(predicted_species, {"nome_italiano": "N/A", "commestibile": "Sconosciuta"})
    return predicted_species, info, confidence, predictions * 100

# --- FUNZIONI DI EXPLAINABLE AI (XAI) ---

def find_last_conv_layer_name(model):
    """Trova dinamicamente il nome dell'ultimo layer convoluzionale."""
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4 and isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    st.warning("Nessun layer convoluzionale trovato per Grad-CAM.")
    return None

@st.cache_data
def make_gradcam_heatmap(_model, img_array, last_conv_layer_name):
    """Genera la heatmap Grad-CAM."""
    grad_model = tf.keras.models.Model(
        [_model.inputs], [_model.get_layer(last_conv_layer_name).output, _model.output]
    )
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

def display_superimposed_heatmap(original_image, heatmap, alpha=0.5):
    """Sovrappone una heatmap all'immagine originale."""
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap, alpha, original_image, 1 - alpha, 0)
    return superimposed_img

@st.cache_data
def explain_with_lime(_model, preprocessed_image_array):
    """Genera una spiegazione LIME per l'immagine."""
    explainer = lime_image.LimeImageExplainer()
    prediction_fn = lambda x: _model.predict(x)
    
    explanation = explainer.explain_instance(
        preprocessed_image_array[0],
        prediction_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000 # Numero di campioni per la perturbazione
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True, num_features=5, hide_rest=False
    )
    lime_img = mark_boundaries(temp / 2 + 0.5, mask)
    return lime_img

# --- FUNZIONI PER L'ATTIVITÀ ACCADEMICA ---

def save_experiment_data(data):
    """Salva i dati dell'esperimento in un file CSV."""
    file_path = 'experiment_results.csv'
    header = "ID_Studente,Nome_File,Specie_AI,Commestibilita_AI,Decisione_Studente,Fiducia_Studente (1-5),Modalita_Spiegazione\n"
    
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write(header)

    with open(file_path, 'a') as f:
        f.write(f"{data['student_id']},{data['filename']},{data['ai_species']},{data['ai_edibility']},{data['student_decision']},{data['student_trust']},{data['explanation_mode']}\n")

# --- INTERFACCIA UTENTE STREAMLIT ---

st.title("🍄 Analisi Funghi con AI Spiegabile (XAI)")

# Caricamento del modello
model = load_model()
if model is None:
    st.stop() # Interrompe l'app se il modello non può essere caricato

# --- SIDEBAR PER CONTROLLI ---
st.sidebar.header("Impostazioni")
uploaded_file = st.sidebar.file_uploader("1. Carica un'immagine di un fungo...", type=["jpg", "jpeg", "png"])

is_experiment_mode = st.sidebar.checkbox("Attiva Modalità Esperimento")

if is_experiment_mode:
    student_id = st.sidebar.text_input("ID Studente", "studente_01")
    explanation_mode = st.sidebar.radio(
        "Modalità di Spiegazione (per Gruppo)",
        ("Nessuna (Black Box)", "Completa (XAI)")
    )
else:
    explanation_mode = "Completa (XAI)" # Default a XAI se non in modalità esperimento

# --- LOGICA PRINCIPALE ---
if uploaded_file is not None:
    # Caricamento e preparazione immagine
    image = Image.open(uploaded_file)
    preprocessed_array, original_resized_array = preprocess_image(image)

    # Predizione
    predicted_species, info, confidence, all_confidences = predict_fungus(model, preprocessed_array)

    # --- MODALITÀ ESPERIMENTO: INIEZIONE DI ERRORI ---
    if is_experiment_mode and uploaded_file.name == "amanita_test_01.jpg":
        st.warning("⚠️ **ATTENZIONE: MODALITÀ ESPERIMENTO ATTIVA** - L'output potrebbe essere alterato.", icon="🔬")
        predicted_species = "Boletus edulis" # Falsifica la predizione per un caso noto
        info = FUNGI_INFO.get(predicted_species, {})
        confidence = 88.42 # Falsifica la confidenza

    st.header("Risultati dell'Analisi AI")
    
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, caption=f"Immagine Caricata: {uploaded_file.name}", use_column_width=True)
    
    with col2:
        st.subheader(f"Predizione: **{predicted_species}**")
        st.write(f"Nome Italiano: **{info.get('nome_italiano', 'N/A')}**")
        st.write(f"Confidenza AI: **{confidence:.2f}%**")

        commestibilita = info.get('commestibile', 'Sconosciuta')
        if commestibilita == "Commestibile":
            st.success(f"**Commestibilità: {commestibilita}** ✅", icon="✅")
        elif commestibilita == "Velenoso":
            st.warning(f"**Commestibilità: {commestibilita}** ⚠️", icon="⚠️")
        elif commestibilita == "Mortale":
            st.error(f"**Commestibilità: {commestibilita}** ☠️", icon="☠️")
        else:
            st.info(f"**Commestibilità: {commestibilita}** ❔", icon="❔")
    
    st.divider()

    # --- SEZIONE EXPLAINABLE AI (XAI) ---
    if explanation_mode == "Completa (XAI)":
        st.header("🤖 Spiegazione della Decisione (XAI)")

        last_conv_layer = find_last_conv_layer_name(model)
        if last_conv_layer:
            with st.spinner("Generazione Grad-CAM..."):
                gradcam_heatmap = make_gradcam_heatmap(model, preprocessed_array, last_conv_layer)
                gradcam_superimposed = display_superimposed_heatmap(original_resized_array, gradcam_heatmap)
            
            with st.spinner("Generazione LIME..."):
                lime_img = explain_with_lime(model, preprocessed_array)

            xai_col1, xai_col2 = st.columns(2)
            with xai_col1:
                st.subheader("Grad-CAM")
                st.image(gradcam_superimposed, caption="Heatmap Grad-CAM", use_column_width=True)
                st.markdown(f"""
                **Cosa significa?** Le aree **rosse** indicano le parti dell'immagine che l'AI ha ritenuto più importanti per classificare il fungo come *{predicted_species}*.
                Controlla se l'AI sta "guardando" le caratteristiche morfologiche corrette (cappello, gambo, lamelle).
                """)
            
            with xai_col2:
                st.subheader("LIME")
                st.image(lime_img, caption="Spiegazione LIME", use_column_width=True)
                st.markdown(f"""
                **Cosa significa?** LIME evidenzia i **gruppi di pixel (superpixel)** che hanno contribuito maggiormente alla previsione *{predicted_species}*.
                È un altro modo per verificare quali aree specifiche dell'immagine hanno guidato la decisione.
                """)

    elif explanation_mode == "Nessuna (Black Box)":
        st.info("🤖 Modalità Black Box: nessuna spiegazione fornita.", icon="⬛")


    # --- SEZIONE RACCOLTA DATI PER ESPERIMENTO ---
    if is_experiment_mode:
        st.divider()
        st.header("🔬 La Tua Valutazione (per l'Esperimento)")
        
        trust_score = st.slider(
            "Su una scala da 1 (nessuna fiducia) a 5 (massima fiducia), quanta fiducia hai nella previsione dell'AI?",
            1, 5, 3
        )
        
        final_decision = st.radio(
            "Qual è la tua decisione finale sulla commestibilità di questo fungo?",
            ("Commestibile", "Non Commestibile / Velenoso", "Non so decidere"),
            index=None
        )
        
        if st.button("Salva e Invia la mia Decisione"):
            if final_decision is not None and student_id:
                experiment_data = {
                    "student_id": student_id,
                    "filename": uploaded_file.name,
                    "ai_species": predicted_species,
                    "ai_edibility": commestibilita,
                    "student_decision": final_decision,
                    "student_trust": trust_score,
                    "explanation_mode": explanation_mode
                }
                save_experiment_data(experiment_data)
                st.success("Decisione registrata con successo! Grazie.")
            else:
                st.error("Per favore, compila l'ID studente e fai una scelta prima di salvare.")
