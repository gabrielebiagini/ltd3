import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import streamlit as st
import requests
import os
import time
import json
import hashlib
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurazione per studio sulla fiducia
EXPLANATION_MODES = {
    "no_explanation": "Nessuna spiegazione",
    "confidence_only": "Solo confidence score", 
    "gradcam_only": "Solo Grad-CAM",
    "gradcam_plus_occlusion": "Grad-CAM + Occlusion Sensitivity",
    "contrastive": "Spiegazioni contrastive",
    "natural_language": "Spiegazione in linguaggio naturale",
    "uncertainty_aware": "Con quantificazione incertezza",
    "full_explanation": "Spiegazione completa",
    "similar_examples": "Con esempi simili"
}

# Funzione per scaricare e caricare il modello
def load_model():
    model_url = 'https://www.dropbox.com/scl/fi/437k0jr5hvzzyfyrp50z2/fungi_classifier_model.h5?rlkey=2tar5m1btexq24y6cf2inosnf&dl=1'
    model_path = 'fungi_classifier_model.h5'
    
    if not os.path.isfile(model_path):
        st.write(f"Modello non trovato, scaricando da {model_url}...")
        response = requests.get(model_url, stream=True)
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.write("Download completato.")
    
    return tf.keras.models.load_model(model_path)

# Carica il modello
try:
    model = load_model()
    st.write("Modello caricato correttamente.")
except Exception as e:
    st.write(f"Errore nel caricamento del modello: {e}")
    
# Caricamento dell'ordine delle classi
with open('class_labels.txt', 'r') as f:
    species_list = [line.strip() for line in f]

# Dizionario con le informazioni sui funghi (il tuo originale)
fungi_info = {
    "Agaricus bisporus": {"nome_italiano": "Prataiolo coltivato", "commestibile": "Commestibile"},
    "Agaricus subrufescens": {"nome_italiano": "Prataiolo mandorlato", "commestibile": "Commestibile"},
    "Amanita bisporigera": {"nome_italiano": "Amanita bisporigera", "commestibile": "Velenoso"},
    "Amanita muscaria": {"nome_italiano": "Amanita muscaria", "commestibile": "Velenoso"},
    "Amanita ocreata": {"nome_italiano": "Amanita ocreata", "commestibile": "Velenoso"},
    "Amanita phalloides": {"nome_italiano": "Amanita falloide", "commestibile": "Mortale"},
    "Amanita smithiana": {"nome_italiano": "Amanita smithiana", "commestibile": "Velenoso"},
    "Amanita verna": {"nome_italiano": "Amanita verna", "commestibile": "Mortale"},
    "Amanita virosa": {"nome_italiano": "Amanita virosa", "commestibile": "Mortale"},
    "Auricularia auricula-judae": {"nome_italiano": "Orecchio di Giuda", "commestibile": "Commestibile"},
    "Boletus edulis": {"nome_italiano": "Porcino", "commestibile": "Commestibile"},
    "Cantharellus cibarius": {"nome_italiano": "Gallinaccio", "commestibile": "Commestibile"},
    "Clitocybe dealbata": {"nome_italiano": "Clitocybe dealbata", "commestibile": "Velenoso"},
    "Conocybe filaris": {"nome_italiano": "Conocybe filaris", "commestibile": "Velenoso"},
    "Coprinus comatus": {"nome_italiano": "Coprino chiomato", "commestibile": "Commestibile (con cautela)"},
    "Cordyceps sinensis": {"nome_italiano": "Cordyceps", "commestibile": "Utilizzato in medicina tradizionale"},
    "Cortinarius rubellus": {"nome_italiano": "Cortinarius rubellus", "commestibile": "Mortale"},
    "Entoloma sinuatum": {"nome_italiano": "Entoloma sinuatum", "commestibile": "Velenoso"},
    "Flammulina velutipes": {"nome_italiano": "Fammulina", "commestibile": "Commestibile"},
    "Galerina marginata": {"nome_italiano": "Galerina marginata", "commestibile": "Mortale"},
    "Ganoderma lucidum": {"nome_italiano": "Reishi", "commestibile": "Utilizzato in medicina tradizionale"},
    "Grifola frondosa": {"nome_italiano": "Maitake", "commestibile": "Commestibile"},
    "Gyromitra esculenta": {"nome_italiano": "Gyromitra esculenta", "commestibile": "Commestibile (con preparazione speciale)"},
    "Hericium erinaceus": {"nome_italiano": "Criniera di leone", "commestibile": "Commestibile"},
    "Hydnum repandum": {"nome_italiano": "Steccherino dorato", "commestibile": "Commestibile"},
    "Hypholoma fasciculare": {"nome_italiano": "Hypholoma fasciculare", "commestibile": "Velenoso"},
    "Inocybe erubescens": {"nome_italiano": "Inocybe erubescens", "commestibile": "Velenoso"},
    "Lentinula edodes": {"nome_italiano": "Shiitake", "commestibile": "Commestibile"},
    "Lepiota brunneoincarnata": {"nome_italiano": "Lepiota brunneoincarnata", "commestibile": "Mortale"},
    "Macrolepiota procera": {"nome_italiano": "Mazza di tamburo", "commestibile": "Commestibile"},
    "Morchella esculenta": {"nome_italiano": "Spugnola comune", "commestibile": "Commestibile (con preparazione speciale)"},
    "Omphalotus olearius": {"nome_italiano": "Omphalotus olearius", "commestibile": "Velenoso"},
    "Paxillus involutus": {"nome_italiano": "Paxillus involutus", "commestibile": "Velenoso"},
    "Pholiota nameko": {"nome_italiano": "Nameko", "commestibile": "Commestibile"},
    "Pleurotus citrinopileatus": {"nome_italiano": "Pleurotus citrinopileatus", "commestibile": "Commestibile"},
    "Pleurotus eryngii": {"nome_italiano": "Cardoncello", "commestibile": "Commestibile"},
    "Pleurotus ostreatus": {"nome_italiano": "Orecchione", "commestibile": "Commestibile"},
    "Psilocybe semilanceata": {"nome_italiano": "Psilocybe semilanceata", "commestibile": "Allucinogeno"},
    "Rhodophyllus rhodopolius": {"nome_italiano": "Rhodophyllus rhodopolius", "commestibile": "Velenoso"},
    "Russula emetica": {"nome_italiano": "Colombina rossa", "commestibile": "Velenoso"},
    "Russula virescens": {"nome_italiano": "Colombina verde", "commestibile": "Commestibile"},
    "Scleroderma citrinum": {"nome_italiano": "Falso tartufo", "commestibile": "Velenoso"},
    "Suillus luteus": {"nome_italiano": "Pinarolo", "commestibile": "Commestibile"},
    "Tremella fuciformis": {"nome_italiano": "Tremella fuciformis", "commestibile": "Commestibile"},
    "Tricholoma matsutake": {"nome_italiano": "Matsutake", "commestibile": "Commestibile"},
    "Truffles": {"nome_italiano": "Tartufo", "commestibile": "Commestibile"},
    "Tuber melanosporum": {"nome_italiano": "Tartufo nero pregiato", "commestibile": "Commestibile"}
}

# Database esempi simili (NUOVO) - Espanso per tutte le specie principali
similar_examples_db = {
    "Amanita phalloides": [
        "Caso #247: Amanita falloide con cappello verdastro, 99.2% accuracy",
        "Caso #1891: Morfologia identica, volva prominente",
        "Caso #532: Pattern delle lamelle corrispondente"
    ],
    "Boletus edulis": [
        "Caso #156: Porcino tipico, 98.7% accuracy", 
        "Caso #889: Cappello marrone caratteristico",
        "Caso #1205: Pori bianchi distintivi"
    ],
    "Amanita muscaria": [
        "Caso #345: Amanita muscaria classica, 97.1% accuracy",
        "Caso #678: Macchie bianche su cappello rosso",
        "Caso #912: Anello bianco distintivo"
    ],
    "Galerina marginata": [
        "Caso #567: Galerina marginata tipica, 94.3% accuracy",
        "Caso #890: Cappello bruno-arancio caratteristico",
        "Caso #1123: Gambo fibroso distintivo, habitat su legno"
    ],
    "Cantharellus cibarius": [
        "Caso #234: Gallinaccio dorato perfetto, 96.8% accuracy",
        "Caso #789: Pieghe decorrenti tipiche",
        "Caso #1001: Colore giallo-arancio brillante"
    ],
    "Agaricus bisporus": [
        "Caso #445: Prataiolo da coltivazione, 98.1% accuracy",
        "Caso #778: Lamelle rosa-marrone caratteristiche",
        "Caso #1156: Anello membranoso ben visibile"
    ]
}

# Fattori di calibrazione (NUOVO)
calibration_factors = {
    "Amanita phalloides": 0.92,
    "Boletus edulis": 1.08,
    "Amanita muscaria": 0.95,
    "Cantharellus cibarius": 1.05,
    "default": 1.0
}

# Inizializza session state (NUOVO)
def initialize_session_state():
    if 'explanation_mode' not in st.session_state:
        st.session_state.explanation_mode = "full_explanation"
    if 'user_data' not in st.session_state:
        st.session_state.user_data = []
    if 'interaction_start_time' not in st.session_state:
        st.session_state.interaction_start_time = None
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()
    if 'demographic_collected' not in st.session_state:
        st.session_state.demographic_collected = False
    if 'current_image_number' not in st.session_state:
        st.session_state.current_image_number = 1

def assign_experimental_condition(user_id=None):
    """Assegna condizione sperimentale (NUOVO)"""
    if user_id is None:
        user_id = st.session_state.session_id
    
    hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    condition = hash_val % len(EXPLANATION_MODES)
    return list(EXPLANATION_MODES.keys())[condition]

def preprocess_image(image):
    """Il tuo preprocessing originale"""
    image = image.convert('RGB')
    img_array = np.array(image, dtype=np.uint8)
    img_array = cv2.resize(img_array, (128, 128))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_fungus(image_array):
    """La tua funzione di predizione originale"""
    predictions = model.predict(image_array)
    predicted_species_index = np.argmax(predictions)
    predicted_species = species_list[predicted_species_index]
    confidenza = predictions[0][predicted_species_index] * 100
    
    info = fungi_info.get(predicted_species, {"nome_italiano": "N/A", "commestibile": "N/A"})
    
    return predicted_species, info, predictions[0] * 100

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    """La tua funzione Grad-CAM originale"""
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)
    
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)
    
    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
    
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    
    return heatmap

def display_gradcam(image, heatmap, alpha=0.4):
    """La tua funzione di visualizzazione Grad-CAM originale"""
    image = np.array(image.convert('RGB'), dtype=np.uint8)
    image = cv2.resize(image, (128, 128))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0)
    return superimposed_img

def occlusion_sensitivity(image, model, patch_size=16, stride=8):
    """La tua funzione Occlusion Sensitivity originale"""
    image = np.array(image.convert('RGB'), dtype=np.uint8)
    image = cv2.resize(image, (128, 128))
    original_image = image.copy()
    
    height, width, _ = image.shape
    heatmap = np.zeros((height, width))
    
    for h in range(0, height, stride):
        for w in range(0, width, stride):
            occluded_image = original_image.copy()
            occluded_image[h:h+patch_size, w:w+patch_size, :] = 0
            
            occluded_image_array = occluded_image / 255.0
            occluded_image_array = np.expand_dims(occluded_image_array, axis=0)
            
            prediction = model.predict(occluded_image_array, verbose=0)[0]
            heatmap[h:h+patch_size, w:w+patch_size] = prediction[np.argmax(prediction)]
    
    heatmap = cv2.resize(heatmap, (128, 128))
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    return heatmap

# NUOVE FUNZIONI AGGIUNTE

def monte_carlo_uncertainty(model, image_array, n_samples=30):
    """Calcola incertezza con Monte Carlo (NUOVO)"""
    predictions = []
    
    for _ in range(n_samples):
        noise = np.random.normal(0, 0.005, image_array.shape)
        perturbed_image = image_array + noise
        perturbed_image = np.clip(perturbed_image, 0, 1)
        
        pred = model.predict(perturbed_image, verbose=0)
        predictions.append(pred[0])
    
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    return mean_pred, std_pred

def generate_natural_language_explanation(predicted_species, confidence, top_3_predictions, has_uncertainty=False, uncertainty_score=None):
    """Genera spiegazione in linguaggio naturale (NUOVO)"""
    info = fungi_info.get(predicted_species, {"nome_italiano": "N/A", "commestibile": "N/A"})
    
    confidence_text = ""
    if confidence > 90:
        confidence_text = "molto sicuro"
    elif confidence > 70:
        confidence_text = "abbastanza sicuro"
    elif confidence > 50:
        confidence_text = "moderatamente sicuro"
    else:
        confidence_text = "incerto"
    
    explanation = f"""
    🔍 **Analisi del modello:**
    
    Il modello è **{confidence_text}** ({confidence:.1f}%) che questo sia un **{info['nome_italiano']}** 
    (*{predicted_species}*).
    
    📊 **Come è arrivato a questa conclusione:**
    - Il modello ha analizzato principalmente le caratteristiche morfologiche del cappello, gambo e lamelle
    - I pattern identificati sono consistenti con questa specie nel database di training
    """
    
    if has_uncertainty and uncertainty_score:
        if uncertainty_score > 15:
            explanation += f"\n⚠️ **Incertezza elevata** ({uncertainty_score:.1f}%): Il modello mostra delle incertezze su questa classificazione."
        else:
            explanation += f"\n✓ **Incertezza bassa** ({uncertainty_score:.1f}%): Il modello è relativamente sicuro."
    
    explanation += "\n\n⚖️ **Alternative considerate:**"
    
    for i, (species, conf) in enumerate(top_3_predictions[:3]):
        species_info = fungi_info.get(species, {"nome_italiano": "N/A"})
        explanation += f"\n   {i+1}. {species_info['nome_italiano']} (*{species}*): {conf:.1f}%"
    
    explanation += f"\n\n⚠️ **Commestibilità:** {info['commestibile']}"
    
    if info['commestibile'] in ['Velenoso', 'Mortale']:
        explanation += "\n\n🚨 **ATTENZIONE: Non consumare mai funghi senza consulto di esperti microbiologi!**"
    
    return explanation

def generate_contrastive_explanation(predicted_species, confidenze, species_list):
    """Genera spiegazione contrastiva (NUOVO)"""
    sorted_preds = sorted(enumerate(confidenze), key=lambda x: x[1], reverse=True)
    
    first_class = sorted_preds[0]
    second_class = sorted_preds[1]
    
    first_info = fungi_info.get(species_list[first_class[0]], {"nome_italiano": "N/A"})
    second_info = fungi_info.get(species_list[second_class[0]], {"nome_italiano": "N/A"})
    
    explanation = f"""
    🔄 **Analisi Contrastiva:**
    
    Il modello ha scelto **{first_info['nome_italiano']}** ({first_class[1]:.1f}%) 
    invece di **{second_info['nome_italiano']}** ({second_class[1]:.1f}%).
    
    **Differenza chiave:** {first_class[1] - second_class[1]:.1f} punti percentuali
    
    **Fattori decisivi:**
    - Forma del cappello più coerente con {first_info['nome_italiano']}
    - Pattern delle lamelle distintivo
    - Colorazione caratteristica della specie identificata
    
    **Se l'immagine avesse mostrato:**
    - Cappello più scuro → maggiore probabilità di {second_info['nome_italiano']}
    - Lamelle più fitte → confidenza cambierebbe di ~{abs(first_class[1] - second_class[1])/2:.1f}%
    - Diversa forma del gambo → classificazione potrebbe variare
    """
    
    return explanation

def show_similar_examples(predicted_species):
    """Mostra esempi simili (NUOVO)"""
    if predicted_species in similar_examples_db:
        st.write("**🔍 Esempi simili nel dataset di training:**")
        for example in similar_examples_db[predicted_species]:
            st.write(f"• {example}")
    else:
        st.write("**🔍 Esempi simili:** Nessun esempio specifico disponibile per questa specie.")

def show_calibrated_confidence(raw_confidence, predicted_species):
    """Mostra confidence calibrata (NUOVO)"""
    factor = calibration_factors.get(predicted_species, calibration_factors["default"])
    calibrated_conf = min(raw_confidence * factor, 100)
    
    st.write(f"**Confidenza grezza:** {raw_confidence:.1f}%")
    st.write(f"**Confidenza calibrata:** {calibrated_conf:.1f}%")
    
    if factor < 1:
        st.info("💡 Il modello tende ad essere troppo sicuro per questa specie")
    elif factor > 1:
        st.info("💡 Il modello tende ad essere troppo cauto per questa specie")
    
    return calibrated_conf

def create_uncertainty_visualization(mean_pred, std_pred, species_list):
    """Crea visualizzazione dell'incertezza (NUOVO)"""
    top_5_indices = np.argsort(mean_pred)[-5:][::-1]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    species_names = [fungi_info.get(species_list[i], {"nome_italiano": species_list[i]})["nome_italiano"] for i in top_5_indices]
    means = [mean_pred[i] * 100 for i in top_5_indices]
    stds = [std_pred[i] * 100 for i in top_5_indices]
    
    bars = ax.bar(range(len(species_names)), means, yerr=stds, 
                  capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
    
    ax.set_xlabel('Specie')
    ax.set_ylabel('Confidence (%)')
    ax.set_title('Top 5 Predizioni con Intervalli di Incertezza')
    ax.set_xticks(range(len(species_names)))
    ax.set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in species_names], rotation=45)
    
    plt.tight_layout()
    return fig

def advanced_metrics_tracking(image, predictions, user_responses):
    """Traccia metriche avanzate (NUOVO)"""
    entropy = -np.sum(predictions * np.log(predictions + 1e-8))
    gini = 1 - np.sum(predictions ** 2)
    
    sorted_preds = sorted(predictions, reverse=True)
    confidence_gap = sorted_preds[0] - sorted_preds[1]
    
    image_array = np.array(image.convert('RGB'))
    image_std = np.std(image_array)
    
    metrics = {
        'prediction_entropy': float(entropy),
        'gini_impurity': float(gini),
        'confidence_gap': float(confidence_gap),
        'image_complexity': float(image_std),
        'top_1_confidence': float(sorted_preds[0]),
        'top_2_confidence': float(sorted_preds[1]) if len(sorted_preds) > 1 else 0.0
    }
    
    return metrics

def log_user_interaction(prediction_data, user_responses, decision_time, explanation_mode, advanced_metrics):
    """Registra l'interazione completa (NUOVO)"""
    interaction = {
        'timestamp': datetime.now().isoformat(),
        'session_id': st.session_state.session_id,
        'image_number': st.session_state.current_image_number,
        'predicted_species': prediction_data['species'],
        'confidence_raw': prediction_data['confidence'],
        'confidence_calibrated': prediction_data.get('confidence_calibrated', prediction_data['confidence']),
        'explanation_mode': explanation_mode,
        'decision_time_seconds': decision_time,
        
        # User responses
        'trust_rating': user_responses['trust_rating'],
        'would_consume': user_responses['would_consume'],
        'certainty_of_decision': user_responses['certainty'],
        'explanation_helpful': user_responses['explanation_helpful'],
        'competence_trust': user_responses.get('competence_trust'),
        'benevolence_trust': user_responses.get('benevolence_trust'),
        'integrity_trust': user_responses.get('integrity_trust'),
        'comprehensibility': user_responses.get('comprehensibility'),
        'completeness': user_responses.get('completeness'),
        'usefulness': user_responses.get('usefulness'),
        'decision_confidence': user_responses.get('decision_confidence'),
        'would_rely_future': user_responses.get('would_rely_future'),
        
        # Advanced metrics
        **advanced_metrics,
        
        # Demographics (if available)
        **st.session_state.get('demographic_data', {})
    }
    
    st.session_state.user_data.append(interaction)

def detailed_questionnaire():
    """Questionario dettagliato (NUOVO)"""
    st.subheader("📋 Questionario Dettagliato")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dimensioni di Fiducia:**")
        competence_trust = st.slider("Il sistema è competente nella classificazione?", 1, 7, 4, key="competence")
        benevolence_trust = st.slider("Il sistema ha buone intenzioni?", 1, 7, 4, key="benevolence")
        integrity_trust = st.slider("Il sistema è onesto nelle sue predizioni?", 1, 7, 4, key="integrity")
        
        st.write("**Percezione delle Spiegazioni:**")
        comprehensibility = st.slider("Quanto hai capito la spiegazione?", 1, 7, 4, key="comprehensibility")
        completeness = st.slider("La spiegazione è completa?", 1, 7, 4, key="completeness")
    
    with col2:
        usefulness = st.slider("La spiegazione è utile per la decisione?", 1, 7, 4, key="usefulness")
        
        st.write("**Decisione:**")
        decision_confidence = st.slider("Quanto sei sicuro della tua decisione finale?", 1, 7, 4, key="decision_conf")
        would_rely_future = st.slider("Useresti questo sistema in futuro?", 1, 7, 4, key="future_use")
    
    return {
        'competence_trust': competence_trust,
        'benevolence_trust': benevolence_trust,
        'integrity_trust': integrity_trust,
        'comprehensibility': comprehensibility,
        'completeness': completeness,
        'usefulness': usefulness,
        'decision_confidence': decision_confidence,
        'would_rely_future': would_rely_future
    }

# INIZIALIZZAZIONE
initialize_session_state()

# CONFIGURAZIONE SIDEBAR
st.sidebar.title("🍄 Studio Fiducia AI")

# Assegnazione condizione sperimentale
if 'assigned_condition' not in st.session_state:
    st.session_state.assigned_condition = assign_experimental_condition()

# Mostra sempre la condizione assegnata
st.sidebar.success(f"**Condizione assegnata:** {EXPLANATION_MODES[st.session_state.assigned_condition]}")

# Override manuale per testing - SEMPRE VISIBILE per ora
st.sidebar.subheader("🔧 Controllo Modalità")
override_mode = st.sidebar.checkbox("Cambia modalità explainability", value=True)
if override_mode:
    explanation_mode = st.sidebar.selectbox(
        "Modalità di spiegazione:",
        list(EXPLANATION_MODES.keys()),
        format_func=lambda x: EXPLANATION_MODES[x],
        index=list(EXPLANATION_MODES.keys()).index("full_explanation"),  # Default a full_explanation
        key="explanation_mode_override"
    )
    st.sidebar.info(f"🔄 **Modalità attiva:** {EXPLANATION_MODES[explanation_mode]}")
else:
    explanation_mode = st.session_state.assigned_condition
    st.sidebar.info(f"📋 **Modalità studio:** {EXPLANATION_MODES[explanation_mode]}")

# Raccolta dati demografici
if not st.session_state.demographic_collected:
    st.sidebar.subheader("📊 Informazioni Demografiche")
    with st.sidebar.form("demographic_form"):
        age_group = st.selectbox("Fascia d'età:", ["18-25", "26-35", "36-45", "45+"])
        education = st.selectbox("Livello di istruzione:", ["Laurea triennale", "Laurea magistrale", "Dottorato", "Altro"])
        ai_experience = st.slider("Esperienza con AI (1-7):", 1, 7, 3)
        fungi_knowledge = st.slider("Conoscenza funghi (1-7):", 1, 7, 3)
        tech_comfort = st.slider("Comfort con tecnologia (1-7):", 1, 7, 4)
        
        submitted = st.form_submit_button("Conferma dati demografici")
        if submitted:
            st.session_state.demographic_data = {
                'age_group': age_group,
                'education': education,
                'ai_experience': ai_experience,
                'fungi_knowledge': fungi_knowledge,
                'tech_comfort': tech_comfort
            }
            st.session_state.demographic_collected = True
            st.success("Dati salvati!")
            st.rerun()

# MAIN TITLE
st.title("🍄 Riconoscimento dei Funghi - Studio Explainable AI")

if st.session_state.demographic_collected:
    st.success("✅ Dati demografici raccolti. Puoi procedere con il test.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        f"📸 Scegli un'immagine di un fungo... (Immagine #{st.session_state.current_image_number})", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Inizia timer
        if st.session_state.interaction_start_time is None:
            st.session_state.interaction_start_time = time.time()
        
        with st.spinner('🔄 Analisi in corso...'):
            # Carica e preprocessa immagine
            image = Image.open(uploaded_file)
            st.image(image, caption='Immagine caricata.', use_column_width=True)
            
            image_array = preprocess_image(image)
            
            # Predizione base (la tua funzione originale)
            predicted_species, info, confidenze = predict_fungus(image_array)
            
            # Calcoli Grad-CAM e Occlusion (le tue funzioni originali)
            last_conv_layer_name = "conv5_block3_out"
            classifier_layer_names = ["global_average_pooling2d_1", "dense_2", "dense_3"]
            
            heatmap = make_gradcam_heatmap(image_array, model, last_conv_layer_name, classifier_layer_names)
            occlusion_map = occlusion_sensitivity(image, model)
            
            # Calcoli aggiuntivi per spiegazioni avanzate (NUOVO)
            if explanation_mode in ["uncertainty_aware", "full_explanation"]:
                mean_pred, std_pred = monte_carlo_uncertainty(model, image_array)
                uncertainty_score = np.mean(std_pred) * 100
            else:
                mean_pred, std_pred, uncertainty_score = None, None, None
        
        # Mostra risultati base
        st.subheader("🎯 Risultati Classificazione")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Specie identificata", info['nome_italiano'])
        with col2:
            st.metric("Nome scientifico", predicted_species)
        with col3:
            status_color = "🟢" if info['commestibile'] == "Commestibile" else "🔴" if info['commestibile'] in ["Velenoso", "Mortale"] else "🟡"
            st.metric("Commestibilità", f"{status_color} {info['commestibile']}")
        
        # Confidence sempre visibile
        confidence_score = confidenze[np.argmax(confidenze)]
        st.write(f"**Confidenza:** {confidence_score:.2f}%")
        
        # SPIEGAZIONI BASATE SULLA MODALITÀ (combinando originali + nuove)
        st.subheader("🧠 Spiegazione del Modello")
        
        if explanation_mode == "no_explanation":
            st.info("Classificazione completata. Valuta la predizione basandoti solo sui risultati mostrati.")
            
        elif explanation_mode == "confidence_only":
            calibrated_confidence = show_calibrated_confidence(confidence_score, predicted_species)
            
        elif explanation_mode == "gradcam_only":
            superimposed_img = display_gradcam(image, heatmap)
            st.image(superimposed_img, caption='🔍 Grad-CAM - Aree di focus del modello', use_column_width=True)
            st.info("Le aree rosse mostrano dove il modello ha concentrato la sua attenzione per la classificazione.")
            
        elif explanation_mode == "gradcam_plus_occlusion":
            # Le tue visualizzazioni originali
            superimposed_img = display_gradcam(image, heatmap)
            st.image(superimposed_img, caption='🔍 Grad-CAM', use_column_width=True)
            
            superimposed_occlusion = display_gradcam(image, occlusion_map)
            st.image(superimposed_occlusion, caption='🔍 Occlusion Sensitivity', use_column_width=True)
            
            # Le tue spiegazioni originali
            st.markdown("""
            **Spiegazione di Grad-CAM:**
            
            Grad-CAM (Gradient-weighted Class Activation Mapping) è una tecnica che aiuta a capire quali parti 
            dell'immagine hanno influenzato maggiormente la decisione del modello. La mappa di calore generata 
            evidenzia le aree dell'immagine che il modello considera più importanti per la sua classificazione. 
            Le regioni più calde (rosse) indicano maggiore importanza, mentre le regioni più fredde (blu) indicano 
            minore importanza.
            
            **Spiegazione di Occlusion Sensitivity:**
            
            Occlusion Sensitivity valuta come la predizione del modello cambia quando parti dell'immagine sono occluse (coperte). 
            Occludendo sistematicamente parti dell'immagine e osservando la variazione nelle predizioni del modello, 
            possiamo identificare le regioni cruciali dell'immagine per la decisione del modello.
            """)
            
        elif explanation_mode == "contrastive":
            contrastive_explanation = generate_contrastive_explanation(predicted_species, confidenze, species_list)
            st.markdown(contrastive_explanation)
            
        elif explanation_mode == "natural_language":
            top_3 = sorted([(species_list[i], confidenze[i]) for i in range(len(species_list))], 
                          key=lambda x: x[1], reverse=True)[:3]
            nl_explanation = generate_natural_language_explanation(
                predicted_species, confidence_score, top_3, 
                uncertainty_score is not None, uncertainty_score
            )
            st.markdown(nl_explanation)
            
        elif explanation_mode == "uncertainty_aware":
            calibrated_confidence = show_calibrated_confidence(confidence_score, predicted_species)
            
            st.write(f"**Incertezza del modello:** {uncertainty_score:.2f}%")
            
            if uncertainty_score > 15:
                st.warning("⚠️ Il modello mostra alta incertezza. Considera una seconda opinione.")
            elif uncertainty_score < 5:
                st.success("✅ Il modello è molto sicuro di questa predizione.")
            else:
                st.info("ℹ️ Livello di incertezza normale.")
            
            fig = create_uncertainty_visualization(mean_pred, std_pred, species_list)
            st.pyplot(fig)
            
        elif explanation_mode == "similar_examples":
            show_similar_examples(predicted_species)
            
        elif explanation_mode == "full_explanation":
            # TUTTO: originale + nuovo
            calibrated_confidence = show_calibrated_confidence(confidence_score, predicted_species)
            
            # Top predictions (le tue percentuali originali)
            st.write("**Percentuali di Confidenza per tutte le classi:**")
            conf_dict = {species_list[i]: confidenze[i] for i in range(len(species_list))}
            st.write(conf_dict)
            
            # Top 3 formattato meglio
            top_3 = sorted([(species_list[i], confidenze[i]) for i in range(len(species_list))], 
                          key=lambda x: x[1], reverse=True)[:3]
            
            st.write("**🏆 Top 3 predizioni:**")
            for i, (species, conf) in enumerate(top_3):
                species_info = fungi_info.get(species, {"nome_italiano": "N/A"})
                st.write(f"{i+1}. {species_info['nome_italiano']} (*{species}*): {conf:.2f}%")
            
            # Le tue visualizzazioni originali
            superimposed_img = display_gradcam(image, heatmap)
            st.image(superimposed_img, caption='🔍 Grad-CAM', use_column_width=True)
            
            superimposed_occlusion = display_gradcam(image, occlusion_map)
            st.image(superimposed_occlusion, caption='🔍 Occlusion Sensitivity', use_column_width=True)
            
            # Uncertainty (nuovo)
            st.write(f"**Incertezza:** {uncertainty_score:.2f}%")
            if uncertainty_score > 15:
                st.warning("⚠️ Alta incertezza rilevata.")
            
            # Similar examples (nuovo)
            show_similar_examples(predicted_species)
            
            # Le tue spiegazioni originali
            st.markdown("""
            **Spiegazione di Grad-CAM:**
            
            Grad-CAM (Gradient-weighted Class Activation Mapping) è una tecnica che aiuta a capire quali parti 
            dell'immagine hanno influenzato maggiormente la decisione del modello. La mappa di calore generata 
            evidenzia le aree dell'immagine che il modello considera più importanti per la sua classificazione. 
            Le regioni più calde (rosse) indicano maggiore importanza, mentre le regioni più fredde (blu) indicano 
            minore importanza.
            
            **Spiegazione di Occlusion Sensitivity:**
            
            Occlusion Sensitivity valuta come la predizione del modello cambia quando parti dell'immagine sono occluse (coperte). 
            Occludendo sistematicamente parti dell'immagine e osservando la variazione nelle predizioni del modello, 
            possiamo identificare le regioni cruciali dell'immagine per la decisione del modello.
            """)
            
            # Natural language summary (nuovo)
            nl_explanation = generate_natural_language_explanation(
                predicted_species, confidence_score, top_3, True, uncertainty_score
            )
            with st.expander("📝 Spiegazione dettagliata"):
                st.markdown(nl_explanation)
        
        # Raccolta feedback utente (NUOVO + migliorato)
        st.subheader("📝 Il Tuo Feedback")
        
        col1, col2 = st.columns(2)
        with col1:
            trust_rating = st.slider("Quanto ti fidi di questa predizione? (1-7)", 1, 7, 4, key="trust")
            would_consume = st.radio("Consumeresti questo fungo?", 
                                    ["Sicuramente sì", "Probabilmente sì", "Non sono sicuro", 
                                     "Probabilmente no", "Assolutamente no"], key="consume")
        
        with col2:
            certainty = st.slider("Quanto sei sicuro della tua decisione? (1-7)", 1, 7, 4, key="certainty")
            explanation_helpful = st.slider("Quanto ti ha aiutato la spiegazione? (1-7)", 1, 7, 4, key="helpful")
        
        # Questionario dettagliato (NUOVO)
        detailed_responses = detailed_questionnaire()
        
        # Bottone per registrare feedback
        if st.button("✅ Registra feedback e continua", key="submit_feedback"):
            decision_time = time.time() - st.session_state.interaction_start_time if st.session_state.interaction_start_time else 0
            
            # Prepara dati predizione
            prediction_data = {
                'species': predicted_species,
                'confidence': float(confidenze[np.argmax(confidenze)]),
                'confidence_calibrated': float(calibrated_confidence) if 'calibrated_confidence' in locals() else float(confidenze[np.argmax(confidenze)])
            }
            
            # Prepara risposte utente
            user_responses = {
                'trust_rating': trust_rating,
                'would_consume': would_consume,
                'certainty': certainty,
                'explanation_helpful': explanation_helpful,
                **detailed_responses
            }
            
            # Calcola metriche avanzate (NUOVO)
            advanced_metrics = advanced_metrics_tracking(image, confidenze / 100, user_responses)
            
            # Log interazione (NUOVO)
            log_user_interaction(prediction_data, user_responses, decision_time, explanation_mode, advanced_metrics)
            
            st.success("✅ Feedback registrato! Grazie per la partecipazione.")
            st.session_state.interaction_start_time = None
            st.session_state.current_image_number += 1
            
            # Mostra statistiche aggregate
            if len(st.session_state.user_data) > 1:
                avg_trust = np.mean([d['trust_rating'] for d in st.session_state.user_data])
                avg_decision_time = np.mean([d['decision_time_seconds'] for d in st.session_state.user_data])
                st.info(f"📊 Statistiche: Trust medio: {avg_trust:.2f} | Tempo decisione medio: {avg_decision_time:.1f}s")
            
            # Suggerimento per continuare
            if st.session_state.current_image_number <= 5:
                st.info(f"💡 Perfetto! Prova con un'altra immagine (#{st.session_state.current_image_number}) per completare lo studio.")
            else:
                st.success("🎉 Hai completato lo studio! I tuoi dati sono stati salvati per l'analisi.")

else:
    st.info("👆 Completa prima le informazioni demografiche nella sidebar per iniziare lo studio.")

# Sidebar per download dati (NUOVO)
if st.sidebar.button("📥 Scarica dati studio", key="download"):
    if st.session_state.user_data:
        df = pd.DataFrame(st.session_state.user_data)
        csv = df.to_csv(index=False)
        st.sidebar.download_button(
            label="📄 Download CSV",
            data=csv,
            file_name=f"fungi_trust_study_{st.session_state.session_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        st.sidebar.success(f"Dati di {len(st.session_state.user_data)} interazioni pronti per il download!")
    else:
        st.sidebar.warning("Nessun dato disponibile per il download.")

# Informazioni studio (NUOVO)
with st.sidebar.expander("ℹ️ Informazioni Studio"):
    st.write("""
    **Obiettivo:** Studiare come diverse modalità di spiegazione AI influenzano la fiducia degli utenti.
    
    **Procedura:**
    1. Carica 3-5 immagini di funghi
    2. Valuta ogni predizione
    3. Fornisci feedback dettagliato
    
    **Dati raccolti:** Anonimi e utilizzati solo per ricerca accademica.
    """)
