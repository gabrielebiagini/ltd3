import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import streamlit as st

# Caricamento del modello
model = tf.keras.models.load_model('C:/Users/gabri/Downloads/fungi_classifier_model (1).h5')

# Caricamento dell'ordine delle classi
with open('C:/Users/gabri/Downloads/class_labels.txt', 'r') as f:
    species_list = [line.strip() for line in f]

# Dizionario con le informazioni sui funghi
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


def preprocess_image(image):
    # Preprocessa l'immagine
    image = image.convert('RGB')  # Assicurati che l'immagine sia in formato RGB
    img_array = np.array(image, dtype=np.uint8)  # Converti in array di tipo uint8
    img_array = cv2.resize(img_array, (128, 128))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_fungus(image_array):
    # Predire la specie del fungo
    predictions = model.predict(image_array)
    predicted_species_index = np.argmax(predictions)
    predicted_species = species_list[predicted_species_index]
    confidenza = predictions[0][predicted_species_index] * 100
    
    # Recuperare le informazioni aggiuntive
    info = fungi_info.get(predicted_species, {"nome_italiano": "N/A", "commestibile": "N/A"})
    
    return predicted_species, info, predictions[0] * 100

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    # Costruisci un modello che mappa gli input alle attivazioni dell'ultimo strato convoluzionale
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)
    
    # Costruisci un modello che mappa le attivazioni dell'ultimo strato convoluzionale alle predizioni finali
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)
    
    # Calcola il gradiente della classe predetta rispetto all'output dell'ultimo strato convoluzionale
    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
    
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Calcola la media ponderata delle attivazioni dell'ultimo strato convoluzionale
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    
    # Genera la heatmap
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    
    return heatmap

def display_gradcam(image, heatmap, alpha=0.4):
    image = np.array(image.convert('RGB'), dtype=np.uint8)  # Converti in array di tipo uint8
    image = cv2.resize(image, (128, 128))  # Assicurati che la dimensione sia coerente
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0)
    return superimposed_img

def occlusion_sensitivity(image, model, patch_size=16, stride=8):
    image = np.array(image.convert('RGB'), dtype=np.uint8)  # Converti in array di tipo uint8
    image = cv2.resize(image, (128, 128))  # Ridimensiona l'immagine
    original_image = image.copy()
    
    height, width, _ = image.shape
    heatmap = np.zeros((height, width))
    
    for h in range(0, height, stride):
        for w in range(0, width, stride):
            occluded_image = original_image.copy()
            occluded_image[h:h+patch_size, w:w+patch_size, :] = 0  # Occlude parte dell'immagine
            
            occluded_image_array = occluded_image / 255.0
            occluded_image_array = np.expand_dims(occluded_image_array, axis=0)
            
            prediction = model.predict(occluded_image_array)[0]
            heatmap[h:h+patch_size, w:w+patch_size] = prediction[np.argmax(prediction)]
    
    heatmap = cv2.resize(heatmap, (128, 128))
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    return heatmap

st.title("Riconoscimento dei Funghi")

uploaded_file = st.file_uploader("Scegli un'immagine di un fungo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner('Caricamento dell\'immagine...'):
        image = Image.open(uploaded_file)
        st.image(image, caption='Immagine caricata.', use_column_width=True)
        
        # Preprocessa l'immagine
        image_array = preprocess_image(image)
        
        # Esegui la predizione
        predicted_species, info, confidenze = predict_fungus(image_array)
        
        # Mostra i risultati
        st.write(f"**Specie:** {predicted_species}")
        st.write(f"**Nome Italiano:** {info['nome_italiano']}")
        st.write(f"**Commestibilità:** {info['commestibile']}")
        st.write(f"**Confidenza:** {confidenze[np.argmax(confidenze)]:.2f}%")
        
        # Mostra le percentuali di confidenza per tutte le classi
        st.write("**Percentuali di Confidenza per tutte le classi:**")
        conf_dict = {species_list[i]: confidenze[i] for i in range(len(species_list))}
        st.write(conf_dict)
        
        # Genera e mostra Grad-CAM
        last_conv_layer_name = "conv5_block3_out"  # Ultimo strato convoluzionale di ResNet50
        classifier_layer_names = ["global_average_pooling2d_1", "dense_2", "dense_3"]  # Strati finali del modello

        heatmap = make_gradcam_heatmap(image_array, model, last_conv_layer_name, classifier_layer_names)
        superimposed_img = display_gradcam(image, heatmap)
        
        st.image(superimposed_img, caption='Grad-CAM', use_column_width=True)
        
        # Genera e mostra Occlusion Sensitivity
        occlusion_map = occlusion_sensitivity(image, model)
        superimposed_occlusion = display_gradcam(image, occlusion_map)
        
        st.image(superimposed_occlusion, caption='Occlusion Sensitivity', use_column_width=True)
        
        # Aggiungi spiegazione per Grad-CAM e Occlusion Sensitivity
        st.markdown(
            """
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
            """
        )