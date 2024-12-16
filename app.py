import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image, ImageDraw, ImageFont
import cv2
import base64

# Konfigurasi Halaman
st.set_page_config(page_title="DIMASTI CNN", page_icon="🩺", layout="wide")

# Konfigurasi Model
MODEL_PATH = 'best_pneumonia_model_initial_labkom_VGG16.h5'  # Ganti dengan nama file model VGG16 Anda
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found.")
    st.stop()

model = load_model(MODEL_PATH)
st.success("Model loaded successfully.")

# Konfigurasi Ukuran Gambar
IMG_HEIGHT = 224
IMG_WIDTH = 224
LAST_CONV_LAYER_NAME = 'block5_conv3'

# Fungsi untuk membuat Grad-CAM heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Fungsi untuk prediksi dan generate heatmap
def predict_and_generate_heatmap(img, model, last_conv_layer_name, preprocess_input, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    img_resized = img.resize(img_size)
    img_array = image.img_to_array(img_resized)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_array_preprocessed = preprocess_input(img_array_expanded)

    preds = model.predict(img_array_preprocessed)
    prob = preds[0][0]
    label = 'Pneumonia' if prob > 0.5 else 'Normal'
    prob_percent = prob * 100 if prob > 0.5 else (1 - prob) * 100

    heatmap = make_gradcam_heatmap(img_array_preprocessed, model, last_conv_layer_name)

    original_width, original_height = img.size
    heatmap = cv2.resize(heatmap, (original_width, original_height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original_array = np.array(img)
    superimposed_img = heatmap * 0.4 + original_array
    superimposed_img = Image.fromarray(np.uint8(superimposed_img))

    draw = ImageDraw.Draw(superimposed_img)
    font = ImageFont.load_default()
    text = f"Prediction: {label}, Probability: {prob_percent:.2f}%"
    bbox = font.getbbox(text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    draw.rectangle([(0, 0), (text_width + 10, text_height + 10)], fill='black')
    draw.text((5, 5), text, fill='white', font=font)

    return superimposed_img, label, prob_percent

# Fungsi untuk memuat CSS dan JS dari file terpisah
def load_css():
    css_path = 'static/css/style.css'  # Sesuaikan dengan path
    with open(css_path, 'r') as file:
        css_code = file.read()
        st.markdown(f"<style>{css_code}</style>", unsafe_allow_html=True)

def load_js():
    js_path = 'static/js/script.js'  # Sesuaikan dengan path
    with open(js_path, 'r') as file:
        js_code = file.read()
        st.markdown(f"<script>{js_code}</script>", unsafe_allow_html=True)

# Memuat CSS dan JS
load_css()
load_js()

# Halaman Utama
st.title("DIMASTI CNN: Deteksi Pneumonia")
st.markdown("### Silakan unggah gambar X-Ray Anda untuk prediksi.")

# Widget upload dan prediksi
uploaded_file = st.file_uploader("Upload gambar (X-Ray):", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Menampilkan gambar asli
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar X-Ray", use_column_width=True)

    # Melakukan prediksi dan generate heatmap
    result_img, label, prob_percent = predict_and_generate_heatmap(img, model, LAST_CONV_LAYER_NAME, preprocess_input)

    # Menampilkan hasil prediksi
    st.subheader(f"**Prediksi: {label}**")
    st.write(f"**Probabilitas: {prob_percent:.2f}%**")

    # Menampilkan gambar dengan heatmap
    st.image(result_img, caption="Gambar dengan Heatmap Grad-CAM", use_column_width=True)

# Footer
footer_html = """
    <div style="position: fixed; bottom: 0; width: 100%; text-align: center; color: gray;">
        <p>&copy; 2024 DIMASTI CNN. All rights reserved.</p>
    </div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
