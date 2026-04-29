import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# === CONFIG ===
st.set_page_config(
    page_title="🛢️ Gas Cylinder Classifier",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# === CONSTANTS ===
IMG_SIZE = (224, 224)
BRANDS = ['HP', 'IND', 'UNKNOWN']
SIZES = ['2KG', '5KG', '15KG']
COLORS = ['BLUE', 'RED', 'UNKNOWN']

PRESENCE_MODEL_PATH = r"D:\PROJECT\GAS\prefinal1\presence_final.keras"
ATTRIBUTE_MODEL_PATH = r"D:\PROJECT\GAS\prefinal1\attribute_final.keras"

# === LOAD MODELS ===
@st.cache_resource
def load_models():
    presence_model = tf.keras.models.load_model(PRESENCE_MODEL_PATH)
    attribute_model = tf.keras.models.load_model(ATTRIBUTE_MODEL_PATH)
    return presence_model, attribute_model

presence_model, attribute_model = load_models()

# === FUNCTION: Preprocess ===
def preprocess_image(image: Image.Image):
    img = image.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return tf.expand_dims(img_array, axis=0)

# === FUNCTION: Predict ===
def predict(img: Image.Image):
    tensor = preprocess_image(img)
    presence_pred = presence_model.predict(tensor, verbose=0)[0][0]

    if presence_pred < 0.5:
        return "Absent", None, None, None
    else:
        brand_pred, size_pred, color_pred = attribute_model.predict(tensor, verbose=0)
        brand = BRANDS[np.argmax(brand_pred)]
        size = SIZES[np.argmax(size_pred)]
        color = COLORS[np.argmax(color_pred)]
        return "Cylinder Present", brand, size, color

# === CUSTOM CSS ===
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.3rem;
        font-size: 1rem;
        font-weight: bold;
        border-radius: 20px;
        background-color: #007bff20;
    }
    .presence {
        background-color: #cce5ff;
        color: #004085;
    }
    .absent {
        background-color: #f8d7da;
        color: #721c24;
    }
    .label {
        font-weight: 600;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# === HEADER ===
st.markdown("<h2 style='text-align: center;'>🛢️ Gas Cylinder Classifier</h2><hr>", unsafe_allow_html=True)

# === FILE UPLOAD ===
uploaded_file = st.file_uploader("Upload an image of a cylinder", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.markdown("#### Prediction Result")
        with st.spinner("Analyzing..."):
            presence, brand, size, color = predict(image)

        # RESULT DISPLAY
        st.markdown('<div class="result-box">', unsafe_allow_html=True)

        if presence == "Cylinder Present":
            st.markdown(f'<div class="badge presence">Presence: {presence}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="badge">Brand: {brand}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="badge">Size: {size}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="badge">Color: {color}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="badge absent">Presence: {presence}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Please upload a JPG/PNG image to start classification.")
