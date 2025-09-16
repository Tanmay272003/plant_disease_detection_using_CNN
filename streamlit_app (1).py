import os
import json
import numpy as np
import tensorflow as tf
import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO
import time
import google.generativeai as genai  # Gemini API

# ---------------------------- SET YOUR GEMINI API KEY ---------------------------- #
genai.configure(api_key="AIzaSyBTPq2cmGa_Tc6Xrw6Eg0AZ_K9gCgoeFy0")  # Replace with your actual Gemini API key

# ---------------------------- UTILITY FUNCTIONS ---------------------------- #

# Convert Image to Base64
def image_to_base64(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img.thumbnail((800, 800))
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

# Load and preprocess the image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Predict the plant disease class
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Get disease prevention and treatment using Gemini
def get_prevention_from_gemini(disease_name):
    prompt = f"What are the best prevention and treatment methods for the plant disease called '{disease_name}'?"
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini API error: {e}"

# ---------------------------- STREAMLIT APP SETUP ---------------------------- #

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="centered")

# Modern styling with white glass effect
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        background-image: url("https://images.pexels.com/photos/403571/pexels-photo-403571.jpeg");
        background-size: cover;
        background-attachment: fixed;
    }
    .glass-panel {
        background: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 1.5rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.15);
        max-width: 750px;
        margin: 2rem auto;
    }
    .stButton>button {
        background-color: #8CC63F;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #6b9e2f;
    }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.title("🌿 Plant Disease Detector")
st.write("Upload a plant leaf image to detect disease and get prevention tips using Gemini AI.")

# Load model and class indices
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "trained_model/plant_disease_prediction_model.h5")
class_indices_path = os.path.join(working_dir, "class_indices.json")

model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(class_indices_path))

# Upload section
uploaded_image = st.file_uploader("📷 Upload an image of a plant leaf", type=["jpg", "jpeg", "png"], key="plant_upload")

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image.resize((300, 300)), caption="Uploaded Image", use_container_width=False)

    if st.button('🔍 Classify'):
        prediction = predict_image_class(model, uploaded_image, class_indices)
        st.success(f"✅ Prediction: **{prediction}**")

        with st.spinner("🔎 Getting prevention tips using Gemini..."):
            prevention = get_prevention_from_gemini(prediction)
            st.info(f"🌱 **Prevention Tips:**\n\n{prevention}")

st.markdown('</div>', unsafe_allow_html=True)



