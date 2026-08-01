import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Image Classifier",
    page_icon="🐱",
    layout="centered"
)

# ==========================================
# LOAD MODEL
# ==========================================

import os
import gdown

MODEL_PATH = "best_model.keras"
FILE_ID = "1Jbe6OtZ6fhgqneZY1ihZo999pp-yDvqe"

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI model... (first run only)"):
            gdown.download(
                id=FILE_ID,
                output=MODEL_PATH,
                quiet=False
            )

    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ==========================================
# TITLE
# ==========================================

st.title("🐱 Image Classifier")
st.write("Upload an image and let the AI predict whether it is a Cat or a Dog.")

# ==========================================
# FILE UPLOADER
# ==========================================

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================
# PREDICTION
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width="stretch")

    image = np.array(image)
    image = cv2.resize(image, (256, 256))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    with st.spinner("Predicting..."):
        prediction = model.predict(image, verbose=0)
        confidence = float(prediction[0][0])

    st.divider()

    if confidence < 0.5:
        label = "🐱 CAT"
        probability = (1 - confidence) * 100
    else:
        label = "🐶 DOG"
        probability = confidence * 100

    st.subheader("Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Class", label)

    with col2:
        st.metric("Confidence", f"{probability:.2f}%")

    st.progress(probability / 100)

    st.success("Prediction completed successfully.")