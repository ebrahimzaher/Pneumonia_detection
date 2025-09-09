import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import joblib

# Load your base model (MobileNetV2) if using it for feature extraction
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

# Load trained SVM model
svm = joblib.load("svm_model.pkl")

class_names = ["Normal", "Pneumonia"]

st.title("Pneumonia Detection from Chest X-ray")
st.write("Upload an X-ray image, and the model will classify it.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    features = base_model.predict(img_array, verbose=0)

    prediction = svm.predict(features)[0]
    st.subheader(f"Prediction: {class_names[prediction]}")
