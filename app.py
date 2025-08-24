import streamlit as st
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import json

# Load the trained model
model = tf.keras.models.load_model("model/butterfly_model.keras")

# Load class names from JSON file
with open("model/class_names.json", "r") as f:
    class_names = json.load(f)

st.title("🦋 Butterfly Classifier")
st.write("Upload a butterfly image and we will tell you its type")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(128, 128))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions[0])
    pred_label = class_names[pred_index]
    confidence = predictions[0][pred_index] * 100

    st.success(f"Prediction: **{pred_label}** ({confidence:.2f}%)")
