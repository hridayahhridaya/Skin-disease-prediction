import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("skin_disease_model.h5")

# Class names (same order used during training)
class_names = ['eczema', 'melanoma', 'psoriasis']

st.title("🧴 Skin Disease Detector")
st.write("Upload a skin image to detect disease")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.subheader("Prediction Result:")
    st.success(f"Disease: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}")
  
