import streamlit as st
import pickle
import numpy as np

# Load model and encoder
model = pickle.load(open("skin_disease_model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

st.set_page_config(page_title="Skin Disease Predictor")

st.title("🧴 Skin Disease Prediction System")
st.write("Enter patient details below:")

# -------------------
# User Inputs
# -------------------

age = st.number_input("Age", min_value=1, max_value=100, value=25)
itching = st.slider("Itching Level (0-10)", 0, 10, 5)
redness = st.slider("Redness Level (0-10)", 0, 10, 5)
swelling = st.slider("Swelling Level (0-10)", 0, 10, 5)
duration = st.number_input("Duration (Days)", min_value=1, max_value=60, value=10)

# -------------------
# Prediction
# -------------------

if st.button("Predict Disease"):

    input_data = np.array([[age, itching, redness, swelling, duration]])

    prediction = model.predict(input_data)
    predicted_label = label_encoder.inverse_transform(prediction)

    st.subheader("Prediction Result")
    st.success(f"Predicted Disease: {predicted_label[0]}")
