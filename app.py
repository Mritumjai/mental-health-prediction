import streamlit as st
import pandas as pd
import joblib
import os
from utils.preprocess import preprocess_input

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Mental Health Risk Predictor",
    page_icon="🧠",
    layout="centered"
)

# -------------------------
# Load CSS
# -------------------------
def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

#load_css()

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    return joblib.load("mental_health_logistic_model.pkl")

model = load_model()

# Save training columns (IMPORTANT)
training_columns = model.feature_names_in_

# -------------------------
# Header
# -------------------------
st.title("🧠 Mental Health Risk Assessment")
st.markdown(
    "This tool estimates the likelihood of an individual seeking mental health treatment "
    "based on workplace and personal factors."
)

st.divider()

# -------------------------
# Input Section
# -------------------------
st.subheader("Personal Information")

age = st.number_input("Age", min_value=18, max_value=80, value=30)

gender = st.selectbox("Gender", ["male", "female", "other"])

family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

work_interfere = st.selectbox(
    "Work Interference due to Mental Health",
    ["Never", "Rarely", "Sometimes", "Often"]
)

remote_work = st.selectbox("Remote Work", ["Yes", "No"])

benefits = st.selectbox("Employer Provides Mental Health Benefits", ["Yes", "No"])

leave = st.selectbox(
    "Ease of Taking Mental Health Leave",
    ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"]
)

st.divider()

# -------------------------
# Prediction Button
# -------------------------
if st.button("🔍 Predict Risk"):

    input_data = {
        "Age": age,
        "Gender": gender,
        "family_history": family_history,
        "work_interfere": work_interfere,
        "remote_work": remote_work,
        "benefits": benefits,
        "leave": leave
    }

    processed_input = preprocess_input(input_data, training_columns)

    probability = model.predict_proba(processed_input)[0][1]
    percent = round(probability * 100, 2)

    # Risk category
    if percent < 40:
        risk_label = "Low Risk"
        css_class = "low-risk"
    elif percent < 70:
        risk_label = "Moderate Risk"
        css_class = "medium-risk"
    else:
        risk_label = "High Risk"
        css_class = "high-risk"

    # Display Result
    st.markdown(
        f"""
        <div class="prediction-box">
            <h2>Risk Probability: {percent}%</h2>
            <p class="{css_class}">{risk_label}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "This prediction is based on statistical modeling and should not replace professional advice."
    )
