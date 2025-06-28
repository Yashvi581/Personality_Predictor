import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Load the trained model and scaler
model = tf.keras.models.load_model("personality_model.h5")
scaler = joblib.load("scaler.joblib")

st.title("🧠 Personality Predictor (Introvert vs Extrovert)")
st.write("Enter your preferences below:")

# --- User Inputs via Text ---
time_alone = st.text_input("Time Spent Alone (hrs/day)", "3")
stage_fear = st.text_input("Do you have stage fear? (Yes/No)", "Yes")
social_events = st.text_input("Social Events Attended (per month)", "6")
going_out = st.text_input("Days Going Outside (per month)", "7")
drained = st.text_input("Do you feel drained after socializing? (Yes/No)", "Yes")
friends_circle = st.text_input("Friends Circle Size", "14")
post_freq = st.text_input("Social Media Post Frequency (per month)", "10")

# Validate and convert inputs
try:
    time_alone = float(time_alone)
    stage_fear_encoded = 1 if stage_fear.strip().lower() == "yes" else 0
    social_events = float(social_events)
    going_out = float(going_out)
    drained_encoded = 1 if drained.strip().lower() == "yes" else 0
    friends_circle = float(friends_circle)
    post_freq = float(post_freq)

    # Prepare input DataFrame
    new_input = pd.DataFrame([[time_alone, stage_fear_encoded, social_events,
                               going_out, drained_encoded, friends_circle, post_freq]],
                             columns=[
                                 'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
                                 'Going_outside', 'Drained_after_socializing',
                                 'Friends_circle_size', 'Post_frequency'
                             ])

    # Scale and predict
    new_input_scaled = scaler.transform(new_input)
    prediction = model.predict(new_input_scaled)[0][0]
    result = "Introvert 🧍‍♀️" if prediction > 0.5 else "Extrovert 🕺"

    st.subheader("🎯 Predicted Personality:")
    st.success(result)

except ValueError:
    st.warning("⚠️ Please enter valid numeric values for all inputs.")
