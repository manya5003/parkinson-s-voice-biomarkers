import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ----------------------------
# Load dataset
# ----------------------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
data = pd.read_csv(url)

# Features and target
X = data.drop(['name','status'], axis=1)
y = data['status']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ----------------------------
# Feature descriptions
# ----------------------------
feature_descriptions = {
    "MDVP:Fo(Hz)": "Average vocal fundamental frequency (Hz)",
    "MDVP:Fhi(Hz)": "Maximum vocal fundamental frequency (Hz)",
    "MDVP:Flo(Hz)": "Minimum vocal fundamental frequency (Hz)",
    "MDVP:Jitter(%)": "Cycle-to-cycle frequency variation (Jitter)",
    "MDVP:Jitter:DDP": "Absolute Jitter in microseconds",
    "MDVP:RAP": "Relative Average Perturbation",
    "MDVP:PPQ": "Pitch Period Perturbation Quotient",
    "Jitter:DDP": "Derivative of absolute jitter values",
    "MDVP:Shimmer": "Cycle-to-cycle amplitude variation (Shimmer)",
    "MDVP:Shimmer(dB)": "Shimmer in decibels",
    "Shimmer:APQ3": "3-point amplitude perturbation quotient",
    "Shimmer:APQ5": "5-point amplitude perturbation quotient",
    "Shimmer:APQ": "Average amplitude perturbation quotient",
    "Shimmer:DDA": "Derivative of amplitude perturbation",
    "NHR": "Noise-to-Harmonics Ratio",
    "HNR": "Harmonics-to-Noise Ratio",
    "RPDE": "Recurrence Period Density Entropy",
    "DFA": "Detrended Fluctuation Analysis",
    "spread1": "Nonlinear measure of fundamental frequency variation",
    "spread2": "Another nonlinear measure of fundamental frequency variation",
    "PPE": "Pitch Period Entropy"
}

# ----------------------------
# Page Title
# ----------------------------
st.markdown("<h1 style='text-align: center; color: darkblue;'>Parkinson's Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Enter your details, symptoms, and voice features to get a prediction</h4>", unsafe_allow_html=True)
st.markdown("---")

# ----------------------------
# Personal Info
# ----------------------------
st.markdown("<h3 style='color: teal;'>Personal Information</h3>", unsafe_allow_html=True)
name = st.text_input("Name")
age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])

# ----------------------------
# Symptoms
# ----------------------------
st.markdown("<h3 style='color: darkgreen;'>Symptoms Check</h3>", unsafe_allow_html=True)
st.write("Check all that apply:")
tremor = st.checkbox("Tremor")
slow_movement = st.checkbox("Slowness of Movement")
speech_changes = st.checkbox("Speech Changes")
balance_issues = st.checkbox("Balance Issues")
fatigue = st.checkbox("Fatigue")

symptom_score = sum([tremor, slow_movement, speech_changes, balance_issues, fatigue])

# ----------------------------
# Voice Feature Inputs (Top 5)
# ----------------------------
st.markdown("<h3 style='color: darkorange;'>Voice Feature Inputs (Top 5 Features)</h3>", unsafe_allow_html=True)
importances = model.feature_importances_
top5_idx = np.argsort(importances)[-5:][::-1]
top5_features = X.columns[top5_idx]

voice_inputs = {}
for feature in top5_features:
    min_val = float(X[feature].min())
    max_val = float(X[feature].max())
    mean_val = float(X[feature].mean())
    description = feature_descriptions.get(feature, "")
    label = f"{feature} ({description})" if description else feature
    voice_inputs[feature] = st.slider(label, min_val, max_val, mean_val, key=feature)

# ----------------------------
# Predict Button
# ----------------------------
st.markdown("---")
predict_button = st.button("Predict", help="Click to get prediction")

if predict_button:
    # Prepare model input
    full_sample = np.zeros(X.shape[1])
    for i, f in enumerate(X.columns):
        if f in top5_features:
            full_sample[i] = voice_inputs[f]
        else:
            full_sample[i] = X[f].mean()
    full_sample = full_sample.reshape(1, -1)

    # Make prediction
    prediction = model.predict(full_sample)[0]

    # Display result with color
    st.markdown("<h3 style='color: purple;'>Prediction Result</h3>", unsafe_allow_html=True)
    if prediction == 1 or symptom_score >= 3:
        st.error(f"⚠️ {name}, Parkinson's disease is likely detected! (Symptom score: {symptom_score})")
    else:
        st.success(f"✅ {name}, you are likely healthy. (Symptom score: {symptom_score})")

# ----------------------------
# Model Accuracy & Feature Importance
# ----------------------------
st.markdown("---")
st.markdown("<h3 style='color: navy;'>Model Accuracy & Feature Importance</h3>", unsafe_allow_html=True)
st.write(f"Model Accuracy: {model.score(X_test, y_test)*100:.2f}%")

fig, ax = plt.subplots(figsize=(10,6))
ax.barh(X.columns, importances, color='skyblue')
ax.set_xlabel("Importance")
ax.set_title("Feature Importance")
st.pyplot(fig)
