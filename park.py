pip install streamlit numpy pandas matplotlib librosa scikit-learn
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

# Custom CSS for Ghibli-style design (soft, whimsical, nature-inspired)
st.markdown("""
<style>
    body {
        background-color: #f0f8ff; /* Light blue, like a sky */
        font-family: 'Arial', sans-serif;
        color: #2e8b57; /* Sea green for text */
    }
    .stButton>button {
        background-color: #98fb98; /* Pale green */
        color: #2e8b57;
        border-radius: 20px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        font-family: 'Dancing Script', cursive; /* Whimsical font */
    }
    .stTextInput, .stSelectbox, .stCheckbox, .stRadio {
        border-radius: 15px;
        border: 2px solid #87ceeb; /* Sky blue border */
    }
    .stProgress > div > div > div {
        background-color: #32cd32; /* Lime green for progress */
    }
    h1, h2, h3 {
        font-family: 'Dancing Script', cursive;
        color: #228b22; /* Forest green */
    }
    .disclaimer {
        background-color: #fffacd; /* Lemon chiffon */
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #daa520; /* Goldenrod */
    }
    .privacy-note {
        font-size: 12px;
        color: #696969; /* Dim gray */
    }
</style>
""", unsafe_allow_html=True)

# Dummy model training (simulate Parkinson's detection)
# In reality, train on datasets like UCI Parkinson's Voice Dataset
def train_dummy_model():
    # Generate synthetic data: features (MFCC-like) + symptoms score
    np.random.seed(42)
    n_samples = 200
    features = np.random.rand(n_samples, 13)  # 13 MFCC features
    symptoms = np.random.randint(0, 5, n_samples)  # Symptom score 0-4
    labels = np.random.choice([0, 1], n_samples)  # 0: No Parkinson's, 1: Parkinson's
    
    X = np.column_stack([features, symptoms])
    X_train, X_test, y_train, y_test = train_test_split(X, y_train, test_size=0.2)
    
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    
    # Save model
    with open('parkinson_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model

# Load or train model
if os.path.exists('parkinson_model.pkl'):
    with open('parkinson_model.pkl', 'rb') as f:
        model = pickle.load(f)
else:
    model = train_dummy_model()

# Function to extract voice features
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, duration=5)  # Load first 5 seconds
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)  # Average MFCCs

# Symptom scoring
symptom_questions = {
    "Tremors": ["None", "Mild", "Moderate", "Severe"],
    "Stiffness": ["None", "Mild", "Moderate", "Severe"],
    "Balance Issues": ["None", "Mild", "Moderate", "Severe"],
    "Speech Changes": ["None", "Mild", "Moderate", "Severe"],
}

def calculate_symptom_score(responses):
    score = sum([symptom_questions[q].index(r) for q, r in responses.items()])
    return score

# App structure
st.title("🌿 Whispering Woods: Parkinson's Voice Detector 🌿")
st.markdown("A gentle tool inspired by Studio Ghibli's magical worlds. Please note: This is for educational purposes only.")

# Progress indicator
progress_bar = st.progress(0)

# Step 1: User Information Form
st.header("Step 1: Your Information")
with st.form("user_info"):
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=0, max_value=120)
    gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
    contact = st.text_input("Contact (Optional)")
    submitted_info = st.form_submit_button("Next")
    if submitted_info:
        st.session_state['user_info'] = {'name': name, 'age': age, 'gender': gender, 'contact': contact}
        progress_bar.progress(25)

# Step 2: Symptoms Questionnaire
if 'user_info' in st.session_state:
    st.header("Step 2: Symptoms Questionnaire")
    responses = {}
    for symptom, options in symptom_questions.items():
        responses[symptom] = st.radio(symptom, options)
    
    if st.button("Submit Symptoms"):
        st.session_state['symptoms'] = responses
        st.session_state['symptom_score'] = calculate_symptom_score(responses)
        progress_bar.progress(50)

# Step 3: Voice Input
if 'symptoms' in st.session_state:
    st.header("Step 3: Voice Sample")
    st.markdown("Upload a short voice sample (e.g., say 'Ahhhh' for 5 seconds) or record one.")
    uploaded_file = st.file_uploader("Upload Audio (WAV/MP3)", type=["wav", "mp3"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file)
        if st.button("Analyze Voice"):
            features = extract_features(uploaded_file)
            st.session_state['voice_features'] = features
            progress_bar.progress(75)

# Step 4: Detection and Results
if 'voice_features' in st.session_state and 'symptom_score' in st.session_state:
    st.header("Step 4: Results")
    # Combine features
    combined_features = np.append(st.session_state['voice_features'], st.session_state['symptom_score'])
    
    # Predict
    prediction = model.predict([combined_features])[0]
    probability = model.predict_proba([combined_features])[0][1]  # Probability of Parkinson's
    
    if prediction == 1:
        result = "Parkinson’s Detected"
        color = "red"
    else:
        result = "No Signs Detected"
        color = "green"
    
    st.markdown(f"<h2 style='color:{color};'>{result}</h2>", unsafe_allow_html=True)
    st.write(f"Confidence: {probability:.2%}")
    
    # Visualization (using matplotlib)
    fig, ax = plt.subplots()
    ax.bar(['No Signs', 'Parkinson’s'], [1-probability, probability], color=['green', 'red'])
    ax.set_ylabel('Probability')
    ax.set_title('Detection Probability')
    st.pyplot(fig)
    
    progress_bar.progress(100)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
    <strong>Disclaimer:</strong> This tool is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read here.
    </div>
    """, unsafe_allow_html=True)

# Privacy Note
st.markdown("""
<div class="privacy-note">
<strong>Privacy:</strong> Your data (name, age, gender, contact, symptoms, and voice sample) is processed locally and not stored or shared. We respect your privacy and comply with data protection standards.
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.write("Built with care, inspired by the magic of Studio Ghibli. 🌸")
