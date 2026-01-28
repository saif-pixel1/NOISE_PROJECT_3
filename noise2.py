import streamlit as st
import numpy as np
from scipy.io import wavfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os
import tempfile
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Noise Level Analyzer", layout="wide", initial_sidebar_state="expanded")

# Title
st.title("🔊 Noise Level Analyzer")
st.markdown("Analyze audio files and classify noise levels (Low, Medium, High)")

# ============ FEATURE EXTRACTION ============
def extract_features(audio_data, sample_rate):
    """Extract RMS and Zero Crossing Rate features"""
    audio = audio_data.astype(np.float32)
    
    # Convert stereo to mono if needed
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # RMS (Root Mean Square) - indicates loudness
    rms = float(np.sqrt(np.mean(audio ** 2)))
    
    # Zero Crossing Rate - indicates pitch/frequency content
    zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
    
    return [rms, zcr], audio, sample_rate


def calculate_db(rms):
    """Convert RMS to dB level"""
    reference = 1.0
    db = 20 * np.log10(rms / reference) if rms > 0 else -np.inf
    return db


# ============ MODEL TRAINING ============
@st.cache_resource
def load_model():
    """Load and train model on the audio folder files"""
    audio_folder = "audio"
    
    if not os.path.isdir(audio_folder):
        return None, None, "Audio folder not found"
    
    X = []
    y = []
    
    for file in os.listdir(audio_folder):
        file_path = os.path.join(audio_folder, file)
        
        if file.lower().endswith(".wav") and not file.lower().startswith("test"):
            try:
                sr, audio = wavfile.read(file_path)
                features, _, _ = extract_features(audio, sr)
                
                fname = file.lower()
                if "low" in fname:
                    y.append(0)
                    X.append(features)
                elif "med" in fname or "medium" in fname:
                    y.append(1)
                    X.append(features)
                elif "high" in fname:
                    y.append(2)
                    X.append(features)
            except Exception as e:
                st.warning(f"Could not load {file}: {e}")
    
    if not X:
        return None, None, "No training files found"
    
    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, f"Model trained on {len(X)} files"


# ============ CLASSIFICATION ============
def classify_audio(model, scaler, features):
    """Classify audio and get confidence"""
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)[0]
    
    labels = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    label = labels[int(prediction[0])]
    confidence = float(probabilities[int(prediction[0])]) * 100
    
    return label, confidence, probabilities


# ============ SIDEBAR ============
st.sidebar.header("📋 Model Info")
model, scaler, status = load_model()

if model:
    st.sidebar.success(f"✅ {status}")
else:
    st.sidebar.error(f"❌ {status}")

st.sidebar.markdown("---")
st.sidebar.header("🎯 Noise Level Ranges")
st.sidebar.info("""
**LOW**: 0-20 dB (Very quiet)
- Confidence: >70%

**MEDIUM**: 20-50 dB (Normal speech)
- Confidence: >70%

**HIGH**: >50 dB (Loud)
- Confidence: >70%
""")

# ============ MAIN INTERFACE ============
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Audio File")
    uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])
    
    if uploaded_file:
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        
        try:
            # Load audio
            sr, audio = wavfile.read(tmp_path)
            features, audio_mono, _ = extract_features(audio, sr)
            
            # Calculate metrics
            rms = features[0]
            zcr = features[1]
            db_value = calculate_db(rms)
            
            # Classify
            if model and scaler:
                label, confidence, probs = classify_audio(model, scaler, features)
                
                with col2:
                    st.subheader("📊 Analysis Results")
                    
                    # Audio player
                    st.subheader("🎧 Play Audio")
                    st.audio(uploaded_file, format="audio/wav")
                    
                    # Main result with color
                    color_map = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
                    st.markdown(f"### {color_map[label]} Classification: **{label}**")
                    st.metric("Confidence", f"{confidence:.1f}%")
                    
                    # Metrics
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("dB Level", f"{db_value:.2f} dB")
                    with col_m2:
                        st.metric("RMS", f"{rms:.4f}")
                    with col_m3:
                        st.metric("Zero Cross Rate", f"{zcr:.4f}")
                
                # Detailed confidence breakdown
                st.markdown("---")
                st.subheader("📈 Confidence Breakdown")
                
                conf_col1, conf_col2, conf_col3 = st.columns(3)
                with conf_col1:
                    st.metric("🟢 LOW", f"{probs[0]*100:.1f}%", delta="-" if probs[0] != max(probs) else "Winner")
                with conf_col2:
                    st.metric("🟡 MEDIUM", f"{probs[1]*100:.1f}%", delta="-" if probs[1] != max(probs) else "Winner")
                with conf_col3:
                    st.metric("🔴 HIGH", f"{probs[2]*100:.1f}%", delta="-" if probs[2] != max(probs) else "Winner")
                
                # Confidence chart
                fig, ax = plt.subplots(figsize=(10, 4))
                labels_chart = ["LOW", "MEDIUM", "HIGH"]
                colors = ["#00AA00", "#FFAA00", "#FF0000"]
                bars = ax.bar(labels_chart, probs * 100, color=colors, alpha=0.7, edgecolor="black", linewidth=2)
                ax.set_ylabel("Confidence (%)", fontsize=12)
                ax.set_title("Noise Level Classification Confidence", fontsize=14, fontweight="bold")
                ax.set_ylim(0, 100)
                
                # Add percentage labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%', ha='center', va='bottom', fontweight="bold")
                
                st.pyplot(fig, use_container_width=True)
                
                # Waveform visualization
                st.markdown("---")
                st.subheader("🌊 Audio Waveform")
                fig, ax = plt.subplots(figsize=(12, 3))
                time = np.linspace(0, len(audio_mono) / sr, num=len(audio_mono))
                ax.plot(time, audio_mono, linewidth=0.5, color="steelblue")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
                ax.set_title(f"Waveform - {uploaded_file.name}")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig, use_container_width=True)
                
                # Precautions based on level
                st.markdown("---")
                st.subheader("⚠️ Precautions & Recommendations")
                
                if label == "LOW":
                    st.info("""
                    ✅ **Low Noise Environment**
                    - Safe for extended listening
                    - Suitable for concentration and focus work
                    - No special precautions needed
                    """)
                elif label == "MEDIUM":
                    st.warning("""
                    ⚠️ **Medium Noise Level**
                    - Prolonged exposure may cause fatigue
                    - Take regular breaks in quiet areas
                    - Consider using noise-cancelling headphones if needed
                    - Recommended maximum: 8 hours/day
                    """)
                else:  # HIGH
                    st.error("""
                    🔴 **High Noise Level - CAUTION**
                    - Risk of hearing damage with prolonged exposure
                    - Use hearing protection (earplugs/headphones)
                    - Limit exposure to 1-2 hours per day
                    - Take frequent breaks in quiet environments
                    - Consult health guidelines for workplace safety
                    """)
            else:
                st.error("Model not loaded. Cannot classify.")
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
        finally:
            os.unlink(tmp_path)
    else:
        with col2:
            st.info("👈 Upload a .wav file to analyze")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
Noise Level Analyzer v1.0 | Trained on: high1, high2, low1, low2, med1, med2
</div>
""", unsafe_allow_html=True)
