import streamlit as st
from audio_recorder_streamlit import audio_recorder
import tempfile
import whisper
from transformers import pipeline
from scipy.io.wavfile import write
import numpy as np

st.title("🎤 Audio Transcription & Emotion Detection")

# -------------------------------
# Cached models
# -------------------------------
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_emotion_model():
    # Fine-grained emotion detection (GoEmotions)
    return pipeline("text-classification", model="bhadresh-savani/bert-base-go-emotion")

whisper_model = load_whisper_model()
emotion_model = load_emotion_model()

# -------------------------------
# Record Audio
# -------------------------------
audio_bytes = audio_recorder()

if audio_bytes:
    # Save bytes to temporary WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmpfile.write(audio_bytes)
        audio_path = tmpfile.name

    # Playback
    st.audio(audio_bytes, format="audio/wav")
    st.success("Audio recorded!")

    # -------------------------------
    # Transcribe with Whisper
    # -------------------------------
    result = whisper_model.transcribe(audio_path)
    text = result["text"].strip()

    st.subheader("📝 Transcription")
    st.write(text if text else "(No speech detected)")

    # -------------------------------
    # Emotion Detection
    # -------------------------------
    if text:
        emotions = emotion_model(text, top_k=3)  # show top 3 emotions
        st.subheader("💖 Detected Emotions")
        for emo in emotions:
            label = emo['label']
            score = round(emo['score'], 2)
            st.write(f"**{label}** — Confidence: {score}")
