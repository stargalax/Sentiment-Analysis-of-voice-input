import streamlit as st
from streamlit_mic_recorder import mic_recorder
import tempfile
import whisper
from transformers import pipeline
from scipy.io.wavfile import write

# -------------------------------
# Cached models
# -------------------------------
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

whisper_model = load_whisper_model()
sentiment_model = load_sentiment_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎤 Record Audio & Analyze Sentiment")

audio = mic_recorder(start_prompt="Start recording", stop_prompt="Stop recording")

if audio:
    # Save bytes to temp WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmpfile.write(audio["bytes"])
        audio_path = tmpfile.name

    # Play back audio
    st.audio(audio["bytes"], format="audio/wav", sample_rate=audio["sample_rate"])
    st.success("Audio recorded successfully!")

    # 🔹 Transcribe
    result = whisper_model.transcribe(audio_path)
    text = result["text"].strip()
    st.subheader("📝 Transcription")
    st.write(text if text else "(No speech detected)")

    # 🔹 Sentiment
    if text:
        sentiment = sentiment_model(text)[0]
        st.subheader("😀 Sentiment Analysis")
        st.write(f"**{sentiment['label']}** ({round(sentiment['score'], 2)})")
