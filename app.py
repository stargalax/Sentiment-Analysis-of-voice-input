import streamlit as st
from audio_recorder_streamlit import audio_recorder
import tempfile
import whisper
from transformers import pipeline
import numpy as np
import soundfile as sf
#
st.title("🎤 Continuous Audio Transcription & Emotion Detection")


# Cached models

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="bhadresh-savani/bert-base-go-emotion")

whisper_model = load_whisper_model()
emotion_model = load_emotion_model()


st.write("Click the button below to record your audio. Record as long as you want.")
audio_bytes = audio_recorder()

if audio_bytes:
    # Save bytes to a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmpfile.write(audio_bytes)
        tmpfile_path = tmpfile.name

    # Read audio as NumPy array
    data, samplerate = sf.read(tmpfile_path)

    # Flatten if stereo
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    st.audio(audio_bytes, format="audio/wav")
    st.success("Audio recorded!")


    CHUNK_SIZE = samplerate * 3 
    chunks = [data[i:i + CHUNK_SIZE] for i in range(0, len(data), CHUNK_SIZE)]

    final_transcript = ""
    st.subheader("📝 Live-like Transcription & Emotion Detection (per chunk)")

    for idx, chunk in enumerate(chunks, 1):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            sf.write(tmpfile.name, chunk, samplerate, subtype="PCM_16")
            chunk_path = tmpfile.name

        # Transcribe
        result = whisper_model.transcribe(chunk_path)
        text = result["text"].strip()
        final_transcript += " " + text

        # Detect emotions
        if text:
            emotions = emotion_model(text, top_k=3)
            st.markdown(f"**Chunk {idx}:** {text}")
            for emo in emotions:
                label = emo['label']
                score = round(emo['score'], 2)
                st.write(f"- **{label}** — Confidence: {score}")
        else:
            st.markdown(f"**Chunk {idx}:** (No speech detected)")

    st.subheader("📄 Full Transcript")
    st.write(final_transcript)
