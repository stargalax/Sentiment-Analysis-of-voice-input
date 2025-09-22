import sounddevice as sd
import numpy as np
import whisper
import torch
from transformers import pipeline
import tempfile
import os
import time
import threading
import queue
import sys
import wave

# Settings
SAMPLE_RATE = 16000
CHUNK_DURATION = 5  # seconds
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# Load models
print("Loading models...")
whisper_model = whisper.load_model("base")  
sentiment_analyzer = pipeline("sentiment-analysis")
print("Models loaded.")

# Queue to hold audio chunks
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

def record_audio():
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
        while True:
            sd.sleep(1000)

def process_audio_chunks():
    buffer = np.empty((0, 1), dtype=np.float32)
    while True:
        try:
            # Collect enough frames for CHUNK_DURATION
            while buffer.shape[0] < CHUNK_SIZE:
                chunk = audio_queue.get()
                buffer = np.concatenate((buffer, chunk), axis=0)

            current_chunk = buffer[:CHUNK_SIZE]
            buffer = buffer[CHUNK_SIZE:]

            # Save to temp WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wavfile = f.name
                with wave.open(wavfile, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes((current_chunk * 32767).astype(np.int16).tobytes())

            # Transcribe
            print("\n Transcribing...")
            result = whisper_model.transcribe(wavfile)
            text = result['text'].strip()
            os.remove(wavfile)

            if text:
                print(f" Text: {text}")
                sentiment = sentiment_analyzer(text)[0]
                label = sentiment['label']
                score = sentiment['score']
                print(f" Sentiment: {label} ({score:.2f})")
            else:
                print("No speech detected.")

        except Exception as e:
            print(f"Error: {e}")

# Run in threads
rec_thread = threading.Thread(target=record_audio, daemon=True)
proc_thread = threading.Thread(target=process_audio_chunks, daemon=True)

rec_thread.start()
proc_thread.start()

try:
    print("🎙️ Listening... Press Ctrl+C to stop.")
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    print("Stopping...")
    sys.exit(0)
