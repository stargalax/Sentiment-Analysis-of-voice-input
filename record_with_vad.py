import collections
import sys
import sounddevice as sd
import webrtcvad
import numpy as np
import soundfile as sf
import time

# Parameters
FORMAT = 'int16'
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION = 30  # ms
VAD_MODE = 3  # 0-3, more aggressive = more filtering
MAX_SILENCE = 1.5  # seconds of silence before stopping

def record_until_silence():
    vad = webrtcvad.Vad(VAD_MODE)
    frame_size = int(SAMPLE_RATE * FRAME_DURATION / 1000)
    ring_buffer = collections.deque(maxlen=int(MAX_SILENCE * 1000 / FRAME_DURATION))

    print("🎤 Speak now...")

    audio_frames = []
    silence_counter = 0

    def callback(indata, frames, time_info, status):
        nonlocal silence_counter
        if status:
            print("⚠️", status)
        pcm_data = indata.tobytes()
        is_speech = vad.is_speech(pcm_data, SAMPLE_RATE)
        audio_frames.append(pcm_data)

        ring_buffer.append(is_speech)
        if not any(ring_buffer):
            silence_counter += 1
        else:
            silence_counter = 0

        if silence_counter > 10:  # 10 * 30ms = ~300ms
            raise sd.CallbackStop()

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE,
                            channels=CHANNELS,
                            dtype=FORMAT,
                            blocksize=frame_size,
                            callback=callback):
            while True:
                time.sleep(0.1)
    except sd.CallbackStop:
        pass

    print("🛑 Recording stopped.")

    # Convert to numpy
    audio_np = np.frombuffer(b''.join(audio_frames), dtype=np.int16)
    return audio_np.astype(np.float32) / 32768.0
