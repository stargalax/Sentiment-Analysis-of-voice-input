# 🎙️ Sentiment Analysis of Voice Input (Real-Time Voice NLP)

This project is a progressive build of a **real-time voice-based emotion and sentiment analysis tool**, inspired by a concept presented during the **LTIMindtree Pre-Placement Talk** at RIT Chennai.

It uses **audio recording**, **speech-to-text**, and **NLP emotion classification** to analyze the emotional tone of spoken input — evolving from a basic CLI prototype to an interactive Streamlit app with near real-time analysis.

---
## architecture diagram
<img width="1077" height="543" alt="image" src="https://github.com/user-attachments/assets/651271bb-1a30-488b-a8a7-097e48d9000d" />

##  Motivation

During LTIMindtree's pre-placement session, the idea of emotion-aware voice interaction systems was introduced. This project brings that concept to life by combining:

-  Voice input
-  Whisper (speech-to-text)
-  Emotion/sentiment detection with Transformers
-  Streamlit-based web UI

---

##  Project Evolution

| Phase | File        | Description |
|-------|-------------|-------------|
|  Initial | `initial.py` | Terminal-based prototype — records 5s audio, transcribes, and analyzes sentiment |
|  Phase 1 | `phase1.py` | Adds browser-based audio recording using Streamlit + basic sentiment output |
|  Phase 2 | `phase2.py` | Improves emotion classification using GoEmotions (`bert-base-go-emotion`) |
|  Final | `app.py` | Splits long recordings into chunks for live-like transcription & multi-label emotion detection |

---

##  Features

-  Record audio from microphone (via browser or system)
-  Transcribe speech using OpenAI's Whisper
-  Detect sentiment/emotions using fine-tuned BERT model
-  Web-based UI (Streamlit)
-  Real-time-ish chunk-wise transcription and analysis
-  No external APIs — runs locally using open-source models

---

##  Tech Stack

| Purpose              | Tech / Library                          |
|----------------------|------------------------------------------|
| Audio Input (CLI)    | `sounddevice`                           |
| Audio Input (Web)    | `streamlit_mic_recorder`, `audio_recorder_streamlit` |
| Transcription        | [`openai/whisper`](https://github.com/openai/whisper) |
| Sentiment/Emotion    | `transformers` — `sentiment-analysis`, `bert-base-go-emotion` |
| UI                   | `Streamlit`                             |
| Audio Processing     | `numpy`, `scipy`, `soundfile`           |

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/stargalax/Sentiment-Analysis-of-voice-input.git
cd Sentiment-Analysis-of-voice-input
```
## Acknowledgements
- Idea inspired by LTMindtree Pre-Placement Talk @ RIT Chennai
- Built with:
    * [`openai/whisper`](https://github.com/openai/whisper)
    * [`bhadresh-savani/bert-base-go-emotion`](https://huggingface.co/bhadresh-savani/bert-base-go-emotion)
    * [`Streamlit`](https://streamlit.io/cloud)
## Improvements currently working on
- Enable streamed trnascription
- Add visuals for emotion's trend over time
- Add mulit- user audio

    

