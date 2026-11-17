# 🛡️ VEYRA AI — Personalized Voice Assistant

> A secure, AI-powered voice assistant that authenticates users by their voice before processing commands. Built with Python (Flask), Whisper, Resemblyzer, and a premium browser-based UI.

---

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [System Architecture](#system-architecture)
4. [Technology Stack](#technology-stack)
5. [Project Structure](#project-structure)
6. [Installation & Setup](#installation--setup)
7. [Usage Guide](#usage-guide)
8. [Module Details](#module-details)
9. [Algorithms & Methodology](#algorithms--methodology)
10. [Security Considerations](#security-considerations)
11. [Future Enhancements](#future-enhancements)
12. [References](#references)

---

## Introduction

VEYRA AI is a personalized voice assistant designed to respond **only** to an authorized user while ignoring other voices and background noise. The system combines five core AI technologies:

- **Speaker Authentication** — verifies identity via d-vector voice embeddings
- **Noise Suppression** — cleans audio using spectral gating
- **Speech Recognition** — transcribes speech using OpenAI Whisper
- **Natural Language Processing** — understands intent via rule-based pattern matching
- **Text-to-Speech** — generates spoken responses using pyttsx3

The project demonstrates how modern speech AI technologies can be integrated to create a secure, private, and reliable voice interaction experience — particularly useful in noisy or multi-person environments.

---

## Features

| Feature | Description |
|---------|-------------|
| 🔐 **Voice Enrollment** | Record 5 voice samples to create a unique voice profile |
| 🔍 **Speaker Verification** | Real-time identity check via cosine similarity on d-vector embeddings |
| 🔇 **Noise Suppression** | Spectral gating removes background noise before processing |
| 📝 **Speech-to-Text** | OpenAI Whisper base model for accurate transcription |
| 🧠 **Intent Recognition** | 13 supported intents (time, date, math, jokes, greetings, etc.) |
| 🔊 **Text-to-Speech** | Offline speech synthesis for responses |
| 🎨 **Premium Web UI** | Dark glassmorphism design with real-time pipeline visualization |
| 📋 **Activity Logging** | Timestamped log of all interactions and pipeline results |
| 🚫 **Access Denial** | Visual feedback when an unauthorized voice is detected |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser (Frontend)                        │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Record   │→ │ WAV       │→ │ API Call │→ │ Display   │  │
│  │ (PCM)    │  │ Encode    │  │ (fetch)  │  │ Results   │  │
│  └──────────┘  └───────────┘  └──────────┘  └───────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP POST (WAV)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Flask Server (Backend)                       │
│                                                              │
│  ❶ Noise Suppression (noisereduce)                          │
│      └→ Spectral gating removes background noise             │
│                         ▼                                    │
│  ❷ Speaker Verification (Resemblyzer)                       │
│      └→ D-vector embedding → Cosine similarity ≥ 0.75       │
│      └→ If FAIL → return "Access Denied"                    │
│                         ▼                                    │
│  ❸ Speech-to-Text (OpenAI Whisper)                          │
│      └→ Transcribe cleaned audio to text                     │
│                         ▼                                    │
│  ❹ NLP Intent Recognition (Rule-based)                      │
│      └→ Regex pattern matching → Intent + Response           │
│                         ▼                                    │
│  ❺ Text-to-Speech (pyttsx3)                                │
│      └→ Synthesize response → WAV file                       │
│                                                              │
│  Return JSON + audio URL to frontend                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| Frontend | HTML5 / CSS3 / JavaScript | — | Browser-based dashboard with mic recording |
| Backend | Flask (Python) | 3.x | REST API server |
| Speaker Auth | Resemblyzer | 0.1.3+ | D-vector speaker embeddings (256-dim) |
| Noise Reduction | noisereduce | 3.x | Spectral gating noise suppression |
| Speech-to-Text | OpenAI Whisper | latest | Transformer-based ASR (base model) |
| NLP | Custom rule engine | — | Regex-based intent classification |
| Text-to-Speech | pyttsx3 | 2.90+ | Offline TTS via system SAPI5 / espeak |
| Audio I/O | soundfile, scipy, librosa | — | WAV read/write, audio processing |

---

## Project Structure

```
veyra-ai/
├── app.py                      # Flask server — main entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This documentation
│
├── modules/                    # Backend AI modules
│   ├── __init__.py
│   ├── speaker_auth.py         # Enrollment & verification (Resemblyzer)
│   ├── noise_suppression.py    # Spectral gating denoiser
│   ├── stt_engine.py           # Whisper speech-to-text
│   ├── nlp_engine.py           # Intent recognition & responses
│   └── tts_engine.py           # pyttsx3 text-to-speech
│
├── templates/
│   └── index.html              # Main dashboard page
│
├── static/
│   ├── css/
│   │   └── style.css           # Premium dark-mode design system
│   ├── js/
│   │   └── app.js              # Frontend logic (recording, API, UI)
│   └── audio/
│       └── tts/                # Generated TTS audio files
│
└── voice_profiles/             # Stored voice embeddings (created at runtime)
    ├── authorized_user.npy     # 256-dim d-vector profile
    └── profile_meta.json       # Enrollment metadata
```

---

## Installation & Setup

### Prerequisites

- **Python 3.9+** (3.10 or 3.11 recommended)
- **pip** (Python package manager)
- **Microphone** (for voice recording)
- **Modern browser** (Chrome, Edge, or Firefox)

### Step-by-step

```bash
# 1. Navigate to the project directory
cd veyra-ai

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# 4. Install dependencies (this may take a few minutes for PyTorch + Whisper)
pip install -r requirements.txt

# 5. Run the server
python app.py

# 6. Open your browser
# Navigate to: http://localhost:5000
```

> **Note:** The first run will download the Whisper `base` model (~140 MB) and the Resemblyzer speaker encoder. This is a one-time download.

---

## Usage Guide

### 1. Voice Enrollment

1. Click **"🎤 Enroll Voice"** on the sidebar
2. A modal will appear with 5 prompted sentences
3. For each sentence:
   - Click the microphone button to start recording
   - Speak the sentence clearly for 2–3 seconds
   - Click again to stop
4. After all 5 samples, the system creates your voice profile
5. The sidebar will show **"Voice Enrolled"** with a green indicator

### 2. Voice Commands

1. Click the large microphone button in the center
2. Speak your command (e.g., "What time is it?")
3. Click again to stop recording
4. Watch the pipeline steps light up as each module processes
5. The assistant's response appears with both text and audio

### 3. Supported Commands

| Intent | Example Phrases |
|--------|----------------|
| Greeting | "Hello", "Hey", "Good morning" |
| Time | "What time is it?", "Tell me the time" |
| Date | "What's today's date?", "What day is it?" |
| Calculate | "Calculate 5 plus 3", "What is 12 times 4?" |
| Joke | "Tell me a joke", "Make me laugh" |
| Name | "What is your name?", "Who are you?" |
| Capabilities | "What can you do?", "Help me" |
| Status | "How are you?", "System status" |
| Farewell | "Goodbye", "See you later" |

---

## Module Details

### 1. Noise Suppression (`noise_suppression.py`)

- **Algorithm:** Spectral gating via `noisereduce`
- **Approach:** Estimates a noise profile assuming stationary background noise, then subtracts the noise spectrum while preserving speech frequencies
- **Parameters:** `prop_decrease=0.8`, `n_fft=2048`, `hop_length=512`
- **Output:** Cleaned audio with noise level reported in dBFS

### 2. Speaker Authentication (`speaker_auth.py`)

- **Model:** Resemblyzer's pre-trained speaker encoder (GE2E loss)
- **Embedding:** 256-dimensional d-vector per utterance
- **Enrollment:** Averages d-vectors from 3–5 voice samples
- **Verification:** Cosine similarity between new audio and stored profile
- **Threshold:** ≥ 0.75 = authorized

### 3. Speech-to-Text (`stt_engine.py`)

- **Model:** OpenAI Whisper (base, ~74M parameters)
- **Input:** 16 kHz mono WAV
- **Features:** Multi-language support (locked to English), robust to noise
- **Output:** Transcribed text string

### 4. NLP Engine (`nlp_engine.py`)

- **Method:** Rule-based intent classification using regex patterns
- **Intents:** 13 categories (greeting, farewell, time, date, weather, calculate, name, capabilities, joke, compliment, status, music, search)
- **Scoring:** Keyword match density → confidence score
- **Dynamic responses:** Time, date, and calculations are computed at runtime

### 5. TTS Engine (`tts_engine.py`)

- **Library:** pyttsx3 (offline)
- **Voice:** System default (prefers female voice if available)
- **Rate:** 160 WPM
- **Output:** WAV file served via Flask endpoint

---

## Algorithms & Methodology

### D-Vector Speaker Embedding

The speaker encoder transforms a variable-length audio signal into a fixed 256-dimensional vector (d-vector) that captures the speaker's unique vocal characteristics. The model was trained using **Generalized End-to-End Loss (GE2E)** on thousands of speakers.

**Enrollment:**
```
voice_profile = mean(d-vector(sample_1), d-vector(sample_2), ..., d-vector(sample_n))
```

**Verification:**
```
similarity = cos(voice_profile, d-vector(new_audio))
authorized = similarity ≥ 0.75
```

### Cosine Similarity

```
cos(A, B) = (A · B) / (‖A‖ × ‖B‖)
```

Returns a value in [-1, 1], where 1 = identical, 0 = orthogonal, -1 = opposite.

### Spectral Gating

1. Compute STFT of the audio signal
2. Estimate noise profile from the signal (assuming stationary noise)
3. For each frequency bin, if energy < noise_threshold → attenuate
4. Reconstruct audio via inverse STFT

---

## Security Considerations

- **Voice profile storage:** Embeddings are stored as `.npy` files locally — no cloud transmission
- **No raw audio stored:** Only the 256-dim embedding is saved, not the original recordings
- **Threshold tuning:** The 0.75 threshold balances security vs. usability. Increase for stricter access.
- **Replay attack:** This prototype does not include liveness detection. Production systems should add anti-spoofing measures.
- **Safe math evaluation:** The calculator uses a restricted `eval()` with no builtins to prevent code injection.

---

## Future Enhancements

1. **Wake word detection** — "Hey Veyra" to activate listening
2. **Continuous listening mode** — always-on with voice activity detection
3. **LLM integration** — Connect to Gemini or GPT for open-ended conversations
4. **Multi-user profiles** — Support for multiple authorized users
5. **Liveness detection** — Anti-spoofing to prevent replay attacks
6. **Emotion recognition** — Detect mood from speech patterns
7. **Smart home integration** — Control IoT devices via voice commands

---

## References

1. Wan, L., et al. (2018). *Generalized End-to-End Loss for Speaker Verification.* IEEE ICASSP.
2. Radford, A., et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision.* OpenAI.
3. Sainburg, T. (2019). *Noisereduce: Noise Reduction in Python using Spectral Gating.* GitHub.
4. Jemine, C. (2019). *Real-Time Voice Cloning (Resemblyzer).* GitHub.
5. pyttsx3 documentation: https://pyttsx3.readthedocs.io/

---

<p align="center">
  Built with ❤️ for academic demonstration<br/>
  <strong>VEYRA AI</strong> — Your Voice, Your Key
</p>
