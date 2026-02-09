# 🧠 Vision Assistant

A real-time multimodal **vision assistant** built using **SmolVLM (500M)** via **llama.cpp**.  
The system supports **image understanding**, **video summarization**, and **live camera narration with Q&A**, while actively preventing hallucinations.

---

## ✨ Features

- 🖼️ **Image Summary**
  - Describes visible objects, people, and actions
  - No guessing or inference

- 🎞️ **Video Summary**
  - Samples multiple frames
  - Detects only visual changes
  - Produces a chronological summary

- 📹 **Live Vision Mode**
  - Real-time camera narration
  - Describes only what changed
  - Adjustable frame interval
  - Toggle voice narration
  - Text & voice-based questions

- 🔊 **Voice Support**
  - Text-to-speech narration
  - Voice-based questions (browser speech API)

- 🛡️ **Hallucination Control**
  - Strict prompts
  - Token limits per task
  - Context-aware live narration

---

## ⚙️ Requirements

### System
- Windows / Linux / macOS
- Webcam (for live mode)

### Software
- **Python 3.10 (recommended)**
- **llama.cpp**
- **SmolVLM-500M-Instruct (GGUF)**

---

## 🧩 Step 1: Install llama.cpp

### Windows
1. Download prebuilt binaries from:


https://github.com/ggerganov/llama.cpp/releases

2. Extract and add the folder to your `PATH`

### Linux / macOS
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
```

Step 2: Run SmolVLM with llama.cpp

Run this command in a separate terminal:
```
llama-server \
  -hf ggml-org/SmolVLM-500M-Instruct-GGUF \
  --ctx-size 2048 \
  --n-gpu-layers 0 \
  --threads 6 \
  --port 8080 \
  --no-jinja \
  --chat-template chatml
```

Expected output:

server is listening on http://127.0.0.1:8080

🐍 Step 3: Backend Setup
```
cd smol_vlm/backend
python -m venv .venv
```

Activate virtual environment:

Windows
```
.venv\Scripts\activate
```

Linux / macOS
```
source .venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```
▶️ Step 4: Run Backend (FastAPI)
```
uvicorn app:app --port 8000
```

Backend runs at:

http://127.0.0.1:8000

🌐 Step 5: Open Frontend

Open in browser:

Image & Video Summary

http://127.0.0.1:8000/


Live Vision Mode

http://127.0.0.1:8000/live

🎥 How Live Mode Works

Captures webcam frames

Sends frames at a configurable interval

Describes only visual changes

Pauses narration during Q&A

Automatically resumes narration

🧠 Token Profiles
Mode	Max Tokens
Live narration	12
Q&A	80
Image summary	160
Video summary	320

This prevents:

Over-verbose responses

Truncated summaries

Hallucinations

⚠️ Known Limitations

SmolVLM is not a temporal model

Live narration relies on previous-frame context

Video processing is frame-based (not streaming)

🧪 Tested With

Python 3.10.x

llama.cpp (latest)

SmolVLM-500M-Instruct

Chrome / Edge (recommended for speech APIs)

📌 Notes

Use good lighting for live mode

Increase live interval if narration feels delayed

Voice input works best in Chromium-based browsers

🙌 Credits

llama.cpp – GGUF inference engine

HuggingFace – SmolVLM model

FastAPI – Backend framework

📬 Usage

This project is suitable for:

Research demos

Computer vision experiments

Multimodal AI projects