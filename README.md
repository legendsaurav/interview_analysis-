# Realtime AI Interview System

This module adds a local, CPU-friendly, real-time interview pipeline with synchronized audio + video analysis.

## Structure

- `frontend/index.html`
- `frontend/app.js`
- `frontend/styles.css`
- `backend/main.py`
- `backend/audio_processor.py`
- `backend/vision_processor.py`
- `backend/state_machine.py`
- `backend/yolo_model.py`

## Features

- WebSocket streaming for microphone PCM audio and webcam JPEG frames
- Real-time speech state (`speaking`, `paused`, `finished`) with silence-end detection
- Optional Whisper transcription using `faster-whisper`
- MediaPipe face behavior analysis:
  - eye-contact proxy score
  - head stability score
  - gaze direction
- YOLOv8 object detection with frame skipping:
  - phone detection alert
  - multiple-person alert
- Interview state machine:
  - `IDLE`, `ASKING_QUESTION`, `LISTENING`, `PROCESSING`, `RESPONDING`
- Live frontend indicators and browser TTS responses

## Run

From the `INTERVIEWER` directory:

```powershell
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://127.0.0.1:8000`.

## Deploy Without Endpoint Errors

This app uses realtime WebSocket streaming. The most stable production setup is:

- Frontend: Vercel
- Backend (FastAPI + WebSocket): Render, Railway, Fly.io, VM, or Docker host

### 1) Push to GitHub

From the repository root:

```powershell
git init
git add .
git commit -m "Initial interview system"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

### 2) Deploy backend first

Use the `INTERVIEWER` folder as the backend project root.

- Install command: `pip install -r requirements.txt`
- Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

Set environment variables on backend host:

- `GEMINI_API_KEY=<your-key>`
- `GEMINI_MODEL=gemini-1.5-flash`
- `INTERVIEW_ALLOWED_ORIGINS=https://<your-vercel-domain>`

Health check URL:

- `/health`

### 3) Deploy frontend on Vercel

- Import GitHub repo in Vercel
- Set Root Directory to `INTERVIEWER`
- Vercel uses `vercel.json` included in this folder

After deploy, edit `frontend/runtime-config.js` to point to backend:

```javascript
window.__INTERVIEW_CONFIG__ = {
  backendUrl: "https://<your-backend-domain>",
  wsUrl: "",
};
```

`wsUrl` can stay empty, it will be derived from `backendUrl`.

### Why this avoids endpoint errors

- Frontend API calls now use a configurable backend base URL
- WebSocket URL is configurable and has safe fallbacks
- CORS is controlled by `INTERVIEW_ALLOWED_ORIGINS`
- `/health` endpoint helps verify backend reachability before UI tests

## Notes

- Designed to run on CPU; no external GPU service required.
- Uses `../yolov8n.pt` by default from repository root.
- If Whisper model load fails, speech-state tracking still runs and transcript can remain empty.
"# interview_analysis-" 
