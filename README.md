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

## Notes

- Designed to run on CPU; no external GPU service required.
- Uses `../yolov8n.pt` by default from repository root.
- If Whisper model load fails, speech-state tracking still runs and transcript can remain empty.
