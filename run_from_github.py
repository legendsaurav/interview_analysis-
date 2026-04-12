#!/usr/bin/env python3
"""
Run SadTalker pipeline from a single interview text file pushed by INTERVIEWER backend.
- Expects a file like 'latest_interview.txt' in the repo root or a known location.
- File format:
  character: Akash
  Q: Tell me about yourself
  Q: ...
- Maps character to image in assets/photos/interviewer/
- Generates audio for each question, then runs SadTalker to generate video.
"""
import os
from pathlib import Path
import sys
import shutil
import subprocess
import time
import urllib.parse
import urllib.request

# CONFIG
INTERVIEW_FILE = Path('latest_interview.txt')  # or Path('path/to/latest_interview.txt')
AVATAR_DIR = Path('assets/photos/interviewer')
AUDIO_DIR = Path('generated/audio')
VIDEO_DIR = Path('generated/video')
SADTALKER_DIR = Path('SadTalker')
SADTALKER_PYTHON = sys.executable  # or specify python path if needed
VOICE_LANG = 'en-us'
VOICE_CODEC = 'MP3'
VOICERSS_API_KEY = os.environ.get('VOICERSS_API_KEY', '759c79c9515242148848e58daaf0d74c')

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

def parse_interview_file(file_path):
    character = None
    questions = []
    for line in file_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line.lower().startswith('character:'):
            character = line.split(':', 1)[1].strip()
        elif line.startswith('Q:'):
            questions.append(line.split('Q:', 1)[1].strip())
    return character, questions

def find_avatar(character):
    # Look for image file matching character name (case-insensitive, any extension)
    for ext in ['.png', '.jpg', '.jpeg', '.webp']:
        candidate = AVATAR_DIR / f"{character.lower()}{ext}"
        if candidate.exists():
            return candidate
        # Try capitalized
        candidate = AVATAR_DIR / f"{character.capitalize()}{ext}"
        if candidate.exists():
            return candidate
    # Fallback: first image in folder
    files = list(AVATAR_DIR.glob('*'))
    return files[0] if files else None

def synthesize_voicerss(text, output_file, lang=VOICE_LANG, codec=VOICE_CODEC):
    url = "https://api.voicerss.org/"
    params = {
        "key": VOICERSS_API_KEY,
        "hl": lang,
        "src": text,
        "c": codec,
    }
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}"
    try:
        with urllib.request.urlopen(full_url) as response:
            data = response.read()
        if data.startswith(b'ERROR'):
            print(f"VoiceRSS error for text '{text}': {data.decode()}")
            return False
        with open(output_file, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"Failed to synthesize audio for '{text}': {e}")
        return False

def run_sadtalker(source_image, audio_path, result_dir):
    inference_py = SADTALKER_DIR / "inference.py"
    result_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        SADTALKER_PYTHON, str(inference_py),
        "--driven_audio", str(audio_path),
        "--source_image", str(source_image),
        "--result_dir", str(result_dir),
        "--preprocess", "full"
    ]
    process = subprocess.run(cmd, cwd=SADTALKER_DIR)
    if process.returncode != 0:
        raise RuntimeError("SadTalker failed")
    # Find the latest mp4 in result_dir
    videos = sorted(result_dir.glob('*.mp4'), key=os.path.getmtime, reverse=True)
    return videos[0] if videos else None

def main():
    if not INTERVIEW_FILE.exists():
        print(f"[ERROR] Interview file not found: {INTERVIEW_FILE}")
        sys.exit(1)
    character, questions = parse_interview_file(INTERVIEW_FILE)
    if not character or not questions:
        print("[ERROR] Interview file missing character or questions.")
        sys.exit(1)
    avatar_img = find_avatar(character)
    if not avatar_img:
        print(f"[ERROR] Avatar image not found for character: {character}")
        sys.exit(1)
    print(f"[INFO] Using avatar: {avatar_img}")
    for idx, question in enumerate(questions, 1):
        audio_file = AUDIO_DIR / f"q{idx:03d}.mp3"
        print(f"[INFO] Synthesizing audio for Q{idx}: {question}")
        if not synthesize_voicerss(question, audio_file):
            print(f"[ERROR] Failed to synthesize audio for: {question}")
            continue
        print(f"[INFO] Running SadTalker for Q{idx}")
        video_file = run_sadtalker(avatar_img, audio_file, VIDEO_DIR)
        print(f"[INFO] Generated video: {video_file}")
    print("[DONE] All questions processed.")

if __name__ == "__main__":
    main()
