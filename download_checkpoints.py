import os
import gdown

os.makedirs('SadTalker/checkpoints', exist_ok=True)

# Download SadTalker_V0.0.2_256.safetensors (example, replace with actual file IDs as needed)
# You can find the latest links in the official SadTalker repo or HuggingFace

# Example Google Drive file ID for SadTalker_V0.0.2_256.safetensors
# Replace with the correct file ID or add more files as needed
MODEL_FILES = [
    {
        'id': '1F7w6nKjQwQnQwQnQwQnQwQnQwQnQwQn',  # <-- Replace with actual file ID
        'output': 'SadTalker/checkpoints/SadTalker_V0.0.2_256.safetensors'
    },
]

for file in MODEL_FILES:
    url = f"https://drive.google.com/uc?id={file['id']}"
    print(f"Downloading {file['output']} ...")
    gdown.download(url, file['output'], quiet=False)
print("All model files downloaded.")
