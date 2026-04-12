import wave
import struct

def generate_silence_wav(filename, duration_sec=2, framerate=16000):
    nframes = int(duration_sec * framerate)
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        silence = struct.pack('<h', 0) * nframes
        wf.writeframes(silence)

if __name__ == "__main__":
    generate_silence_wav("silent.wav", duration_sec=2)
