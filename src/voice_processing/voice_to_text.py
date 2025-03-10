import whisper
import numpy as np
import torchaudio

def transcribe_audio(audio_path):
    print(f"Processing: {audio_path}")

    # Load and check audio format
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Ensure the waveform is a NumPy array
    audio_array = waveform.numpy()
    audio_array = np.array(audio_array, dtype=np.float32)  # Ensure correct dtype

    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

if __name__ == "__main__":
    text = transcribe_audio("audio.mp3")
    print(text)
