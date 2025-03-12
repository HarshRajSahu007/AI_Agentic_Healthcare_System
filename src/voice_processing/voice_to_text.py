import whisper
import librosa
import tempfile
import os

def transcribe_audio(audio_file):
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_file_path = tmp_file.name

    # Load the audio file using librosa
    waveform, sample_rate = librosa.load(tmp_file_path, sr=None)

    # Transcribe the audio using Whisper
    model = whisper.load_model("base")
    result = model.transcribe(tmp_file_path)

    # Clean up the temporary file
    os.remove(tmp_file_path)

    return result["text"]

# Example usage
if __name__ == "__main__":
    with open("path_to_audio.mp3", "rb") as f:
        text = transcribe_audio(f)
        print(text)