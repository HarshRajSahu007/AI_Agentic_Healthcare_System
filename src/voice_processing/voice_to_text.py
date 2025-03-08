import whisper
def transcribe_audio(audio_path):
    model=whisper.load_model("base")
    result=model.transcribe(audio_path)
    return result["text"]

if __name__=="__main__":
    text=transcribe_audio("path_to_audio.mp3")
    print(text)
