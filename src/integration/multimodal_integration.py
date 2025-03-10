from voice_processing.voice_to_text import transcribe_audio
from text_processing.symptom_analysis import analyze_symptoms
from image_processing.image_analysis import analyze_image

def process_inputs(audio_path, image_path):
    # Step 1: Transcribe audio
    symptoms = transcribe_audio(audio_path)

    # Step 2: Analyze symptoms
    symptom_analysis = analyze_symptoms(symptoms)

    # Step 3: Analyze image
    image_analysis = analyze_image(image_path)

    # Step 4: Combine results
    return {
        "symptoms": symptom_analysis,
        "image_analysis": image_analysis,
    }

# Example usage
if __name__ == "__main__":
    result = process_inputs("/Users/harshrajsahu/Desktop/AI_Hack/AI-Healthcare-System/data/voice.mp3", "/Users/harshrajsahu/Desktop/AI_Hack/AI-Healthcare-System/data/cut_pic.jpg")
    print(result)