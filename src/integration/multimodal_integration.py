from voice_processing.voice_to_text import transcribe_audio
from text_processing.symptom_analysis import analyze_symptoms
from image_processing.image_analysis import analyze_image

def process_inputs(audio_file, image_file):
    # Step 1: Transcribe audio
    symptoms = transcribe_audio(audio_file)

    # Step 2: Analyze symptoms
    symptom_analysis = analyze_symptoms(symptoms)

    # Step 3: Analyze image
    image_analysis = analyze_image(image_file)

    # Step 4: Combine results
    return {
        "symptoms": symptom_analysis,
        "image_analysis": image_analysis,
    }

# Example usage
if __name__ == "__main__":
    with open("path_to_audio.mp3", "rb") as audio_file, open("path_to_image.jpg", "rb") as image_file:
        result = process_inputs(audio_file, image_file)
        print(result)