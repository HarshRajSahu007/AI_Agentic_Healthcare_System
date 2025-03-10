from transformers import pipeline

def analyze_symptoms(text):
    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    result = classifier(text)
    return result

if __name__ == "__main__":
    symptoms = "I have a fever and cough."
    analysis = analyze_symptoms(symptoms)
    print(analysis)
