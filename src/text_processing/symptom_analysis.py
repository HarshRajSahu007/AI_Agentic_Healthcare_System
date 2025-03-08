from transformers import pipeline

def analyze_symptons(text):
    classifier = pipeline("text-classification",model="distilbert-base-uncased")
    result=classifier(text)
    return result

if __name__=="__main__":
    symptons="I have a fever and cough,"
    analysis=analyze_symptons(symptons)
    print (analysis)
