from transformers import AutoTokenizer, AutoModelForTokenClassification
import spacy
import torch

class TextProcessingAgent:
    def __init__(self, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ner_tokenizer = AutoTokenizer.from_pretrained(config["ner_model"])
        self.ner_model = AutoModelForTokenClassification.from_pretrained(config["ner_model"]).to(self.device)
        self.nlp = spacy.load("en_core_web_sm")
        
    def extract_medical_entities(self, text):
        inputs = self.ner_tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.ner_model(**inputs)
        
        predictions = torch.argmax(outputs.logits, dim=2)
        tokens = self.ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        entities = []
        
        current_entity = ""
        current_label = ""
        for token, prediction in zip(tokens, predictions[0]):
            label = self.ner_model.config.id2label[prediction.item()]
            if label.startswith("B-"):
                if current_entity:
                    entities.append((current_entity.strip(), current_label))
                current_entity = token
                current_label = label[2:]
            elif label.startswith("I-"):
                current_entity += " " + token
            else:
                if current_entity:
                    entities.append((current_entity.strip(), current_label))
                current_entity = ""
                current_label = ""
        
        return entities
    
    def process_text(self, text):
        doc = self.nlp(text)
        symptoms = [ent.text for ent in doc.ents if ent.label_ == "SYMPTOM"]
        conditions = [ent.text for ent in doc.ents if ent.label_ == "CONDITION"]
        medications = [ent.text for ent in doc.ents if ent.label_ == "MEDICATION"]
        
        entities = self.extract_medical_entities(text)
        
        return {
            "symptoms": symptoms,
            "conditions": conditions,
            "medications": medications,
            "named_entities": entities,
            "processed_text": text
        }