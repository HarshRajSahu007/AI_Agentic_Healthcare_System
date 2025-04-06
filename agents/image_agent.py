import cv2
import torch
from transformers import ViTImageProcessor, ViTForImageClassification

class ImageProcessingAgent:
    def __init__(self,config):
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.processor=ViTImageProcessor.from_pretrained(config["vit_model"])
        self.model=ViTForImageClassification.from_pretrained(config["vit_model"]).to(self.device)
        self.desease_classes=config["disease_classes"]

    def process_image(self,image_path):
        image=cv2.imread(image_path)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        inputs=self.processor(image=image,return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs=self.model(**inputs)
            logits=outputs.logits
            probs=torch.nn.functional.softmax(logits,dim=-1)

        predicted_class_idx=logits.argmax(-1).item()
        predicted_class=self.model.config.id2label[predicted_class_idx]

        confidence=probs[0][predicted_class_idx].item()

        return {
            "predicted_class":predicted_class,
            "confidence":confidence,
            "all_probs":probs.cpu().numpy(),
            "image_features":outputs.last_hidden_state.cpu().numpy()
        }
