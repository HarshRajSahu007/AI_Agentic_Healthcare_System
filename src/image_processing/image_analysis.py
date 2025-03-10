import cv2
from torchvision import models, transforms
import torch

def analyze_image(image_path):
 
    model = models.resnet18(pretrained=True)
    model.eval()


    image = cv2.imread(image_path)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)


    with torch.no_grad():
        output = model(image)
    return output


if __name__ == "__main__":
    result = analyze_image("image.jpg")