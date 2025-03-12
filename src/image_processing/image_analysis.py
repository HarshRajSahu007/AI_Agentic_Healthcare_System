import cv2
import tempfile
import os
from torchvision import models, transforms
import torch
import json

def analyze_image(image_file):
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(image_file.read())
        tmp_file_path = tmp_file.name

    # Load the image using OpenCV
    image = cv2.imread(tmp_file_path)

    # Preprocess the image for the model
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)

    # Load a pre-trained model (e.g., ResNet)
    model = models.resnet18(pretrained=True)
    model.eval()

    # Perform inference
    with torch.no_grad():
        output = model(image)

    # Convert the output to a JSON-serializable format
    output = output.squeeze().tolist()  # Convert tensor to list

    # Clean up the temporary file
    os.remove(tmp_file_path)

    # Return the result as a dictionary
    return {
        "output": output,
        "message": "Image analysis completed successfully."
    }

# Example usage
if __name__ == "__main__":
    with open("path_to_image.jpg", "rb") as f:
        result = analyze_image(f)
        print(json.dumps(result, indent=4))