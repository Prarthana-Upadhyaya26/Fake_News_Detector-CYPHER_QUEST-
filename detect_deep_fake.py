import torch
from torchvision import models, transforms
from PIL import Image
import sys

# Define a function to load and preprocess the image
image_path="C:\Users\prart\OneDrive\Desktop\real.jpg",
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0)  # Add batch dimension

# Define a function to classify the image
def classify_image(image_path, model):
    model.eval()
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

# Main script
if __name__ == "_main_":
    if len(sys.argv) != 2:
        print("Usage: python detect_fake_image.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Load a pre-trained model (ResNet for this example)
    model = models.resnet18(pretrained=True)

    # You can replace this with your own trained model if available
    print("Classifying image...")
    class_id = classify_image(image_path, model)
    print(f"Class ID: {class_id}")