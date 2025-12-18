import torch
from torchvision import models, transforms
from PIL import Image

# Load your saved model
model = models.resnet18()
num_classes = 6  # Adjust to your dataset
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

model_path = "defect_model_resnet18.pth"
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Class names
classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

def predict_image(image: Image.Image):
    """
    Input: PIL image
    Output: dict with class name and confidence
    """
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        return {"class": classes[pred.item()], "confidence": conf.item()}