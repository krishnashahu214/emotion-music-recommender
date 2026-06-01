import torch
import timm
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import sys

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_rafdb_model.pth"
CLASS_NAMES = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# MODEL (MATCH TRAINING)
# =========================
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.base_model = timm.create_model(
            'efficientnet_b0',
            pretrained=False,
            num_classes=0   # IMPORTANT → remove default classifier
        )

        in_features = self.base_model.num_features

        
        self.base_model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 7)
        )

    def forward(self, x):
        return self.base_model(x)
# =========================
# LOAD MODEL
# =========================
def load_model():
    model = MyModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# =========================
# PREDICT FUNCTION
# =========================
def predict(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    Prediction = CLASS_NAMES[pred.item()]
    Confidence = round(conf.item() * 100, 2)

    return Prediction, Confidence

    # print("\n Image:", image_path)
    # print(" Prediction:", CLASS_NAMES[pred.item()])
    # print(" Confidence:", round(conf.item() * 100, 2), "%")

# # =========================
# # MAIN
# # =========================
# if __name__ == "__main__":
    
#     model = load_model()

#     # OPTION 1: command line input
#     if len(sys.argv) > 1:
#         predict(sys.argv[1], model)

#     # OPTION 2: manual loop
#     else:
#         while True:
#             path = input("\nEnter image path (or 'exit'): ")
#             if path.lower() == "exit":
#                 break
#             predict(path, model)