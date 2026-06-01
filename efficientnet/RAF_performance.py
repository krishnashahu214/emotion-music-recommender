import torch
import timm
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_rafdb_model.pth"

# 👉 CHANGE THIS PATH
VAL_DIR = r"C:\Users\ASUS\OneDrive\Desktop\Projects\DL_Project\DATASET\test"   # example: "dataset/val"

CLASS_NAMES = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

# =========================
# TRANSFORM
# =========================
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# DATASET
# =========================
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# =========================
# MODEL (MATCH TRAINING)
# =========================
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.base_model = timm.create_model(
            'efficientnet_b0',
            pretrained=False,
            num_classes=0
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
# EVALUATION
# =========================
def evaluate_model(model, dataloader):
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"\n Validation Accuracy: {accuracy:.2f}%")

    return all_preds, all_labels

# =========================
# CONFUSION MATRIX
# =========================
def plot_confusion(all_preds, all_labels):
    cm = confusion_matrix(all_labels, all_preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=CLASS_NAMES)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    model = load_model()

    preds, labels = evaluate_model(model, val_loader)

    # OPTIONAL (for presentation)
    plot_confusion(preds, labels)