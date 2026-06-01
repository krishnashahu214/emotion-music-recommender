import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import timm
from tqdm import tqdm
from collections import Counter

# =========================
# 1. DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# =========================
# 2. TRANSFORMS
# =========================
transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# =========================
# 3. DATASET
# =========================
train_dir = r'C:\Users\ASUS\OneDrive\Desktop\Projects\DL_Project\DATASET\train'
val_dir   = r'C:\Users\ASUS\OneDrive\Desktop\Projects\DL_Project\DATASET\test'

train_dataset = datasets.ImageFolder(train_dir, transform=transform['train'])
val_dataset   = datasets.ImageFolder(val_dir, transform=transform['val'])

print("Classes:", train_dataset.classes)

# =========================
# 4. WEIGHTED SAMPLER (FIX CLASS COLLAPSE)
# =========================
targets = train_dataset.targets
class_counts = Counter(targets)

class_weights_dict = {cls: 1.0 / count for cls, count in class_counts.items()}
sample_weights = [class_weights_dict[label] for label in targets]

sampler = WeightedRandomSampler(
    sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)

# =========================
# 5. CLASS WEIGHTS (FOR LOSS)
# =========================
num_classes = len(train_dataset.classes)
total_samples = len(targets)

class_weights = []
for i in range(num_classes):
    weight = total_samples / (num_classes * class_counts[i])
    class_weights.append(weight)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# =========================
# 6. MODEL
# =========================
class EmotionModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        
        in_features = self.base_model.classifier.in_features
        
        self.base_model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),  # reduced dropout
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

model = EmotionModel().to(device)

# =========================
# 7. LOSS & OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss(weight=class_weights)  # no label smoothing

optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=2, factor=0.5, verbose=True
)

# =========================
# 8. EARLY STOPPING
# =========================
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best = None
        self.counter = 0

    def step(self, metric):
        if self.best is None or metric > self.best:
            self.best = metric
            self.counter = 0
            return False
        
        self.counter += 1
        print(f"EarlyStopping: {self.counter}/{self.patience}")
        
        return self.counter >= self.patience

early_stopping = EarlyStopping(patience=5)

# =========================
# 9. VALIDATION + DEBUG
# =========================
def validate():
    model.eval()
    correct = 0
    total = 0
    pred_list = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pred_list.extend(predicted.cpu().numpy())

    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")
    print("Prediction distribution:", Counter(pred_list))  # DEBUG
    
    return acc

# =========================
# 10. TRAIN LOOP
# =========================
def train_model(epochs=30):
    best_acc = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping (IMPORTANT)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        avg_loss = running_loss / len(train_loader)
        
        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {avg_loss:.4f}")
        
        val_acc = validate()
        
        # Scheduler
        scheduler.step(val_acc)
        
        # Save last model
        torch.save(model.state_dict(), "last_model.pth")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Best model saved!")
        
        # Early stopping
        if early_stopping.step(val_acc):
            print("⛔ Early stopping triggered!")
            break

# =========================
# 11. START TRAINING
# =========================
train_model(epochs=30)