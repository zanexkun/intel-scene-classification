import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report


# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────
TRAIN_DIR = "data/seg_train/seg_train"
TEST_DIR  = "data/seg_test/seg_test"

BATCH_SIZE  = 32
EPOCHS      = 80
LR          = 0.0005
MOMENTUM    = 0.9
IMG_SIZE    = 150
NUM_CLASSES = 6
CHECKPOINT  = "best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


# ─────────────────────────────────────────
# Dataset & DataLoaders
# ─────────────────────────────────────────
# Load the training folder twice with different transforms so the
# validation split receives clean (non-augmented) images.
train_dataset    = ImageFolder(root=TRAIN_DIR, transform=train_transforms)
val_dataset_full = ImageFolder(root=TRAIN_DIR, transform=val_transforms)

train_size = int(0.8 * len(train_dataset))
val_size   = len(train_dataset) - train_size
indices    = torch.randperm(len(train_dataset))

train_set = torch.utils.data.Subset(train_dataset,    indices[:train_size])
val_set   = torch.utils.data.Subset(val_dataset_full, indices[train_size:])
test_set  = ImageFolder(root=TEST_DIR, transform=val_transforms)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")


# ─────────────────────────────────────────
# Model
# ─────────────────────────────────────────
class IntelCNN(nn.Module):
    """
    Custom CNN for 6-class natural scene classification.

    Architecture:
        3 convolutional blocks (Conv → BatchNorm → ReLU → MaxPool)
        followed by a 4-layer fully-connected classifier with Dropout.

    Input:  (B, 3, 150, 150)
    Output: (B, 6)
    """
    def __init__(self, num_classes: int = 6):
        super(IntelCNN, self).__init__()

        # Conv block 1: 3 → 64 channels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.bn1   = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Conv block 2: 64 → 64 channels
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Conv block 3: 64 → 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Classifier head
        # Spatial size after 3 conv+pool blocks on 150×150 input: 17×17
        self.fc1     = nn.Linear(128 * 17 * 17, 128)
        self.fc2     = nn.Linear(128, 64)
        self.fc3     = nn.Linear(64, 32)
        self.fc4     = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = torch.flatten(x, 1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


model     = IntelCNN(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)


# ─────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    # ── Training ──────────────────────────
    model.train()
    running_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        preds = model(x_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # ── Validation ────────────────────────
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds    = model(x_batch)
            val_loss += criterion(preds, y_batch).item()

    avg_train = running_loss / len(train_loader)
    avg_val   = val_loss    / len(val_loader)

    scheduler.step(avg_val)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), CHECKPOINT)

    print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
          f"Train Loss: {avg_train:.4f} | "
          f"Val Loss:   {avg_val:.4f}")


# ─────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────
model.load_state_dict(torch.load(CHECKPOINT))
model.eval()

all_preds, all_labels = [], []
correct = total = 0

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        preds = model(x_batch)

        _, predicted = torch.max(preds, 1)
        total   += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

accuracy = 100 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%\n")
print(classification_report(
    all_labels,
    all_preds,
    target_names=test_set.classes,
))
