import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models
from sklearn.metrics import classification_report


# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────
TRAIN_DIR = "data/seg_train/seg_train"
TEST_DIR  = "data/seg_test/seg_test"

BATCH_SIZE  = 32
IMG_SIZE    = 224
NUM_CLASSES = 6
CHECKPOINT  = "best_resnet.pth"

FEATURE_EXTRACT_EPOCHS = 50
FINE_TUNE_EPOCHS       = 40

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────
# ResNet was trained on ImageNet, so we use ImageNet's mean/std
# instead of a generic 0.5 normalization, and resize to 224x224
# to match what the pretrained weights expect.
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────
# Dataset & DataLoaders
# ─────────────────────────────────────────
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
# Model — ResNet18 pretrained on ImageNet
# ─────────────────────────────────────────
model = models.resnet18(pretrained=True)

# Stage 1: freeze every layer, replace the classifier head for our 6 classes
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(512, NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss()


def run_epoch(model, loader, optimizer=None):
    """Runs one pass over `loader`. Trains if optimizer is given, else evaluates."""
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            if is_training:
                optimizer.zero_grad()

            preds = model(x_batch)
            loss  = criterion(preds, y_batch)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(loader)


def train_loop(model, optimizer, scheduler, epochs, checkpoint_path, best_val_loss=float("inf")):
    for epoch in range(epochs):
        train_loss = run_epoch(model, train_loader, optimizer)
        val_loss   = run_epoch(model, val_loader)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")

    return best_val_loss


# ─────────────────────────────────────────
# Stage 1: Feature extraction
# Only the new fc layer trains. Everything else stays frozen.
# ─────────────────────────────────────────
print("\n--- Stage 1: Feature extraction (fc layer only) ---")

optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

best_val_loss = train_loop(model, optimizer, scheduler, FEATURE_EXTRACT_EPOCHS, CHECKPOINT)


# ─────────────────────────────────────────
# Stage 2: Fine-tuning
# Unfreeze layer4 (the last conv block) and train it together with fc,
# using a much lower learning rate so we don't destroy the pretrained weights.
# ─────────────────────────────────────────
print("\n--- Stage 2: Fine-tuning (layer4 + fc) ---")

model.load_state_dict(torch.load(CHECKPOINT))

for param in model.layer4.parameters():
    param.requires_grad = True

optimizer = optim.SGD([
    {"params": model.layer4.parameters()},
    {"params": model.fc.parameters()},
], lr=0.0001, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

train_loop(model, optimizer, scheduler, FINE_TUNE_EPOCHS, CHECKPOINT, best_val_loss=best_val_loss)


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
