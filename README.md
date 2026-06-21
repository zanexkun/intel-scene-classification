# Intel Image Classification

Two approaches to classifying natural scene images into 6 categories: a custom CNN built from scratch, and Transfer Learning with ResNet18.

**Best result: 93% test accuracy** (ResNet18, fine-tuned)

---

## Dataset

[Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) — 25,000 images, 6 classes: buildings, forest, glacier, mountain, sea, street.

---

## Results

| Approach | Accuracy |
|----------|----------|
| Custom CNN (3 conv blocks, BatchNorm, Dropout, augmentation) | 87% |
| ResNet18 — feature extraction only | 90% |
| ResNet18 — fine-tuned (layer4 + fc unfrozen) | **93%** |

Per-class F1 score, fine-tuned ResNet18:

| Class | F1-Score |
|-------|----------|
| Buildings | 0.92 |
| Forest | 0.99 |
| Glacier | 0.89 |
| Mountain | 0.89 |
| Sea | 0.97 |
| Street | 0.93 |

---

## Part 1: Custom CNN (`intel_scene_classification.py`)

Built from scratch, no pretrained weights.

3 conv blocks (Conv → BatchNorm → ReLU → MaxPool), channels 3 → 64 → 64 → 128, followed by a 4-layer FC classifier with Dropout (p=0.2).

Trained with SGD, ReduceLROnPlateau scheduler, and data augmentation (horizontal flip, rotation, color jitter) on the training set only.

Got to 87% by going through several rounds of diagnosis — checking train/val loss curves, fixing overfitting and underfitting, and adding regularization step by step. Started at 81% with a bare 2-block CNN.

## Part 2: Transfer Learning (`transfer_learning.py`)

Same dataset, ResNet18 pretrained on ImageNet.

**Stage 1 — Feature extraction:** froze all ResNet weights, replaced the final layer with a `Linear(512, 6)` for our classes. Only this new layer trains. Got to 90%.

**Stage 2 — Fine-tuning:** unfroze `layer4` (the last conv block) along with the classifier, and continued training at a much lower learning rate (0.0001) so the pretrained weights adjust to this dataset without being destroyed. Got to 93%.

The jump from 87% to 90% with just a frozen pretrained model (no extra effort beyond swapping the last layer) is the clearest illustration of why Transfer Learning is the default approach in real CV work rather than training from scratch.

---

## Requirements

```
torch
torchvision
scikit-learn
```

## Usage

Download the dataset from Kaggle, update `TRAIN_DIR` and `TEST_DIR` at the top of either script, then run:

```bash
python intel_scene_classification.py
# or
python transfer_learning.py
```
