# Intel Image Classification — Custom CNN

A custom Convolutional Neural Network built from scratch in PyTorch to classify natural scenes into 6 categories. Achieves **87% test accuracy** without using any pretrained models or transfer learning.

---

## Dataset

[Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) — 25,000 images across 6 natural scene categories.

| Class | Description |
|-------|-------------|
| Buildings | Urban structures |
| Forest | Dense woodland |
| Glacier | Ice and snow landscapes |
| Mountain | Mountain terrain |
| Sea | Ocean and coastlines |
| Street | Urban roads and paths |

---

## Results

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Buildings | 0.83 | 0.86 | 0.84 |
| Forest | 0.95 | 0.97 | 0.96 |
| Glacier | 0.87 | 0.81 | 0.84 |
| Mountain | 0.84 | 0.80 | 0.82 |
| Sea | 0.85 | 0.91 | 0.88 |
| Street | 0.88 | 0.89 | 0.89 |
| **Overall** | **0.87** | **0.87** | **0.87** |

**Test Accuracy: 87.00%** on 3,000 unseen images.

---

## Architecture

A 3-block CNN with a 4-layer fully connected classifier.

```
Input (3 × 150 × 150)
│
├── Conv Block 1: Conv2d(3→64, k=3) → BatchNorm → ReLU → MaxPool
├── Conv Block 2: Conv2d(64→64, k=3) → BatchNorm → ReLU → MaxPool
├── Conv Block 3: Conv2d(64→128, k=3) → BatchNorm → ReLU → MaxPool
│
├── Flatten → (128 × 17 × 17)
│
├── FC1: 36992 → 128 → Dropout(0.2)
├── FC2: 128 → 64  → Dropout(0.2)
├── FC3: 64  → 32  → Dropout(0.2)
└── FC4: 32  → 6   (output logits)
```

**Key design decisions:**
- BatchNorm after every conv layer — stabilizes training and acts as a regularizer in the conv blocks
- Dropout (p=0.2) after every FC layer except the output — prevents co-dependency in the classifier
- Channel progression 64 → 64 → 128 — increases representational capacity at deeper layers

---

## Training

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | SGD |
| Learning Rate | 0.0005 |
| Momentum | 0.9 |
| Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Batch Size | 32 |
| Epochs | 80 |
| Image Size | 150 × 150 |

**Data split:** 80% train / 20% validation from the training set, with separate transforms applied correctly to each split.

**Training augmentation:**
- Random horizontal flip
- Random rotation (±30°)
- Color jitter (brightness and contrast ±0.2)

**Validation/Test:** No augmentation — resize, normalize only.

**Checkpointing:** Best model saved whenever validation loss improves.

---

## Progression

This model was improved iteratively from a baseline:

| Version | Changes | Test Accuracy |
|---------|---------|---------------|
| Baseline | 2 conv blocks, no regularization | 81.27% |
| v2 | 3rd conv block, BatchNorm, Dropout, validation loop | 84.97% |
| v3 | Data augmentation, proper val/train transform split | **87.00%** |

---

## Requirements

```
torch
torchvision
scikit-learn
```

---

## Usage

1. Download the [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) from Kaggle.

2. Update the paths in the configuration section:
```python
TRAIN_DIR = "data/seg_train/seg_train"
TEST_DIR  = "data/seg_test/seg_test"
```

3. Run:
```bash
python intel_scene_classification.py
```

The best model checkpoint will be saved as `best_model.pth`. Final accuracy and per-class metrics will be printed after training.
