# Training Guide

## Overview

The proposed framework performs parameter-efficient image-text alignment by training only a lightweight Projection Head while keeping both the EVA02-CLIP vision encoder and the CLIP text encoder completely frozen.

Unlike conventional Vision-Language Models, this framework does **not** perform end-to-end fine-tuning. Instead, the pretrained encoders act as fixed feature extractors, while the Projection Head learns to align image embeddings with the corresponding text embeddings.

---

# Training Pipeline

The overall training process consists of the following stages:

```
Dataset
   │
   ▼
Load Image–Caption Pair
   │
   ▼
Frozen EVA02-CLIP Vision Encoder
   │
   ▼
768-D Image Embedding
   │
   ▼
Projection Head
   │
   ▼
Projected Image Embedding
   │
   ▼
Normalization
   │
   ▼
Hybrid Alignment Loss
   ▲
   │
Normalized Text Embedding
   ▲
   │
Frozen CLIP Text Encoder
   ▲
   │
Caption
```

Only the Projection Head receives gradient updates during optimization.

---

# Dataset Preparation

The framework is designed to work with paired image-text datasets.

Each training sample consists of:

- Image
- Caption

The experiments presented in the paper use the **Fashion Product Text-Images Dataset**.

Expected directory structure:

```
datasets/
│
├── images/
│   ├── 0001.jpg
│   ├── 0002.jpg
│   ├── ...
│
└── metadata.csv
```

---

# Metadata Format

The CSV file should contain at least two columns.

| Column | Description |
|---------|-------------|
| image | Image filename |
| caption | Corresponding textual description |

Example:

| image | caption |
|---------|---------|
| shirt001.jpg | Blue cotton formal shirt |
| dress004.jpg | Floral summer dress |
| shoes021.jpg | Black leather shoes |

---

# Image Preprocessing

All images undergo the preprocessing pipeline provided by the pretrained EVA02-CLIP model.

The preprocessing includes:

- Resize
- Center Crop
- RGB Conversion
- Tensor Conversion
- CLIP Normalization

No additional data augmentation is applied during training to preserve semantic consistency between images and text.

---

# Text Preprocessing

Text descriptions are processed using the CLIP tokenizer.

The textual fields may include:

- Product title
- Description
- Category

These fields can be concatenated into a single caption before tokenization.

Example:

```
Blue Cotton Shirt
+
Men's Formal Wear
+
Cotton Long Sleeve
```

↓

```
Blue Cotton Shirt Men's Formal Wear Cotton Long Sleeve
```

---

# Model Components

The complete model consists of three modules.

## Frozen Vision Encoder

```
EVA02-CLIP Vision Transformer
```

Status:

```
Frozen
```

Trainable parameters:

```
0
```

---

## Frozen Text Encoder

```
CLIP Text Encoder
```

Status:

```
Frozen
```

Trainable parameters:

```
0
```

---

## Projection Head

```
768
 ↓

Linear

 ↓

Layer Normalization

 ↓

GELU

 ↓

Linear

 ↓

768
```

Status:

```
Trainable
```

This is the **only** component optimized during training.

---

# Loss Function

Training minimizes the Hybrid Alignment Loss.

```
Hybrid Loss

=

λ × NMSE

+

(1−λ) × InfoNCE
```

where

- NMSE improves absolute alignment.
- InfoNCE improves semantic discrimination.

---

# Optimizer

The Projection Head is optimized using

```
AdamW
```

Typical configuration:

```
Optimizer : AdamW
Weight Decay : Enabled
Mixed Precision : Supported
```

Only Projection Head parameters are passed to the optimizer.

---

# Training Loop

Each training iteration performs the following operations.

### Step 1

Load a minibatch.

```
(Image, Caption)
```

---

### Step 2

Extract frozen image embeddings.

```
Image

↓

EVA02-CLIP

↓

768-D Embedding
```

---

### Step 3

Extract frozen text embeddings.

```
Caption

↓

CLIP Text Encoder

↓

768-D Embedding
```

---

### Step 4

Project image embeddings.

```
Projection Head

↓

Projected Image Embedding
```

---

### Step 5

Normalize embeddings.

```
Image

↓

L2 Normalize

Text

↓

L2 Normalize
```

---

### Step 6

Compute

```
NMSE Loss
```

---

### Step 7

Compute

```
InfoNCE Loss
```

---

### Step 8

Compute

```
Hybrid Loss
```

---

### Step 9

Backpropagate.

Only the Projection Head receives gradients.

---

### Step 10

Update parameters using AdamW.

---

# Running Training

To start training,

```
python training/phase1.py
```

or

```
bash scripts/train.sh
```

Training automatically:

- Loads the dataset
- Builds the model
- Computes Hybrid Loss
- Updates the Projection Head
- Saves checkpoints

---

# Checkpoints

The trained Projection Head is saved under

```
checkpoints/
```

Example:

```
checkpoints/

projector_phase1.pt
```

Only the Projection Head weights are stored because the pretrained encoders remain unchanged.

---

# Monitoring Training

During training the console reports

- Training Loss
- NMSE Loss
- InfoNCE Loss
- Positive Similarity
- Negative Similarity
- Alignment Gap

Example

```
Epoch 4

Hybrid Loss : 0.271

NMSE : 0.081

InfoNCE : 0.462

Positive Similarity : 0.48

Negative Similarity : 0.17

Alignment Gap : 0.31
```

---

# GPU Requirements

The framework is designed for efficient training.

Typical requirements:

| Component | Requirement |
|-----------|-------------|
| Python | 3.10+ |
| PyTorch | 2.x |
| CUDA | Optional |
| GPU Memory | 8 GB or higher |

Because only the Projection Head is trained, GPU memory consumption is significantly lower than end-to-end multimodal fine-tuning.

---

# Reproducibility

For reproducible experiments:

- Fix random seeds.
- Keep both encoders frozen.
- Use the same preprocessing pipeline.
- Use identical tokenizer settings.
- Preserve the original image-caption pairs.
- Train only the Projection Head.

---

# Troubleshooting

## CUDA Out of Memory

Reduce

```
Batch Size
```

or enable

```
Mixed Precision Training
```

---

## Slow Training

Verify that:

- Vision encoder is frozen.
- Text encoder is frozen.
- Only Projection Head parameters are passed to AdamW.

---

## Poor Alignment

Check:

- Dataset quality.
- Image-caption correspondence.
- Tokenization.
- Projection Head checkpoint.
- Loss balancing parameter (λ).

---

# Best Practices

- Do not fine-tune the frozen encoders.
- Preserve CLIP preprocessing.
- Use normalized embeddings.
- Train only the Projection Head.
- Monitor both similarity scores and Hybrid Loss during training.

---

# Summary

The proposed training strategy focuses on efficient image-text alignment by optimizing only a lightweight Projection Head while leveraging the pretrained semantic knowledge of frozen EVA02-CLIP and CLIP encoders. This design enables stable optimization, reduced computational cost, and strong multimodal alignment without requiring end-to-end fine-tuning.

For additional implementation details and mathematical derivations, refer to the accompanying Springer paper.
