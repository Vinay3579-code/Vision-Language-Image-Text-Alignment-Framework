# Image-Text Alignment Framework for Fashion Vision-Language Models

<p align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)]()
[![OpenCLIP](https://img.shields.io/badge/OpenCLIP-EVA02--L14-green.svg)]()

</p>

---

## Overview

This repository contains the official implementation of our **Image–Text Alignment Framework** for fashion vision-language models.

Unlike conventional multimodal architectures that rely on cross-attention, feature fusion, or end-to-end fine-tuning, the proposed framework performs **phase-wise metric learning** by training only a lightweight image projection module while keeping both the visual encoder and text encoder frozen.

The framework learns a shared embedding space where projected visual embeddings are aligned with frozen semantic text embeddings using metric learning objectives.

---

## Key Contributions

- Phase-wise Image–Text Alignment Framework
- Frozen EVA02-CLIP Vision Encoder
- Frozen CLIP Text Embedding Space
- Lightweight Trainable Projection Layer
- Metric Learning based Alignment
- Hybrid InfoNCE + Normalized MSE Optimization
- Positive Image-Text Similarity
- Negative Image-Text Similarity
- Alignment Gap
- Out-of-Domain (OOD) Validation
- Ablation Studies
- LoRA Baseline Comparison

---

# Framework

<p align="center">

<img src="figures/architecture.png" width="1000">

</p>

---

# Repository Structure

```text
Image-Text-Alignment-Framework/

│

├── configs/
│ └── config.py

│

├── src/
│ ├── model.py
│ ├── projector.py
│ ├── losses.py
│ ├── utils.py
│ ├── build_database.py
│ ├── retrieve.py
│ └── evaluate.py

│

├── training/
│ ├── phase1.py
│ ├── phase2.py
│ ├── phase2_hidden.py
│ ├── phase3_ac_v2.py
│ └── lora_train.py

│

├── evaluation/
│ ├── evaluate.py
│ ├── ablation.py
│ ├── statistics.py
│ └── retrieval_metrics.py

│

├── demo/
│ ├── demo.py
│ └── app.py

│

├── figures/

│

├── docs/

│

├── scripts/

│

├── datasets/

│

├── checkpoints/

│

├── README.md

├── requirements.txt

└── .gitignore
```

---

# Methodology

The proposed framework consists of four stages.

### Stage 1

Input Processing

- Fashion Images
- Fashion Captions

---

### Stage 2

Frozen Backbone

Visual Encoder

- EVA02-CLIP-L-14

Text Encoder

- CLIP

Both encoders remain frozen throughout alignment training.

---

### Stage 3

Image Projection Alignment

Only the image projection layer is trainable.

```
Image Embedding (768)

↓

FC Layer (768 → 1536)

↓

ReLU

↓

FC Layer (1536 → 768)

↓

Projected Image Embedding
```

The projected image embedding is aligned with frozen text embeddings using cosine similarity.

---

### Stage 4

Alignment Evaluation

The framework computes

- Positive Image-Text Similarity
- Negative Image-Text Similarity
- Alignment Gap
- Retrieval Metrics

---

# Training Strategy

## Phase 1

Projection Warm-up

Loss

- MSE

---

## Phase 2

Projection Refinement

Loss

- Regression Loss

---

## Phase 3

Hybrid Metric Learning

Loss

- InfoNCE
- Normalized MSE

---

# Loss Function

The proposed framework optimizes

\[
\mathcal{L}
=
\mathcal{L}_{InfoNCE}
+
\lambda
\mathcal{L}_{NMSE}
\]

where

- InfoNCE improves discriminative alignment.
- Normalized MSE preserves embedding consistency.

---

# Datasets

The framework was trained and evaluated using

- DeepFashion
- Polyvore
- Amazon Fashion
- FashionIQ

---

# Installation

Clone repository

```bash
git clone https://github.com/<username>/Image-Text-Alignment-Framework.git
```

Move into repository

```bash
cd Image-Text-Alignment-Framework
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Training

Phase 1

```bash
python training/phase1.py
```

Phase 2

```bash
python training/phase2.py
```

Phase 3

```bash
python training/phase3_ac_v2.py
```

---

# Evaluation

```bash
python evaluation/evaluate.py
```

---

# Demo

```bash
python demo/demo.py
```

---

# Experimental Results

The framework is evaluated using

- Cosine Similarity
- Alignment Gap
- Mean
- Standard Deviation

---

# Ablation Study

The following variants are compared

| Model | Projection | Loss |
|--------|------------|------|
| MSE Only | ✓ | MSE |
| Regression Only | ✓ | Regression |
| Hybrid (Ours) | ✓ | InfoNCE + NMSE |
| LoRA Baseline | LoRA | Contrastive |

---

# Qualitative Results

The repository includes

- Retrieval Examples
- Matched Image-Text Pairs
- Mismatched Image-Text Pairs
- OOD Validation Examples

---

# Future Work

- Multimodal Conversational Fashion Assistant
- User Preference Modeling
- Style Personalization
- Multi-turn Dialogue
- Large-scale Fashion Retrieval

---

# Citation

```bibtex

```



---

# Contact

For questions regarding this work, please open an issue in this repository.

---

## Acknowledgement

This work utilizes

- PyTorch
- Hugging Face Transformers
- EVA02-CLIP
- EVA02
