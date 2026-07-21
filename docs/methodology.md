# Methodology

## Overview

This project presents a **parameter-efficient image–text alignment framework** for fashion data using frozen vision-language encoders. Instead of fine-tuning the complete Vision-Language Model (VLM), the proposed framework keeps both the vision and text encoders frozen and trains only a lightweight image projection module.

The objective is to align image and text embeddings in a shared semantic space while preserving the pretrained knowledge of the foundation models. This approach significantly reduces computational cost, avoids representation drift, and enables controlled analysis of multimodal semantic alignment.

Unlike conventional multimodal architectures, this framework does **not** employ:

- Cross-attention
- Feature fusion
- Multimodal transformers
- End-to-end fine-tuning

Instead, alignment is achieved entirely through metric learning in the embedding space.

---

# Framework Architecture

The proposed framework consists of three primary components.

```
                Input Image
                     │
                     ▼
        Frozen EVA02-CLIP Vision Encoder
                     │
             768-D Image Embedding
                     │
                     ▼
          Trainable Projection Head
                     │
       Projected Image Embedding
                     │
                     │
                     ▼
             Cosine Similarity
                     ▲
                     │
        Frozen CLIP Text Encoder
                     ▲
                     │
               Input Caption
```

Only the **Projection Head** is trainable.

The EVA02-CLIP Vision Encoder and CLIP Text Encoder remain frozen throughout training.

---

# Component 1: Frozen EVA02-CLIP Vision Encoder

The visual backbone is the pretrained **EVA02-CLIP Vision Transformer**.

Its responsibilities include:

- Extracting semantic image representations
- Preserving pretrained visual knowledge
- Producing fixed-dimensional embeddings

For an input image

```
I
```

the encoder generates

```
v = fvision(I)
```

where

```
v ∈ R^768
```

The vision encoder is **never updated** during training.

Keeping the encoder frozen provides several advantages:

- Lower computational cost
- Stable feature representations
- No catastrophic forgetting
- Faster convergence
- Better reproducibility

---

# Component 2: Frozen CLIP Text Encoder

Each textual description is encoded using the pretrained CLIP text encoder.

For an input caption

```
T
```

the encoder produces

```
t = ftext(T)
```

where

```
t ∈ R^768
```

The text encoder acts as a **semantic anchor** throughout the training process.

Because it is frozen,

- semantic representations remain stable,
- target embeddings never drift,
- optimization becomes significantly easier.

---

# Component 3: Projection Head

Instead of modifying the vision encoder, a lightweight projection module learns to map image embeddings closer to their corresponding text embeddings.

The projection performs

```
z = Wv
```

where

- W is a learnable projection matrix
- only W is optimized during training

The projection preserves the embedding dimensionality.

```
768 → 768
```

After projection,

both embeddings are normalized.

```
ẑ = z / ||z||

t̂ = t / ||t||
```

Normalization removes magnitude information and ensures that alignment depends solely on angular similarity.

---

# Why Only Train the Projection Head?

Training the entire vision-language model requires hundreds of millions of trainable parameters.

The proposed framework updates fewer than **2.4 million parameters**, providing:

- reduced GPU memory usage
- faster training
- improved stability
- preservation of pretrained knowledge

This makes the framework practical for academic research and resource-constrained environments.

---

# Alignment Objectives

The framework employs a hybrid loss function composed of two complementary objectives.

## 1. Normalized Mean Squared Error (NMSE)

NMSE minimizes the Euclidean distance between normalized image and text embeddings.

```
LNMSE = ||ẑ − t̂||²
```

This objective encourages

- absolute correspondence
- stable optimization
- consistent embedding alignment

---

## 2. InfoNCE Loss

Contrastive learning is used to improve semantic discrimination.

Cosine similarity is computed as

```
sij = (ẑᵀ t̂) / τ
```

where

- τ is the temperature parameter.

The InfoNCE objective is

```
LInfoNCE
=
−log
exp(sii)
------------------------
Σ exp(sij)
```

This encourages

- matched image-text pairs to become closer,
- mismatched pairs to move farther apart.

---

# Hybrid Alignment Loss

The final optimization objective combines both losses.

```
L

=

λ LNMSE

+

(1−λ) LInfoNCE
```

where

- λ controls the contribution of each objective.

The hybrid objective combines

NMSE

- stable convergence
- numerical alignment

with

InfoNCE

- semantic discrimination
- improved embedding geometry.

---

# Training Pipeline

The complete training workflow is illustrated below.

```
Image
 │
 ▼
Frozen EVA02-CLIP
 │
 ▼
Image Embedding
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
Hybrid Loss
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

The optimization process consists of the following steps.

1. Load an image-caption pair.

2. Encode the image using the frozen EVA02-CLIP encoder.

3. Encode the caption using the frozen CLIP text encoder.

4. Project the image embedding.

5. Normalize both embeddings.

6. Compute NMSE Loss.

7. Compute InfoNCE Loss.

8. Compute Hybrid Loss.

9. Update only the Projection Head using AdamW.

10. Repeat for every minibatch.

---

# Training Algorithm

For every minibatch,

```
Image
      │
      ▼
Vision Encoder
      │
      ▼
Image Embedding

Caption
      │
      ▼
Text Encoder
      │
      ▼
Text Embedding

Projected Image Embedding
      │
      ▼
Normalization
      │
      ▼
Hybrid Loss
      │
      ▼
Backpropagation
      │
      ▼
Update Projection Head
```

No gradients are propagated into either encoder.

---

# Inference

Inference follows the same pipeline except that no parameters are updated.

The workflow is

1. Encode image.
2. Encode caption.
3. Project image embedding.
4. Normalize embeddings.
5. Compute cosine similarity.

The cosine similarity score indicates the semantic correspondence between the image and the caption.

Higher values indicate stronger alignment.

---

# Computational Advantages

Compared with end-to-end multimodal fine-tuning, the proposed framework offers several advantages.

| Aspect | Proposed Framework |
|---------|-------------------|
| Vision Encoder | Frozen |
| Text Encoder | Frozen |
| Trainable Parameters | 2.4 Million |
| Fine-tuning | Projection Head Only |
| Cross Attention | No |
| Feature Fusion | No |
| Computational Cost | Low |
| Memory Usage | Low |

---

# Key Design Principles

The framework is designed around the following principles.

- Preserve pretrained semantic knowledge.
- Train only lightweight alignment parameters.
- Perform alignment entirely in embedding space.
- Avoid representation drift.
- Reduce computational requirements.
- Improve semantic discrimination using hybrid metric learning.

---

# Summary

The proposed methodology demonstrates that effective vision-language alignment does not require end-to-end multimodal fine-tuning. By freezing both the EVA02-CLIP vision encoder and the CLIP text encoder while optimizing only a lightweight Projection Head with a hybrid NMSE and InfoNCE objective, the framework achieves efficient and robust image-text alignment for fashion data with a significantly reduced number of trainable parameters.

For mathematical derivations, architectural details, and experimental justification, please refer to the accompanying Springer paper.
