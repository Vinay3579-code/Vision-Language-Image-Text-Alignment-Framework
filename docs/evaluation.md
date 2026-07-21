# Evaluation

## Overview

This project evaluates the quality of image-text alignment rather than downstream retrieval or classification performance. The objective is to analyze how effectively the trained Projection Head aligns image embeddings with their corresponding text embeddings while preserving the semantic representations learned by the frozen EVA02-CLIP vision encoder and CLIP text encoder.

Unlike traditional Vision-Language Models that are evaluated using retrieval accuracy or classification metrics, this framework focuses exclusively on **embedding-level semantic alignment**.

---

# Evaluation Pipeline

The evaluation process follows the same inference pipeline used during training, except that no parameters are updated.

```
Input Image
     │
     ▼
Frozen EVA02-CLIP Vision Encoder
     │
768-D Image Embedding
     │
     ▼
Projection Head
     │
Projected Image Embedding
     │
     ▼
L2 Normalization
     │
     ▼
Cosine Similarity
     ▲
     │
L2 Normalized Text Embedding
     ▲
     │
Frozen CLIP Text Encoder
     ▲
     │
Input Caption
```

Only forward propagation is performed during evaluation.

---

# Running Evaluation

The repository provides two evaluation stages.

## Phase 2 Evaluation

Run

```bash
python training/phase2.py
```

or

```bash
bash scripts/evaluate.sh
```

Phase 2 evaluates the Projection Head immediately after alignment training.

---

## Phase 3 Evaluation

Run

```bash
python training/phase3.py
```

Phase 3 performs the refined alignment evaluation and reports the final similarity metrics.

---

# Evaluation Metrics

The framework evaluates semantic alignment using cosine similarity between projected image embeddings and text embeddings.

Four primary metrics are reported.

---

## 1. Positive Image–Text Similarity

Positive similarity measures the average cosine similarity between correctly matched image-text pairs.

Higher values indicate stronger semantic alignment.

```
Image
↓

Projection

↓

Cosine Similarity

↓

Correct Caption
```

Desired behaviour

```
High
```

---

## 2. Negative Image–Text Similarity

Negative similarity measures the cosine similarity between an image and unrelated captions.

```
Image

↓

Projection

↓

Cosine Similarity

↓

Incorrect Caption
```

Desired behaviour

```
Low
```

A lower value indicates better separation between unrelated image-text pairs.

---

## 3. Alignment Gap

The Alignment Gap is defined as

```
Positive Similarity

−

Negative Similarity
```

It measures how well the model separates matched and mismatched image-text pairs.

Large values indicate

- stronger semantic discrimination
- better embedding geometry
- improved alignment quality

This is the primary metric used throughout the paper to evaluate semantic discrimination.

---

## 4. Positive Similarity Standard Deviation

The standard deviation of positive similarities measures the consistency of alignment across the evaluation dataset.

Lower values indicate

- more stable embeddings
- consistent semantic grounding
- reliable alignment

---

# Cosine Similarity

Similarity between projected image embeddings and text embeddings is computed using cosine similarity.

```
Similarity

=

cos( ImageEmbedding , TextEmbedding )
```

Cosine similarity measures angular similarity instead of Euclidean distance.

The values lie within

```
[-1, 1]
```

where

| Value | Interpretation |
|--------|----------------|
| 1.0 | Perfect alignment |
| 0.0 | No semantic relationship |
| -1.0 | Opposite direction |

---

# Quantitative Evaluation

The paper evaluates four different alignment configurations.

| Model | Description |
|--------|-------------|
| Untrained Projector | Random Projection Head |
| NMSE Only | Regression alignment |
| InfoNCE Only | Contrastive alignment |
| Hybrid Alignment | NMSE + InfoNCE |

The Hybrid Alignment strategy demonstrates the strongest semantic discrimination despite producing slightly lower absolute cosine similarity, highlighting the importance of relative embedding geometry.

---

# Alignment Performance

The framework compares four alignment variants.

| Model | Matched Cosine | Mismatched Cosine | Alignment Gap |
|--------|---------------:|------------------:|--------------:|
| Zero-shot CLIP | 0.42 | 0.31 | 0.11 |
| Raw CLIP (EVA02-L-14) | 0.42 | 0.18 | 0.24 |
| Phase-2 Image Projector | 0.47 | 0.16 | 0.31 |
| Phase-3 Refined Alignment | 0.50 | 0.15 | 0.35 |

These results demonstrate that progressively training the Projection Head improves semantic discrimination while keeping both pretrained encoders frozen.

---

# Training Stability

Training stability is analyzed by monitoring

- Hybrid Loss
- NMSE Loss
- InfoNCE Loss

The paper reports that the Hybrid Alignment Loss converges more smoothly than using either NMSE or InfoNCE independently.

The combined objective reduces oscillations during optimization and provides more stable convergence.

---

# Qualitative Evaluation

In addition to numerical evaluation, qualitative analysis is performed.

For each query image,

the framework compares

```
Image

↓

Correct Caption

↓

Similarity
```

against

```
Image

↓

Incorrect Caption

↓

Similarity
```

Correct image-caption pairs consistently receive higher cosine similarity than mismatched pairs, demonstrating effective semantic grounding.

---

# Output Files

Evaluation produces

```
results/

├── phase2_metrics.json
├── phase3_metrics.json
├── similarity_scores.csv
└── evaluation.log
```

Depending on the implementation, additional visualizations or logs may also be generated.

---

# Console Output

Typical evaluation output

```
==================================

Evaluation Results

==================================

Positive Similarity : 0.50

Negative Similarity : 0.15

Alignment Gap : 0.35

Positive Std : 0.04

==================================
```

---

# Interpreting Results

## Good Alignment

```
Positive Similarity

High
```

```
Negative Similarity

Low
```

```
Alignment Gap

Large
```

---

## Poor Alignment

```
Positive Similarity

Low
```

```
Negative Similarity

High
```

```
Alignment Gap

Small
```

---

# Computational Efficiency

The framework updates fewer than **2.4 million trainable parameters** while both foundation encoders remain frozen.

Compared to conventional end-to-end multimodal fine-tuning, this approach

- reduces GPU memory consumption,
- accelerates training,
- improves optimization stability,
- preserves pretrained semantic knowledge.

---

# Best Practices

For reliable evaluation,

- use the same preprocessing pipeline used during training,
- normalize all embeddings before similarity computation,
- keep both encoders frozen,
- evaluate using identical tokenizer settings,
- compare both matched and mismatched pairs.

---

# Summary

The proposed evaluation framework assesses embedding-level image-text alignment using cosine similarity between projected image embeddings and frozen text embeddings. Performance is measured using Positive Similarity, Negative Similarity, Alignment Gap, and Positive Similarity Standard Deviation.

Experimental results show that the Hybrid Alignment Loss improves semantic discrimination while training fewer than **2.4 million** parameters, demonstrating that effective vision-language alignment can be achieved without end-to-end fine-tuning. This lightweight evaluation pipeline provides a reliable and reproducible way to analyze semantic alignment in fashion vision-language models.

For additional implementation details and mathematical derivations, refer to the accompanying Springer paper.
