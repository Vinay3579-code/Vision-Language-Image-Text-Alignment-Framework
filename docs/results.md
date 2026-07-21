# Results

## Overview

This section summarizes the experimental results obtained using the proposed parameter-efficient image-text alignment framework. The evaluation demonstrates that training only a lightweight Projection Head while keeping both the EVA02-CLIP vision encoder and CLIP text encoder frozen produces effective semantic alignment with significantly fewer trainable parameters.

---

# Alignment Performance

The proposed framework was evaluated using cosine similarity between projected image embeddings and their corresponding text embeddings.

The following metrics were used throughout the experiments:

- Mean Positive Image–Text Similarity
- Mean Negative Image–Text Similarity
- Alignment Gap
- Positive Similarity Standard Deviation

The Hybrid Alignment strategy achieved the best overall semantic separation by increasing the similarity of matched image-text pairs while reducing the similarity of mismatched pairs. :contentReference[oaicite:1]{index=1}

---

# Model Comparison

The paper compares multiple alignment strategies.

| Method | Objective |
|---------|-----------|
| Zero-shot CLIP | Frozen pretrained baseline |
| Raw CLIP (EVA02-L-14) | Frozen encoder embeddings |
| NMSE Only | Regression-based alignment |
| InfoNCE Only | Contrastive alignment |
| Hybrid Alignment | NMSE + InfoNCE |

Among these approaches, the Hybrid Alignment framework provides the strongest semantic discrimination while maintaining a lightweight training pipeline. :contentReference[oaicite:2]{index=2}

---

# Key Observations

The experimental results highlight several important findings:

- Frozen pretrained encoders preserve rich semantic representations.
- Training only the Projection Head is sufficient to improve image-text alignment.
- Combining NMSE and InfoNCE produces better embedding separation than either objective alone.
- Hybrid optimization improves both alignment quality and training stability.

---

# Computational Efficiency

A major contribution of the framework is its parameter efficiency.

Compared with conventional end-to-end Vision-Language Model fine-tuning, the proposed approach:

- trains fewer than **2.4 million** parameters,
- reduces GPU memory requirements,
- shortens training time,
- preserves pretrained knowledge by freezing both encoders. :contentReference[oaicite:3]{index=3}

---

# Discussion

The experiments demonstrate that effective image-text alignment can be achieved without updating the underlying foundation models. By learning a lightweight projection module, the framework successfully maps image embeddings into the frozen text embedding space while maintaining computational efficiency.

These results suggest that parameter-efficient adaptation is a practical alternative to full model fine-tuning for domain-specific multimodal applications.

---

# Limitations

The current framework focuses on embedding-level alignment and does not perform:

- End-to-end multimodal fine-tuning
- Cross-attention between image and text
- Multimodal feature fusion
- Conversational reasoning or downstream task optimization

Future work can extend this framework toward multimodal retrieval, conversational assistants, and recommendation systems. :contentReference[oaicite:4]{index=4}

---

# Summary

The proposed Hybrid Alignment framework demonstrates that high-quality semantic image-text alignment can be achieved using frozen vision-language encoders and a lightweight trainable Projection Head. The experimental results validate the effectiveness of the hybrid optimization strategy while maintaining low computational cost and strong parameter efficiency, making the framework suitable for scalable multimodal learning applications.
