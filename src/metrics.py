# Evaluation metrics for image-text retrieval.

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def cosine_similarity_matrix(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
) -> torch.Tensor:

    # Compute the cosine similarity matrix between image and text embeddings.
    image_embeddings = F.normalize(image_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)

    return image_embeddings @ text_embeddings.T


def compute_ranks(
    similarity_matrix: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:

    # Compute image-to-text and text-to-image retrieval ranks.
    similarity = similarity_matrix.cpu().numpy()

    num_samples = similarity.shape[0]

    i2t_ranks = np.empty(num_samples, dtype=np.int32)
    t2i_ranks = np.empty(num_samples, dtype=np.int32)

    for i in range(num_samples):
        ranking = np.argsort(-similarity[i])
        i2t_ranks[i] = np.where(ranking == i)[0][0] + 1

    for i in range(num_samples):
        ranking = np.argsort(-similarity[:, i])
        t2i_ranks[i] = np.where(ranking == i)[0][0] + 1

    return i2t_ranks, t2i_ranks


def recall_at_k(
    ranks: np.ndarray,
    k: int,
) -> float:

    # Compute Recall@K.
    return float(np.mean(ranks <= k) * 100)


def mean_rank(
    ranks: np.ndarray,
) -> float:

    # Compute Mean Rank.
    return float(np.mean(ranks))


def median_rank(
    ranks: np.ndarray,
) -> float:

    # Compute Median Rank.
    return float(np.median(ranks))


def mean_reciprocal_rank(
    ranks: np.ndarray,
) -> float:

    # Compute Mean Reciprocal Rank (MRR).
    return float(np.mean(1.0 / ranks))


def evaluate_retrieval(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
) -> dict:

    # Evaluate retrieval performance for a single run.
    similarity = cosine_similarity_matrix(
        image_embeddings,
        text_embeddings,
    )

    i2t_ranks, t2i_ranks = compute_ranks(similarity)

    return {

        "I2T": {
            "R@1": recall_at_k(i2t_ranks, 1),
            "R@5": recall_at_k(i2t_ranks, 5),
            "R@10": recall_at_k(i2t_ranks, 10),
            "Mean Rank": mean_rank(i2t_ranks),
            "Median Rank": median_rank(i2t_ranks),
            "MRR": mean_reciprocal_rank(i2t_ranks),
        },

        "T2I": {
            "R@1": recall_at_k(t2i_ranks, 1),
            "R@5": recall_at_k(t2i_ranks, 5),
            "R@10": recall_at_k(t2i_ranks, 10),
            "Mean Rank": mean_rank(t2i_ranks),
            "Median Rank": median_rank(t2i_ranks),
            "MRR": mean_reciprocal_rank(t2i_ranks),
        },
    }


def print_metrics(
    metrics: dict,
) -> None:
  
    # Print retrieval metrics.
    for task in ("I2T", "T2I"):

        print(f"\n{task} Retrieval")
        print("-" * 30)

        print(f"Recall@1     : {metrics[task]['R@1']:.2f}")
        print(f"Recall@5     : {metrics[task]['R@5']:.2f}")
        print(f"Recall@10    : {metrics[task]['R@10']:.2f}")
        print(f"Mean Rank    : {metrics[task]['Mean Rank']:.2f}")
        print(f"Median Rank  : {metrics[task]['Median Rank']:.2f}")
        print(f"MRR          : {metrics[task]['MRR']:.4f}")
