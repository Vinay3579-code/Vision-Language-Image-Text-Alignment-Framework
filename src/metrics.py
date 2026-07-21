"""
Evaluation metrics for the Image-Text Alignment Framework.

The proposed framework evaluates the quality of the learned shared
embedding space using cosine similarity statistics rather than
retrieval metrics.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def cosine_similarity_matrix(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the cosine similarity matrix between image and
    text embeddings.
    """

    image_embeddings = F.normalize(image_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)

    return image_embeddings @ text_embeddings.T


def mean_positive_similarity(
    similarity_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the mean cosine similarity of matching
    image-text pairs.
    """

    return similarity_matrix.diag().mean()


def mean_negative_similarity(
    similarity_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the mean cosine similarity of all
    non-matching image-text pairs.
    """

    num_samples = similarity_matrix.size(0)

    mask = ~torch.eye(
        num_samples,
        dtype=torch.bool,
        device=similarity_matrix.device,
    )

    return similarity_matrix[mask].mean()


def alignment_gap(
    positive_similarity: torch.Tensor,
    negative_similarity: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the Alignment Gap.

    Alignment Gap =
        Mean Positive Similarity
        -
        Mean Negative Similarity
    """

    return positive_similarity - negative_similarity


def similarity_std(
    similarity_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the standard deviation of the positive
    cosine similarities.
    """

    return similarity_matrix.diag().std(unbiased=False)


def evaluate_alignment(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
) -> Dict[str, float]:
    """
    Evaluate the alignment quality of the shared
    image-text embedding space.
    """

    similarity = cosine_similarity_matrix(
        image_embeddings,
        text_embeddings,
    )

    positive = mean_positive_similarity(similarity)

    negative = mean_negative_similarity(similarity)

    gap = alignment_gap(
        positive,
        negative,
    )

    std = similarity_std(similarity)

    return {
        "Mean Positive Similarity": positive.item(),
        "Mean Negative Similarity": negative.item(),
        "Alignment Gap": gap.item(),
        "Standard Deviation": std.item(),
    }


def print_metrics(
    metrics: Dict[str, float],
) -> None:

    # Print the alignment evaluation metrics.

    print("\nAlignment Evaluation")
    print("-" * 40)

    print(
        f"Mean Positive Similarity : "
        f"{metrics['Mean Positive Similarity']:.4f}"
    )

    print(
        f"Mean Negative Similarity : "
        f"{metrics['Mean Negative Similarity']:.4f}"
    )

    print(
        f"Alignment Gap            : "
        f"{metrics['Alignment Gap']:.4f}"
    )

    print(
        f""Positive Similarity Standard Deviation"       : "
        f"{metrics['"Positive Similarity Standard Deviation"']:.4f}"
    )
