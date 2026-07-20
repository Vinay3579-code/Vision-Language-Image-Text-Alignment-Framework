# Utilities for image-text retrieval.

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def cosine_similarity(
    query: torch.Tensor,
    database: torch.Tensor,
) -> torch.Tensor:

    # Compute cosine similarity between a query embedding and a database of embeddings.
    query = F.normalize(query, dim=-1)
    database = F.normalize(database, dim=-1)

    return query @ database.T


def rank_embeddings(
    query: torch.Tensor,
    database: torch.Tensor,
):

    # Rank database embeddings by cosine similarity.
    similarity = cosine_similarity(
        query,
        database,
    )

    scores, indices = torch.sort(
        similarity.squeeze(0),
        descending=True,
    )

    return scores, indices


def top_k(
    query: torch.Tensor,
    database: torch.Tensor,
    k: int = 5,
):

    # Return the Top-K nearest neighbours.
    scores, indices = rank_embeddings(
        query,
        database,
    )

    return (
        scores[:k],
        indices[:k],
    )


def retrieve(
    query_embedding: torch.Tensor,
    database_embeddings: torch.Tensor,
    metadata: Sequence,
    k: int = 5,
):
    """
    Retrieve the Top-K most similar items.

    Parameters
    ----------
    query_embedding
        Query feature.

    database_embeddings
        Database feature matrix.

    metadata
        List containing information associated with each
        database embedding (image path, caption, etc.).

    k
        Number of retrieved items.

    Returns
    -------
    list[dict]
    """

    scores, indices = top_k(
        query_embedding,
        database_embeddings,
        k,
    )

    results = []

    for score, index in zip(
        scores.tolist(),
        indices.tolist(),
    ):

        results.append(
            {
                "index": index,
                "score": float(score),
                "metadata": metadata[index],
            }
        )

    return results


def best_match(
    query_embedding: torch.Tensor,
    database_embeddings: torch.Tensor,
    metadata: Sequence,
):

    # Return only the highest-ranked result.
    return retrieve(
        query_embedding,
        database_embeddings,
        metadata,
        k=1,
    )[0]


def similarity_score(
    embedding_a: torch.Tensor,
    embedding_b: torch.Tensor,
) -> float:

    # Compute cosine similarity between two embeddings.
    embedding_a = F.normalize(
        embedding_a,
        dim=-1,
    )

    embedding_b = F.normalize(
        embedding_b,
        dim=-1,
    )

    score = (
        embedding_a @ embedding_b.T
    ).item()

    return float(score)
