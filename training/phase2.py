"""
Phase 2 Evaluation.

This stage evaluates the trained Projection Head by
measuring image-text alignment quality using cosine
similarity statistics.

The following metrics are reported:

1. Mean Positive Similarity
2. Mean Negative Similarity
3. Alignment Gap
4. Positive Similarity Standard Deviation
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from configs.config import (
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    CHECKPOINTS,
)

from src.seed import (
    set_seed,
    seed_worker,
    create_generator,
)

from src.model import ImageTextAlignmentModel
from src.metrics import (
    evaluate_alignment,
    print_metrics,
)
from src.checkpoint import load_projector

from training.datasets import FashionDataset


def main():


    # Reproducibility

    set_seed(SEED)

    
    # Load model

    model = ImageTextAlignmentModel()

    load_projector(
        projector=model.projector,
        filepath=CHECKPOINTS["phase1"],
    )

    model.eval()


    # Dataset

    dataset = FashionDataset(
        annotations_file="datasets/train.csv",
        image_root="datasets/images",
        transform=model.image_encoder.preprocess,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        generator=create_generator(SEED),
    )


    # Collect embeddings

    projected_embeddings = []
    text_embeddings = []

    with torch.inference_mode():

        for batch in dataloader:

            images = batch["image"].to(
                next(model.parameters()).device
            )

            captions = batch["caption"]

            outputs = model(
                images,
                captions,
            )

            projected_embeddings.append(
                outputs["projected_embeddings"]
            )

            text_embeddings.append(
                outputs["text_embeddings"]
            )

    projected_embeddings = torch.cat(
        projected_embeddings,
        dim=0,
    )

    text_embeddings = torch.cat(
        text_embeddings,
        dim=0,
    )


    # Evaluate alignment

    metrics = evaluate_alignment(
        projected_embeddings,
        text_embeddings,
    )

    print()
    print("=" * 60)
    print("PHASE 2 IMAGE PROJECTOR EVALUATION")
    print("=" * 60)

    print_metrics(metrics)

    print("=" * 60)


if __name__ == "__main__":
    main()
