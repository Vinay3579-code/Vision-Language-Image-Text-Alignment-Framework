"""
Phase 3 Refined Alignment Evaluation.

This stage performs the final alignment evaluation using the
trained Projection Head. The reported metrics correspond to
the refined alignment model presented in the paper.

Metrics:
    1. Mean Positive Similarity
    2. Mean Negative Similarity
    3. Alignment Gap
    4. Positive Similarity Standard Deviation
"""

from __future__ import annotations

import json
from pathlib import Path

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


OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


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


    # Final evaluation

    metrics = evaluate_alignment(
        projected_embeddings,
        text_embeddings,
    )

    print()
    print("=" * 60)
    print("PHASE 3 REFINED ALIGNMENT")
    print("=" * 60)

    print_metrics(metrics)

    print("=" * 60)


    # Save results

    output_file = OUTPUT_DIR / "phase3_metrics.json"

    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
