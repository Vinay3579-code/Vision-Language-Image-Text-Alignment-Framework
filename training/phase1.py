"""
Phase 1 training.

This stage initializes the projection network using
Mean Squared Error (MSE) loss while keeping both
the image encoder and text encoder frozen.
"""

from __future__ import annotations

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from configs.config import (
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_WORKERS,
    GRADIENT_CLIP,
    SEED,
    CHECKPOINTS,
)

from src.seed import (
    set_seed,
    seed_worker,
    create_generator,
)

from src.model import ImageTextAlignmentModel
from src.losses import NormalizedMSELoss
from training.datasets import FashionDataset
from training.trainer import Trainer


def main():

    set_seed(SEED)

    model = ImageTextAlignmentModel()

    dataset = FashionDataset(
        annotations_file="datasets/train.csv",
        image_root="datasets/images",
        transform=model.image_encoder.preprocess,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        generator=create_generator(SEED),
    )

    optimizer = AdamW(
        model.projector.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    trainer = Trainer(
        model=model,
        loss_fn=NormalizedMSELoss(),
        optimizer=optimizer,
        device=model.projector.network[0].weight.device,
        checkpoint_path=CHECKPOINTS["phase1"],
        gradient_clip=GRADIENT_CLIP,
    )

    trainer.fit(
        dataloader,
        epochs=NUM_EPOCHS,
    )


if __name__ == "__main__":
    main()
