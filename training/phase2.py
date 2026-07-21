"""
Phase 2 Training.

This stage refines the Projection Head using the
InfoNCE loss while keeping both the EVA02-CLIP vision
encoder and text encoder frozen.

The Projection Head is initialized from the Phase 1
checkpoint trained with the Normalized MSE loss.
"""

from __future__ import annotations

from torch.optim import AdamW
from torch.utils.data import DataLoader

from configs.config import (
    BATCH_SIZE,
    PHASE2_EPOCHS,
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
from src.losses import InfoNCELoss
from src.checkpoint import load_projector
from training.datasets import FashionDataset
from training.trainer import Trainer


def main():

    set_seed(SEED)

    model = ImageTextAlignmentModel()


    # Initialize the Projection Head from Phase 1
  
    load_projector(
        projector=model.projector,
        filepath=CHECKPOINTS["phase1"],
    )


    # Dataset

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


    # Optimizer

    optimizer = AdamW(
        model.projector.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    # Trainer

    trainer = Trainer(
        model=model,
        loss_fn=InfoNCELoss(),
        optimizer=optimizer,
        device=model.projector.network[0].weight.device,
        checkpoint_path=CHECKPOINTS["phase2"],
        gradient_clip=GRADIENT_CLIP,
    )

  
    # Training

    trainer.fit(
        dataloader,
        epochs=PHASE2_EPOCHS,
    )


if __name__ == "__main__":
    main()
