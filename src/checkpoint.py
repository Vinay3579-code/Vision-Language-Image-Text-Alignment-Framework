# Utilities for saving and loading model checkpoints.

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    model,
    filepath: str | Path,
    epoch: int,
    optimizer=None,
    scheduler=None,
    loss: float | None = None,
    **metadata,
) -> None:
    """
    Save a training checkpoint.

    Parameters
    ----------
    model : nn.Module
        Model to save.

    filepath : str or Path
        Output checkpoint path.

    epoch : int
        Current training epoch.

    optimizer : Optimizer, optional
        Optimizer state.

    scheduler : Scheduler, optional
        Learning rate scheduler.

    loss : float, optional
        Training loss.

    metadata : dict
        Additional experiment information.
    """

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "loss": loss,
        "metadata": metadata,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, filepath)

    print(f"Checkpoint saved to: {filepath}")


def load_checkpoint(
    model,
    filepath: str | Path,
    optimizer=None,
    scheduler=None,
    map_location="cpu",
):
    """
    Load a training checkpoint.

    Returns
    -------
    dict
        Dictionary containing checkpoint metadata.
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(
        filepath,
        map_location=map_location,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if (
        optimizer is not None
        and "optimizer_state_dict" in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if (
        scheduler is not None
        and "scheduler_state_dict" in checkpoint
    ):
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    print(f"Checkpoint loaded from: {filepath}")

    return checkpoint


def load_projector(
    projector,
    filepath: str | Path,
    map_location="cpu",
) -> None:
    """
    Load the trained Projection Head weights.

    This utility restores only the trainable projection
    head, while the frozen EVA02-CLIP encoders are loaded
    independently from their pretrained checkpoints.
    """
    
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(
        filepath,
        map_location=map_location,
    )

    if "model_state_dict" in checkpoint:
        projector.load_state_dict(
            checkpoint["model_state_dict"],
            strict=False,
        )
    else:
        projector.load_state_dict(
            checkpoint,
            strict=False,
        )

    print(f"Projection Head loaded from: {filepath}")


def save_projector(
    projector,
    filepath: str | Path,
) -> None:
    """
    Save only the trained Projection Head.

    The vision and text encoders remain frozen and are
    restored directly from the pretrained EVA02-CLIP
    checkpoint during model initialization.
    """
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        projector.state_dict(),
        filepath,
    )

    print(f"Projection Head saved to: {filepath}")
