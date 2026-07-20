# General utility functions used across the project.

from __future__ import annotations

import time
from pathlib import Path
from typing import Union

import torch
from PIL import Image


def get_device() -> torch.device:

    # Return the available compute device.

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def load_image(
    image_path: Union[str, Path],
) -> Image.Image:

    # Load an image from disk.

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    return Image.open(image_path).convert("RGB")


def count_parameters(
    model: torch.nn.Module,
    trainable_only: bool = False,
) -> int:

    # Count the number of parameters in a model.

    if trainable_only:
        return sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

    return sum(
        p.numel()
        for p in model.parameters()
    )


def format_parameters(
    num_parameters: int,
) -> str:

    # Format parameter counts into a human-readable string.

    if num_parameters >= 1_000_000_000:
        return f"{num_parameters / 1e9:.2f}B"

    if num_parameters >= 1_000_000:
        return f"{num_parameters / 1e6:.2f}M"

    if num_parameters >= 1_000:
        return f"{num_parameters / 1e3:.2f}K"

    return str(num_parameters)


class Timer:

    # Simple timer for measuring execution time.

    def __init__(self):

        self.start_time = None

    def start(self):

        self.start_time = time.perf_counter()

    def stop(self) -> float:

        if self.start_time is None:
            raise RuntimeError(
                "Timer has not been started."
            )

        elapsed = (
            time.perf_counter()
            - self.start_time
        )

        self.start_time = None

        return elapsed


def print_header(
    title: str,
) -> None:

    # Print a section header.

    line = "=" * len(title)

    print(f"\n{line}")
    print(title)
    print(line)


def move_to_device(
    batch,
    device: torch.device,
):

    # Move tensors in a batch to the specified device.

    if isinstance(batch, torch.Tensor):
        return batch.to(device)

    if isinstance(batch, dict):
        return {
            key: move_to_device(value, device)
            for key, value in batch.items()
        }

    if isinstance(batch, (list, tuple)):
        return type(batch)(
            move_to_device(item, device)
            for item in batch
        )

    return batch
