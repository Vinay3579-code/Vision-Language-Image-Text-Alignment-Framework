"""
Image-Text Alignment Framework.

This module combines the frozen image encoder, frozen text encoder,
and the trainable projection network into a single model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.encoders import (
    FrozenImageEncoder,
    FrozenTextEncoder,
)

from src.projector import VisualProjector

from configs.config import (
    IMAGE_EMBED_DIM,
    PROJECTOR_HIDDEN_DIM,
)


class ImageTextAlignmentModel(nn.Module):
    """
    Image-Text Alignment Framework.

    Components
    ----------
    - Frozen EVA02 image encoder
    - Frozen Phi-3.5 text encoder
    - Trainable projection network
    """

    def __init__(
        self,
        load_text_encoder_4bit: bool = True,
    ):
        super().__init__()

        self.image_encoder = FrozenImageEncoder()

        self.text_encoder = FrozenTextEncoder(
            load_in_4bit=load_text_encoder_4bit
        )

        self.projector = VisualProjector(
            input_dim=IMAGE_EMBED_DIM,
            hidden_dim=PROJECTOR_HIDDEN_DIM,
            output_dim=IMAGE_EMBED_DIM,
        )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:

        # Extract frozen image embeddings.
        return self.image_encoder(images)

    def encode_text(
        self,
        texts: list[str],
    ) -> torch.Tensor:

        # Extract frozen text embeddings.
        return self.text_encoder(texts)

    def project(
        self,
        image_embeddings: torch.Tensor,
    ) -> torch.Tensor:

        # Project image embeddings into the shared embedding space.
        return self.projector(image_embeddings)

    def forward(
        self,
        images: torch.Tensor,
        texts: list[str],
    ) -> dict:

        image_embeddings = self.encode_image(images)

        projected_embeddings = self.project(
            image_embeddings
        )

        text_embeddings = self.encode_text(texts)

        return {
            "image_embeddings": image_embeddings,
            "projected_embeddings": projected_embeddings,
            "text_embeddings": text_embeddings,
        }

    @property
    def trainable_parameters(self) -> int:
        return sum(
            p.numel()
            for p in self.projector.parameters()
            if p.requires_grad
        )

    @property
    def total_parameters(self) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
        )

    @property
    def frozen_parameters(self) -> int:
        return (
            self.total_parameters
            - self.trainable_parameters
        )

    def print_model_summary(self) -> None:

        print("\nImage-Text Alignment Framework")
        print("-" * 40)

        print(f"Image Encoder : {self.image_encoder.__class__.__name__}")
        print(f"Text Encoder  : {self.text_encoder.__class__.__name__}")
        print(f"Projector     : {self.projector.__class__.__name__}")

        print()

        print(f"Trainable Parameters : {self.trainable_parameters:,}")
        print(f"Frozen Parameters    : {self.frozen_parameters:,}")
        print(f"Total Parameters     : {self.total_parameters:,}")
