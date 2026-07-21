"""
Image-Text Alignment Framework.

This module integrates the frozen EVA02-CLIP vision encoder,
the frozen CLIP text encoder, and the trainable projection
head into a unified alignment model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import open_clip
from typing import Dict

from src.encoders import (
    FrozenImageEncoder,
    FrozenTextEncoder,
)

from src.projector import ProjectionHead

from configs.config import (
    VISION_MODEL,
    VISION_PRETRAINED,
    VISION_EMBED_DIM,
    PROJECTOR_HIDDEN_DIM,
)


class ImageTextAlignmentModel(nn.Module):
    """
    Image-Text Alignment Framework.

    Components
    ----------
    - Frozen EVA02-CLIP vision encoder
    - Frozen CLIP text encoder
    - Trainable projection head
    """

    def __init__(self):
        super().__init__()

        foundation_model, _, preprocess = open_clip.create_model_and_transforms(
            VISION_MODEL,
            pretrained=VISION_PRETRAINED,
        )
        
        self.image_encoder = FrozenImageEncoder(
            foundation_model.visual,
            preprocess,
        )
        
        self.text_encoder = FrozenTextEncoder(
            foundation_model,
        )

        self.projector = ProjectionHead(
            input_dim=VISION_EMBED_DIM,
            hidden_dim=PROJECTOR_HIDDEN_DIM,
            output_dim=VISION_EMBED_DIM,
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
    ) -> Dict[str, torch.Tensor]:

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
            for p in self.parameters()
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
        print(f"Vision Backbone : {VISION_MODEL}")
        print(f"Pretrained Weights : {VISION_PRETRAINED}")
        print(f"Projector     : {self.projector.__class__.__name__}")

        print()

        print(f"Trainable Parameters : {self.trainable_parameters:,}")
        print(f"Frozen Parameters    : {self.frozen_parameters:,}")
        print(f"Total Parameters     : {self.total_parameters:,}")
