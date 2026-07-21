# Frozen image and text encoders used by the Image-Text Alignment Framework.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


from configs.config import (
    DEVICE,
    VISION_MODEL,
    VISION_PRETRAINED,
)


class FrozenImageEncoder(nn.Module):
    """
    EVA02-CLIP visual encoder.

    The encoder remains frozen during training and produces
    normalized image embeddings.
    """

    def __init__(self):
        super().__init__()

        model, _, preprocess = open_clip.create_model_and_transforms(
            VISION_MODEL,
            pretrained=VISION_PRETRAINED,
        )

        self.encoder = model.visual
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(VISION_MODEL)

        self.encoder.eval()

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:

        embeddings = self.encoder(images)

        return F.normalize(embeddings, dim=-1)


class FrozenTextEncoder(nn.Module):
    class FrozenTextEncoder(nn.Module):
    """
    Frozen CLIP text encoder.

    The encoder remains frozen during training and produces
    normalized text embeddings in the shared image-text
    embedding space.
    """

    def __init__(self):
        super().__init__()

        model, _, _ = open_clip.create_model_and_transforms(
            VISION_MODEL,
            pretrained=VISION_PRETRAINED,
        )

        self.encoder = model
        self.tokenizer = open_clip.get_tokenizer(VISION_MODEL)

        self.encoder.eval()

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def forward(
        self,
        texts: list[str],
    ) -> torch.Tensor:

        tokens = self.tokenizer(texts)

        tokens = tokens.to(DEVICE)

        embeddings = self.encoder.encode_text(tokens)

        return F.normalize(embeddings, dim=-1)
