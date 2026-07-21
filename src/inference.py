"""
Inference utilities for the Image-Text Alignment Framework.

This module loads the trained Projection Head together with the
frozen EVA02-CLIP image encoder and CLIP text encoder to perform
image-text similarity inference.

Supported functionality
-----------------------
1. Encode an image
2. Encode text
3. Compute cosine similarity
4. Match one image against one or more captions
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image

from configs.config import (
    DEVICE,
    CHECKPOINTS,
)

from src.model import ImageTextAlignmentModel
from src.checkpoint import load_projector


class ImageTextInference:

    # Image-Text Alignment inference engine.

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        device: str = DEVICE,
    ):

        self.device = torch.device(device)

        self.model = ImageTextAlignmentModel().to(self.device)

        checkpoint = checkpoint or CHECKPOINTS["phase1"]

        load_projector(
            projector=self.model.projector,
            filepath=checkpoint,
        )

        self.model.eval()

    @torch.inference_mode()
    def encode_image(
        self,
        image: str | Path | Image.Image,
    ) -> torch.Tensor:

        # Encode a single image into the shared embedding space.

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        image_tensor = self.model.image_encoder.preprocess(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        image_embedding = self.model.encode_image(image_tensor)

        projected_embedding = self.model.project(image_embedding)

        projected_embedding = F.normalize(
            projected_embedding,
            dim=-1,
        )

        return projected_embedding.squeeze(0)

    @torch.inference_mode()
    def encode_text(
        self,
        text: str,
    ) -> torch.Tensor:

        # Encode a single text description.

        embedding = self.model.encode_text([text])

        embedding = F.normalize(
            embedding,
            dim=-1,
        )

        return embedding.squeeze(0)

    @torch.inference_mode()
    def similarity(
        self,
        image: str | Path | Image.Image,
        text: str,
    ) -> float:

        # Compute cosine similarity between an image and text.

        image_embedding = self.encode_image(image)

        text_embedding = self.encode_text(text)

        similarity = F.cosine_similarity(
            image_embedding.unsqueeze(0),
            text_embedding.unsqueeze(0),
        )

        return float(similarity.item())

    @torch.inference_mode()
    def rank_captions(
        self,
        image: str | Path | Image.Image,
        captions: List[str],
    ):

        # Rank multiple captions for a single image.

        image_embedding = self.encode_image(image)

        results = []

        for caption in captions:

            text_embedding = self.encode_text(caption)

            score = F.cosine_similarity(
                image_embedding.unsqueeze(0),
                text_embedding.unsqueeze(0),
            ).item()

            results.append(
                {
                    "caption": caption,
                    "similarity": float(score),
                }
            )

        results.sort(
            key=lambda x: x["similarity"],
            reverse=True,
        )

        return results

    @torch.inference_mode()
    def predict(
        self,
        image: str | Path | Image.Image,
        captions: List[str],
    ):

        # Alias for rank_captions().

        return self.rank_captions(
            image=image,
            captions=captions,
        )


if __name__ == "__main__":

    engine = ImageTextInference()

    image = "sample.jpg"

    captions = [
        "Red floral summer dress",
        "Blue denim jacket",
        "Black leather handbag",
    ]

    predictions = engine.predict(
        image=image,
        captions=captions,
    )

    print("\nSimilarity Ranking")
    print("-" * 40)

    for result in predictions:

        print(
            f"{result['similarity']:.4f}  {result['caption']}"
        )
