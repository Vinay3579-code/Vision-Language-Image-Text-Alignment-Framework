"""
Loss functions used for image-text alignment.

The framework supports individual losses for ablation experiments as well
as the hybrid objective used by the proposed model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):

    # Symmetric InfoNCE loss for image-text alignment.
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:

        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        logits = image_embeddings @ text_embeddings.T
        logits = logits / self.temperature

        targets = torch.arange(
            logits.size(0),
            device=logits.device,
        )

        loss_i2t = F.cross_entropy(logits, targets)
        loss_t2i = F.cross_entropy(logits.T, targets)

        return (loss_i2t + loss_t2i) * 0.5


class NormalizedMSELoss(nn.Module):

    # Mean squared error computed on normalized embeddings.
    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:

        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        return F.mse_loss(image_embeddings, text_embeddings)


class HybridAlignmentLoss(nn.Module):
    """
    Hybrid objective used in the proposed framework.

    Total Loss =
        InfoNCE +
        lambda × Normalized MSE
    """

    def __init__(
        self,
        temperature: float = 0.07,
        lambda_nmse: float = 0.10,
    ):
        super().__init__()

        self.infonce = InfoNCELoss(temperature)
        self.nmse = NormalizedMSELoss()

        self.lambda_nmse = lambda_nmse

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ):
        info_loss = self.infonce(
            image_embeddings,
            text_embeddings,
        )

        nmse_loss = self.nmse(
            image_embeddings,
            text_embeddings,
        )

        total_loss = info_loss + self.lambda_nmse * nmse_loss

        return {
            "loss": total_loss,
            "infonce": info_loss,
            "nmse": nmse_loss,
        }
