"""
Projection module used for image-text alignment.

The visual encoder produces a fixed image embedding. This module learns
a lightweight mapping that projects the frozen image representation into
a space that aligns more closely with the frozen text embeddings.

Only this module is optimized during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VisualProjector(nn.Module):
    """
    Lightweight projection network for image embeddings.

    Architecture
    ------------
    Input (768)
        ↓
    Linear (768 → 1536)
        ↓
    ReLU
        ↓
    Linear (1536 → 768)
        ↓
    LayerNorm
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 1536,
        output_dim: int = 768,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialize linear layers using Xavier initialization.

        This generally provides stable convergence for shallow
        projection networks.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, image_embedding: torch.Tensor) -> torch.Tensor:
        """
        Project image embeddings into the shared embedding space.

        Parameters
        ----------
        image_embedding : torch.Tensor
            Tensor of shape (batch_size, input_dim)

        Returns
        -------
        torch.Tensor
            Projected embedding of shape (batch_size, output_dim)
        """
        return self.network(image_embedding)

    @property
    def num_parameters(self) -> int:

        # Returns the number of trainable parameters.
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"trainable_parameters={self.num_parameters:,})"
        )
