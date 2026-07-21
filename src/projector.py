# Projection network for aligning image embeddings with the text embedding space.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    Projection head that maps frozen image embeddings into the
    shared image-text embedding space.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 1536,
        output_dim: int = 768,
        activation: str = "relu",
        dropout: float = 0.0,
        normalize: bool = PROJECTOR_NORMALIZE,
    ):
        super().__init__()

        activation = activation.lower()

        activations = {
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }

        if activation not in activations:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Choose from {list(activations.keys())}."
            )

        layers = [
            nn.Linear(input_dim, hidden_dim),
            activations[activation],
        ]

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.extend([
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        ])

        self.network = nn.Sequential(*layers)

        self.normalize = normalize

        self._initialize_weights()

    def _initialize_weights(self) -> None:

        for module in self.modules():

            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        image_embeddings: torch.Tensor,
    ) -> torch.Tensor:

        projected = self.network(image_embeddings)
        
        if self.normalize:
            projected = F.normalize(
                projected,
                p=2,
                dim=-1,
            )
    
        return projected
    

    @property
    def num_parameters(self) -> int:

        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}"
            f"(parameters={self.num_parameters:,})"
        )
