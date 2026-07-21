"""
Dataset definitions for image-text alignment.

Each sample consists of:

    • One fashion image
    • One corresponding text description

The dataset is used for training and evaluating the
Projection Head while keeping both EVA02-CLIP encoders
frozen, following the methodology described in the paper.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from PIL import Image

from torch.utils.data import Dataset


class FashionDataset(Dataset):

    """
    PyTorch Dataset for paired image-text samples.
    
    Expected CSV format:
    
    image,caption
    dress001.jpg,Red floral summer dress
    shirt123.jpg,Blue denim shirt
    
    Returns:
        {
            "image": Tensor,
            "caption": str
        }
    """

    def __init__(
        self,
        annotations_file: str | Path,
        image_root: str | Path,
        transform=None,
        image_column: str = "image",
        text_column: str = "caption",
    ):

        self.annotations = pd.read_csv(annotations_file)

        required_columns = {image_column, text_column}

        missing = required_columns - set(self.annotations.columns)
        
        if missing:
            raise ValueError(
                f"Missing required column(s): {sorted(missing)}"
            )
    
        self.image_root = Path(image_root)
    
        self.transform = transform
    
        self.image_column = image_column
        self.text_column = text_column

    def __len__(self) -> int:

        return len(self.annotations)

    def __getitem__(self, index: int):

        sample = self.annotations.iloc[index]

        image_path = self.image_root / sample[self.image_column]

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )
        
        image = Image.open(image_path).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
        
        return {
            "image": image,
            "caption": sample[self.text_column],
            "image_path": str(image_path),
        }
