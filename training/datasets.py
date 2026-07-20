# Dataset definitions used for image-text alignment.

from __future__ import annotations

from pathlib import Path

import pandas as pd

from PIL import Image

from torch.utils.data import Dataset


class FashionDataset(Dataset):

    # Dataset for paired image-text samples.

    def __init__(
        self,
        annotations_file,
        image_root,
        transform=None,
        image_column="image",
        text_column="caption",
    ):

        self.annotations = pd.read_csv(annotations_file)
    
        self.image_root = Path(image_root)
    
        self.transform = transform
    
        self.image_column = image_column
        self.text_column = text_column

    def __len__(self):

        return len(self.annotations)

    def __getitem__(self, index):

        sample = self.annotations.iloc[index]

        image = Image.open(
            self.image_root / sample[self.image_column]
        ).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
        
        return {
            "image": image,
            "caption": sample[self.text_column],
        }
