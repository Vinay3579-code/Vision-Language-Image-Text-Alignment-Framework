# Frozen image and text encoders used by the Image-Text Alignment Framework.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from configs.config import (
    DEVICE,
    VISION_MODEL,
    VISION_PRETRAINED,
    TEXT_MODEL,
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

        self.encoder.eval()

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:

        embeddings = self.encoder(images)

        return F.normalize(embeddings, dim=-1)


class FrozenTextEncoder(nn.Module):
    """
    Phi-3.5 text encoder.

    Sentence embeddings are obtained by mean pooling the
    last hidden state.
    """

    def __init__(
        self,
        load_in_4bit: bool = True,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            TEXT_MODEL,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization = None

        if load_in_4bit and torch.cuda.is_available():

            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.encoder = AutoModelForCausalLM.from_pretrained(
            TEXT_MODEL,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None,
            quantization_config=quantization,
        )

        self.encoder.eval()

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def forward(
        self,
        texts: list[str],
    ) -> torch.Tensor:

        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        tokens = {
            key: value.to(DEVICE)
            for key, value in tokens.items()
        }

        outputs = self.encoder(
            **tokens,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden = outputs.hidden_states[-1]

        embeddings = hidden.mean(dim=1)

        return F.normalize(embeddings, dim=-1)
