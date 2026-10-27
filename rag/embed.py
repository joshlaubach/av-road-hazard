"""Turning photos and text into vectors that can be compared.

CLIP puts photos and sentences in one space, so a sentence can be matched against photos.
`open_clip` and torch are imported on first use, so importing this module stays light.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

MODEL = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
DIM = 512
BATCH = 32


class Embedder(Protocol):
    dim: int

    def embed_images(self, images: Sequence[np.ndarray]) -> np.ndarray: ...

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray: ...


class ClipEmbedder:
    """OpenCLIP ViT-B/32 (LAION-2B weights). Rows come back unit length (dot product = cosine)."""

    dim = DIM

    def __init__(self, device: str = "cpu") -> None:
        import open_clip
        import torch

        self._torch: Any = torch
        self._device = device
        model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
        self._model = model.eval().to(device)
        self._preprocess = preprocess
        self._tokenize = open_clip.get_tokenizer(MODEL)

    def _unit(self, features: Any) -> np.ndarray:
        out = features.float().cpu().numpy()
        unit: np.ndarray = out / np.linalg.norm(out, axis=1, keepdims=True)
        return unit

    def embed_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        from PIL import Image

        rows = []
        with self._torch.no_grad():
            for start in range(0, len(images), BATCH):
                batch = self._torch.stack(
                    [self._preprocess(Image.fromarray(i)) for i in images[start : start + BATCH]]
                ).to(self._device)
                rows.append(self._unit(self._model.encode_image(batch)))
        return np.concatenate(rows) if rows else np.zeros((0, DIM), dtype=np.float32)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        with self._torch.no_grad():
            for start in range(0, len(texts), BATCH):
                tokens = self._tokenize(list(texts[start : start + BATCH])).to(self._device)
                rows.append(self._unit(self._model.encode_text(tokens)))
        return np.concatenate(rows) if rows else np.zeros((0, DIM), dtype=np.float32)
