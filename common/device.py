"""Compute device choice; torch is imported lazily because the base install has none."""

from __future__ import annotations


def select_device() -> str:
    """Best available device: cuda, then mps, then cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
