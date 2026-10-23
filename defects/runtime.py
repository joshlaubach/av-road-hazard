"""Device choice and the forward pass shared by training and inference."""

from __future__ import annotations

from typing import Any


def amp_enabled(cfg: dict[str, Any]) -> bool:
    """Whether to run under bf16 autocast; only "none" and "bf16" are supported."""
    mode = cfg.get("amp", "none")
    if mode not in ("none", "bf16"):
        raise ValueError(f"amp must be 'none' or 'bf16', got {mode!r}")
    return bool(mode == "bf16")


def forward_logits(model: Any, images: Any, device: str, cfg: dict[str, Any]) -> Any:
    """fp32 logits, computed under the optional bf16 autocast and channels_last layout."""
    import torch

    images = images.to(device)
    if cfg.get("channels_last", False):
        images = images.contiguous(memory_format=torch.channels_last)
    with torch.autocast(
        device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=amp_enabled(cfg)
    ):
        return model(images).float()


def select_device() -> str:
    """Best available device: cuda, then mps, then cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
