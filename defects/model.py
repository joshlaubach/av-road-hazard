"""Configurable timm multi-label classifier (logits; sigmoid at prediction time)."""

import json
from pathlib import Path
from typing import Any

from defects.dataset import file_sha256
from defects.labels import CODES


def build_model(cfg: dict[str, Any]) -> Any:
    import timm

    return timm.create_model(
        cfg["model_name"],
        pretrained=cfg["pretrained"],
        num_classes=len(CODES),
        drop_rate=cfg["head_dropout"],
        drop_path_rate=cfg["drop_path"],
    )


def load_checkpoint(checkpoint: Path, cfg: dict[str, Any], *, device: str = "cpu") -> Any:
    """Check byte integrity and class order before loading weights."""
    import torch

    sidecar = json.loads(checkpoint.with_suffix(".json").read_text())
    if sidecar["class_codes"] != list(CODES) or sidecar["checkpoint_sha256"] != file_sha256(
        checkpoint
    ):
        raise ValueError("Checkpoint provenance mismatch")
    model = build_model({**cfg, "pretrained": False}).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model
