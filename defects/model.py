"""Configurable timm multi-label classifier (logits; sigmoid at prediction time)."""

from typing import Any

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
