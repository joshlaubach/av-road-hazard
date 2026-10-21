"""VOC samples, byte-level dataset identity and optional torch image loading."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from defects.label_check import iter_annotations
from defects.labels import CODES, class_index
from defects.validation import image_issues, read_annotation

Manifest = dict[str, dict[str, Any]]
# Keep provenance/split commands usable without the optional ML extra.
DatasetBase: Any = (
    importlib.import_module("torch.utils.data").Dataset
    if importlib.util.find_spec("torch") is not None
    else object
)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_path: Path
    labels: tuple[int, ...]


class AnnotationReport(Counter[str]):
    """Unexpected codes and whole-image exclusions, exportable as a JSON report."""

    def __init__(self) -> None:
        super().__init__()
        self.excluded_ids: list[str] = []
        self.exclusion_reasons: dict[str, list[str]] = {}
        self.included_images = 0

    def exclude(self, sample_id: str, reasons: list[str]) -> None:
        self.excluded_ids.append(sample_id)
        self.exclusion_reasons[sample_id] = sorted(set(reasons))

    def write_json(self, path: Path) -> None:
        """Save counts and the exclusion manifest; paths/ids are dataset-relative."""
        path.write_text(
            json.dumps(
                {
                    "included_images": self.included_images,
                    "excluded_images": len(self.excluded_ids),
                    "unexpected_codes": dict(sorted(self.items())),
                    "exclusions": [
                        {"sample_id": sid, "reasons": self.exclusion_reasons[sid]}
                        for sid in sorted(self.excluded_ids)
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


def build_samples(
    root: Path, image_fn: Callable[[Path], Path]
) -> tuple[list[Sample], AnnotationReport]:
    """Return usable samples and unexpected-code counts with excluded ids.

    Any structurally invalid annotation excludes the entire image, since this
    loader cannot mask uncertain regions. Out-of-scope-only annotations are
    excluded before image resolution. Valid empty annotations remain negatives;
    valid mixed-code annotations retain the existing in-scope-label policy.
    """
    samples: list[Sample] = []
    unexpected = AnnotationReport()
    for xml in iter_annotations(root):
        labels = [0] * len(CODES)
        boxes, size, issues = read_annotation(xml)
        sample_id = xml.relative_to(root).with_suffix("").as_posix()
        for code, *_ in boxes:
            if code in CODES:
                labels[class_index(code)] = 1
            else:
                unexpected[code] += 1
        if issues:
            unexpected.exclude(sample_id, issues)
            continue
        if boxes and not any(labels):
            unexpected.exclude(sample_id, ["out_of_scope_only"])
            continue
        try:
            image_path = image_fn(xml)
        except OSError:
            unexpected.exclude(sample_id, ["unreadable_image"])
            continue
        issues = image_issues(image_path, size, boxes)
        if issues:
            unexpected.exclude(sample_id, issues)
            continue
        samples.append(Sample(sample_id, image_path, tuple(labels)))
        unexpected.included_images += 1
    return samples, unexpected


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_dataset_manifest(samples: Sequence[Sample]) -> Manifest:
    manifest: Manifest = {}
    for sample in sorted(samples, key=lambda s: s.sample_id):
        if sample.sample_id in manifest:
            raise ValueError(f"Duplicate sample id: {sample.sample_id}")
        if len(sample.labels) != len(CODES) or any(x not in (0, 1) for x in sample.labels):
            raise ValueError("Expected four binary labels")
        manifest[sample.sample_id] = {
            "sha256": file_sha256(sample.image_path),
            "labels": list(sample.labels),
        }
    return manifest


def verify_dataset_manifest(samples: Sequence[Sample], manifest: Manifest) -> list[str]:
    """Return changed, missing and extra ids; missing image files are mismatches too."""
    seen: set[str] = set()
    mismatches = set(manifest) - {s.sample_id for s in samples}
    for sample in samples:
        if sample.sample_id in seen:
            raise ValueError(f"Duplicate sample id: {sample.sample_id}")
        seen.add(sample.sample_id)
        try:
            actual = build_dataset_manifest([sample])[sample.sample_id]
        except OSError:
            mismatches.add(sample.sample_id)
            continue
        if actual != manifest.get(sample.sample_id):
            mismatches.add(sample.sample_id)
    return sorted(mismatches)


def dataset_identity(samples: Sequence[Sample]) -> str:
    return manifest_identity(build_dataset_manifest(samples))


def manifest_identity(manifest: Manifest) -> str:
    """Derive identity from already hashed bytes and labels without reading images again."""
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


class DefectDataset(DatasetBase):
    """ImageNet normalization; modest training-only horizontal/color augmentation."""

    def __init__(self, samples: Sequence[Sample], cfg: dict[str, Any], *, training: bool) -> None:
        from torchvision import transforms

        self.samples = list(samples)
        ops: list[Any] = [transforms.Resize((cfg["input_size"], cfg["input_size"]))]
        if training:
            ops.extend(
                [
                    transforms.RandomHorizontalFlip(cfg["horizontal_flip"]),
                    transforms.ColorJitter(
                        brightness=cfg["color_jitter"],
                        contrast=cfg["color_jitter"],
                        saturation=cfg["color_jitter"],
                    ),
                ]
            )
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        import torch

        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.tensor(sample.labels, dtype=torch.float32)
