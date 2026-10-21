"""Synthetic bytes and annotations exercise identity and correction detection."""

from dataclasses import replace
from pathlib import Path

from PIL import Image

from defects.dataset import (
    Sample,
    build_dataset_manifest,
    build_samples,
    dataset_identity,
    verify_dataset_manifest,
)


def test_manifest_detects_bytes_and_labels(tmp_path: Path) -> None:
    image = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(image)
    samples = [Sample("a", image, (1, 0, 0, 0))]
    manifest = build_dataset_manifest(samples)
    original = dataset_identity(samples)
    assert verify_dataset_manifest(samples, manifest) == []
    corrected = [replace(samples[0], labels=(0, 1, 0, 0))]
    assert dataset_identity(corrected) != original
    assert verify_dataset_manifest(corrected, manifest) == ["a"]
    Image.new("RGB", (8, 8), "red").save(image)
    assert dataset_identity(samples) != original
    assert verify_dataset_manifest(samples, manifest) == ["a"]
    image.unlink()
    assert verify_dataset_manifest(samples, manifest) == ["a"]


def test_annotation_loader(tmp_path: Path) -> None:
    xml = tmp_path / "a.xml"
    xml.write_text(
        "<annotation>"
        + "".join(
            f"<object><name>{code}</name><bndbox><xmin>1</xmin><ymin>1</ymin>"
            "<xmax>5</xmax><ymax>5</ymax></bndbox></object>"
            for code in ["D00", "D00", "D40", "other"]
        )
        + "</annotation>"
    )
    image = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(image)
    samples, unexpected = build_samples(tmp_path, lambda _: image)
    assert samples == [Sample("a", image, (1, 0, 0, 1))]
    assert unexpected == {"other": 1}


def test_out_of_scope_only_images_are_excluded(tmp_path: Path) -> None:
    from defects.splits import make_splits

    # Include enough clean images to exercise grouped splitting after exclusion.
    cases = {"outside": ["D44"], "mixed": ["D00", "D44"], "clean": []}
    cases.update({f"clean{i}": [] for i in range(10)})
    for sample_id, codes in cases.items():
        objects = "".join(
            f"<object><name>{code}</name><bndbox><xmin>1</xmin><ymin>1</ymin>"
            "<xmax>5</xmax><ymax>5</ymax></bndbox></object>"
            for code in codes
        )
        (tmp_path / f"{sample_id}.xml").write_text(f"<annotation>{objects}</annotation>")
        Image.new("RGB", (8, 8)).save(tmp_path / f"{sample_id}.png")
    samples, report = build_samples(tmp_path, lambda xml: xml.with_suffix(".png"))
    assert report == {"D44": 2}
    assert report.excluded_ids == ["outside"]
    labels = {s.sample_id: s.labels for s in samples}
    assert "outside" not in labels
    assert labels["mixed"] == (1, 0, 0, 0)
    assert labels["clean"] == (0, 0, 0, 0)
    assert set(build_dataset_manifest(samples)) == set(labels)
    identity = dataset_identity(samples)
    Image.new("RGB", (8, 8), "red").save(tmp_path / "outside.png")
    assert dataset_identity(samples) == identity
    splits = make_splits(list(labels), lambda x: x, n_folds=3, seed=42)
    for train, val, heldout in splits.folds:
        assert set(train + val + heldout + splits.test) == set(labels)
