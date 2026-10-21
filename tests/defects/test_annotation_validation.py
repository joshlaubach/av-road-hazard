"""Whole-image exclusion uses generated images and synthetic VOC annotations only."""

import json
from pathlib import Path

import pytest
from PIL import Image

from defects.dataset import build_samples


def annotation(box: str, code: str = "D00") -> str:
    return f"<object><name>{code}</name>{box}</object>"


GOOD_BOX = "<bndbox><xmin>1</xmin><ymin>1</ymin><xmax>5</xmax><ymax>5</ymax></bndbox>"


@pytest.mark.parametrize(
    ("bad_object", "reason"),
    [
        (annotation(""), "missing_box"),
        (annotation(GOOD_BOX.replace("<xmin>1</xmin>", "")), "invalid_coordinates"),
        (annotation(GOOD_BOX.replace("<xmin>1", "<xmin>nan")), "invalid_coordinates"),
        (annotation(GOOD_BOX.replace("<xmin>1", "<xmin>oops")), "invalid_coordinates"),
        (annotation(GOOD_BOX.replace("<xmin>1", "<xmin>5")), "invalid_box"),
        (annotation(GOOD_BOX.replace("<xmin>1", "<xmin>-1")), "box_outside_image"),
        (annotation(GOOD_BOX.replace("<xmax>5", "<xmax>20")), "box_outside_image"),
    ],
)
def test_invalid_object_excludes_entire_image(tmp_path: Path, bad_object: str, reason: str) -> None:
    xml = tmp_path / "mixed.xml"
    original = f"<annotation>{annotation(GOOD_BOX)}{bad_object}</annotation>"
    xml.write_text(original)
    Image.new("RGB", (8, 8)).save(xml.with_suffix(".png"))
    samples, report = build_samples(tmp_path, lambda p: p.with_suffix(".png"))
    assert samples == []
    assert report.excluded_ids == ["mixed"]
    assert reason in report.exclusion_reasons["mixed"]
    assert xml.read_text() == original
    output = tmp_path / "report.json"
    report.write_json(output)
    saved = json.loads(output.read_text())
    assert saved["exclusions"] == [{"sample_id": "mixed", "reasons": [reason]}]
    assert saved["included_images"] == 0


@pytest.mark.parametrize("problem", ["missing", "corrupt", "xml", "size"])
def test_unusable_samples_are_reported_without_aborting(tmp_path: Path, problem: str) -> None:
    xml = tmp_path / "bad.xml"
    content = f"<annotation>{annotation(GOOD_BOX)}</annotation>"
    if problem == "xml":
        content = "<annotation>"
    if problem == "size":
        content = content.replace(
            "<annotation>", "<annotation><size><width>99</width><height>8</height></size>"
        )
    xml.write_text(content)
    image = xml.with_suffix(".png")
    if problem == "corrupt":
        image.write_bytes(b"not an image")
    elif problem != "missing":
        Image.new("RGB", (8, 8)).save(image)
    samples, report = build_samples(tmp_path, lambda p: p.with_suffix(".png"))
    assert samples == []
    assert report.excluded_ids == ["bad"]
    assert report.exclusion_reasons["bad"]


def test_valid_labels_and_true_negative_survive(tmp_path: Path) -> None:
    for name, objects in [("positive", annotation(GOOD_BOX, "D40")), ("negative", "")]:
        (tmp_path / f"{name}.xml").write_text(f"<annotation>{objects}</annotation>")
        Image.new("RGB", (8, 8)).save(tmp_path / f"{name}.png")
    samples, report = build_samples(tmp_path, lambda p: p.with_suffix(".png"))
    assert {s.sample_id: s.labels for s in samples} == {
        "negative": (0, 0, 0, 0),
        "positive": (0, 0, 0, 1),
    }
    assert report.excluded_ids == []


def test_cli_exports_only_valid_samples(tmp_path: Path) -> None:
    from defects.validate_annotations import main

    data = tmp_path / "data"
    data.mkdir()
    for name, box in [("valid", GOOD_BOX), ("invalid", "")]:
        (data / f"{name}.xml").write_text(f"<annotation>{annotation(box)}</annotation>")
        Image.new("RGB", (8, 8)).save(data / f"{name}.png")
    before = {p.name: p.read_bytes() for p in data.iterdir()}
    report = tmp_path / "out" / "report.json"
    index = tmp_path / "out" / "samples.json"
    assert main([str(data), "--report", str(report), "--sample-index", str(index)]) == 0
    rows = json.loads(index.read_text())
    assert [row["id"] for row in rows] == ["valid"]
    assert (index.parent / rows[0]["image_path"]).resolve() == data / "valid.png"
    assert json.loads(report.read_text())["excluded_images"] == 1
    assert {p.name: p.read_bytes() for p in data.iterdir()} == before
