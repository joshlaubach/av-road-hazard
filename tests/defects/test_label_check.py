from __future__ import annotations

from pathlib import Path

from PIL import Image

from defects.voc import ALL, parse_annotation
from tools.label_check import format_table, main, sample_overlays, summarize


def _xml(width: int, height: int, objects: list[tuple[str, int, int, int, int]]) -> str:
    """Build a minimal Pascal VOC annotation string: an image size and a list of named boxes."""
    items = "".join(
        f"<object><name>{n}</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin>"
        f"<xmax>{c}</xmax><ymax>{d}</ymax></bndbox></object>"
        for n, a, b, c, d in objects
    )
    return (
        f"<annotation><size><width>{width}</width><height>{height}</height></size>{items}"
        "</annotation>"
    )


def _make_dataset(root: Path, with_images: bool = True) -> None:
    # Layout guess: <country>/train/{images,annotations/xmls}. The script must not depend on it.
    """Write a tiny fake dataset (two countries, XML plus gray images) under root."""
    cases = {
        "Japan": [
            ("a", [("D00", 0, 0, 10, 40), ("D40", 5, 5, 25, 25)]),
            ("b", [("D00", 0, 0, 20, 80)]),
        ],
        "Norway": [("c", [("D10", 0, 0, 80, 10), ("D44", 1, 1, 9, 9), ("D20", 0, 0, 0, 0)])],
    }
    for country, files in cases.items():
        xmls = root / country / "train" / "annotations" / "xmls"
        images = root / country / "train" / "images"
        xmls.mkdir(parents=True)
        images.mkdir(parents=True)
        for stem, objects in files:
            (xmls / f"{stem}.xml").write_text(_xml(100, 100, objects))
            if with_images:
                Image.new("RGB", (100, 100), "gray").save(images / f"{stem}.jpg")


def test_parse_annotation_returns_size_and_every_object(tmp_path: Path) -> None:
    """Parsing returns the image size and every object, including names outside the four codes."""
    path = tmp_path / "x.xml"
    path.write_text(_xml(640, 480, [("D40", 1, 2, 3, 4), ("other", 5, 6, 7, 8)]))
    size, boxes = parse_annotation(path)
    assert size == (640, 480)
    assert boxes == [("D40", 1, 2, 3, 4), ("other", 5, 6, 7, 8)]


def test_summarize_counts_shapes_per_country_and_code(tmp_path: Path) -> None:
    """Counts, aspect ratios and area shares are right per country and code; ALL adds up."""
    _make_dataset(tmp_path)
    summary = summarize(tmp_path)
    d00 = summary.stats[("Japan", "D00")]
    assert (d00.boxes, d00.images) == (2, 2)
    assert sorted(d00.aspect_ratios) == [0.25, 0.25]  # 10/40 and 20/80
    d40 = summary.stats[("Japan", "D40")]
    assert d40.aspect_ratios == [1.0]
    assert d40.area_shares == [400 / 10_000]
    assert summary.stats[("Norway", "D10")].aspect_ratios == [8.0]
    assert summary.stats[(ALL, "D00")].boxes == 2
    assert summary.annotation_files == 3


def test_unexpected_codes_and_invalid_boxes_are_reported_not_dropped(tmp_path: Path) -> None:
    """Codes outside the four, and zero-size boxes, are counted and shown instead of vanishing."""
    _make_dataset(tmp_path)
    summary = summarize(tmp_path)
    assert summary.unexpected["D44"] == 1
    assert summary.invalid_boxes == 1  # the zero-size D20
    assert ("Norway", "D20") not in summary.stats
    assert "D44=1" in format_table(summary)


def test_sample_overlays_is_seeded_and_limited(tmp_path: Path) -> None:
    """Sampling respects the per-code limit and gives identical files for the same seed."""
    data = tmp_path / "data"
    _make_dataset(data)
    first = sample_overlays(data, tmp_path / "o1", per_code=1, seed=7)
    second = sample_overlays(data, tmp_path / "o2", per_code=1, seed=7)
    assert first == second == {"D00": 1, "D10": 1, "D20": 0, "D40": 1}
    assert sorted(p.name for p in (tmp_path / "o1").iterdir()) == [
        "D00_00.jpg",
        "D10_00.jpg",
        "D40_00.jpg",
    ]


def test_sample_overlays_skips_boxes_without_an_image(tmp_path: Path) -> None:
    """Boxes whose image is missing are skipped, not crashed on."""
    data = tmp_path / "data"
    _make_dataset(data, with_images=False)
    assert sample_overlays(data, tmp_path / "o", per_code=5, seed=1) == dict.fromkeys(
        ("D00", "D10", "D20", "D40"), 0
    )


def test_main_needs_a_root(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Running without a folder (and without RDD2022_ROOT) is a usage error, exit code 2."""
    monkeypatch.delenv("RDD2022_ROOT", raising=False)
    assert main(["--out", str(tmp_path / "out")]) == 2


def test_main_writes_csv_and_samples(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """A full run exits 0, writes the CSV, and prints the unexpected-code line."""
    data = tmp_path / "data"
    _make_dataset(data)
    assert main([str(data), "--out", str(tmp_path / "out"), "--samples", "1"]) == 0
    assert (tmp_path / "out" / "label_stats.csv").is_file()
    assert "codes outside" in capsys.readouterr().out
