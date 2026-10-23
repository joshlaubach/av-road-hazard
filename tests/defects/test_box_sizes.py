"""Per-country defect size in pixels, on a tiny fake dataset with hand-computed answers."""

import json
from pathlib import Path

import pytest

from tools.box_sizes import main, summarize_sizes


def xml(width: int, height: int, objects: list[tuple[str, int, int, int, int]]) -> str:
    items = "".join(
        f"<object><name>{n}</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin>"
        f"<xmax>{c}</xmax><ymax>{d}</ymax></bndbox></object>"
        for n, a, b, c, d in objects
    )
    return (
        f"<annotation><size><width>{width}</width><height>{height}</height></size>{items}"
        "</annotation>"
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    cases = {
        "Japan": {
            "a": (100, 100, [("D00", 0, 0, 10, 40), ("D40", 5, 5, 25, 25)]),
            "b": (100, 100, [("D00", 0, 0, 20, 80)]),
        },
        "Norway": {
            "c": (400, 200, [("D10", 0, 0, 80, 10), ("D44", 1, 1, 9, 9), ("D20", 0, 0, 0, 0)]),
            "empty": (400, 200, []),
        },
    }
    for country, files in cases.items():
        folder = tmp_path / country / "train" / "annotations" / "xmls"
        folder.mkdir(parents=True)
        for stem, (width, height, objects) in files.items():
            (folder / f"{stem}.xml").write_text(xml(width, height, objects))
    return tmp_path


def test_sizes_per_country_ignore_out_of_scope_and_invalid_boxes(root: Path) -> None:
    stats = summarize_sizes(root, tile_size=50)
    japan, norway = stats["Japan"], stats["Norway"]
    assert japan["images"] == 2 and japan["boxes"] == 3
    assert (japan["median_width"], japan["median_height"], japan["median_side"]) == (20, 40, 40)
    assert japan["share_larger_than_tile"] == pytest.approx(1 / 3)  # only the 80 px side
    assert japan["median_image_width"] == 100
    # Norway: D44 is out of scope and the 0 x 0 D20 is invalid, so one box (80 x 10) remains.
    assert norway["images"] == 2 and norway["boxes"] == 1
    assert norway["median_side"] == 80 and norway["share_larger_than_tile"] == 1.0
    assert norway["median_image_width"] == 400


def test_a_total_over_all_countries_is_included(root: Path) -> None:
    total = summarize_sizes(root, tile_size=50)["ALL"]
    assert total["images"] == 4 and total["boxes"] == 4


def test_a_country_without_boxes_reports_none_not_a_crash(tmp_path: Path) -> None:
    folder = tmp_path / "Czech" / "train" / "annotations" / "xmls"
    folder.mkdir(parents=True)
    (folder / "x.xml").write_text(xml(10, 10, []))
    czech = summarize_sizes(tmp_path, tile_size=50)["Czech"]
    assert czech["boxes"] == 0 and czech["median_side"] is None
    assert czech["share_larger_than_tile"] is None


def test_the_command_line_writes_the_json_report(root: Path, tmp_path: Path) -> None:
    out = tmp_path / "out" / "sizes.json"
    assert main(["--root", str(root), "--tile-size", "50", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["Japan"]["boxes"] == 3


def test_a_non_positive_tile_size_is_rejected(root: Path) -> None:
    with pytest.raises(ValueError, match="tile_size"):
        summarize_sizes(root, tile_size=0)
