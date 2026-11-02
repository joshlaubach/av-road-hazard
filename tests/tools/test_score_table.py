"""score_table: bands above each class threshold, review merging, and the tile-level summary."""

import json
from pathlib import Path

from eval.classifier import Alert
from tools.score_table import (
    BAND,
    class_table,
    count_by_call,
    format_report,
    in_band,
    in_band_only_tiles,
    load_reviews,
)

T = {"D00": 0.9, "D10": 0.75, "D20": 0.65, "D40": 0.75}


def alert(name: str, score: float, box: tuple[int, int, int, int] = (0, 0, 1, 1)) -> Alert:
    return Alert("f.jpg", box, name, score, False, 1.0)


def test_band_is_closed_at_the_threshold_and_open_at_the_top() -> None:
    assert in_band(0.75, 0.75)
    assert in_band(0.75 + BAND - 0.01, 0.75)
    assert not in_band(0.75 + BAND, 0.75)
    assert not in_band(0.74, 0.75)


def test_class_table_splits_by_band_and_call_and_marks_unreviewed() -> None:
    reviewed = {("f.jpg", (0, 0, 1, 1)): "No crack", ("f.jpg", (5, 5, 6, 6)): "Crack"}
    alerts = [
        alert("D10", 0.80),  # in band, No crack
        alert("D10", 0.99, (5, 5, 6, 6)),  # above band, Crack
        alert("D10", 0.76, (9, 9, 10, 10)),  # in band, never reviewed
    ]

    table = class_table(alerts, reviewed, T)["D10"]

    assert table["in_band"]["No crack"] == 1
    assert table["in_band"]["Unreviewed"] == 1
    assert table["above"]["Crack"] == 1


def test_a_tile_flagged_by_two_classes_keeps_alerting_if_one_alert_is_above_band() -> None:
    both_in = [alert("D10", 0.80), alert("D20", 0.70)]
    mixed = [alert("D10", 0.80), alert("D20", 0.99)]

    assert in_band_only_tiles(both_in, T) == {("f.jpg", (0, 0, 1, 1))}
    assert in_band_only_tiles(mixed, T) == set()


def test_d00_alerts_are_all_in_band_unless_treated_as_above() -> None:
    d00_only = [alert("D00", 0.99)]

    assert in_band_only_tiles(d00_only, T) == {("f.jpg", (0, 0, 1, 1))}
    assert in_band_only_tiles(d00_only, T, d00_above=True) == set()


def test_count_by_call_counts_distinct_tiles() -> None:
    reviewed = {("f.jpg", (0, 0, 1, 1)): "Crack"}
    tiles = {("f.jpg", (0, 0, 1, 1)), ("f.jpg", (2, 2, 3, 3))}

    assert count_by_call(tiles, reviewed) == {
        "Crack": 1,
        "No crack": 0,
        "Ambiguous": 0,
        "Not road": 0,
        "Unreviewed": 1,
    }


def write_batch(folder: Path, call: str) -> tuple[Path, Path]:
    key = folder / "key.json"
    calls = folder / "calls.csv"
    folder.mkdir()
    key.write_text(json.dumps({"t1": {"frame": "f.jpg", "box": [0, 0, 1, 1], "classes": {}}}))
    calls.write_text(f"tile_id,call\nt1,{call}\n")
    return key, calls


def test_a_later_batch_overrides_an_earlier_call_for_the_same_tile(tmp_path: Path) -> None:
    earlier = write_batch(tmp_path / "b2", "Ambiguous")
    later = write_batch(tmp_path / "b3", "No crack")

    assert load_reviews([earlier, later]) == {("f.jpg", (0, 0, 1, 1)): "No crack"}
    assert load_reviews([later, earlier]) == {("f.jpg", (0, 0, 1, 1)): "Ambiguous"}


def test_report_marks_d00_uninformative() -> None:
    alerts = [alert("D00", 0.95), alert("D10", 0.80, (5, 5, 6, 6))]

    text = format_report(alerts, {}, T)

    d00_section = text.split("D10")[0]
    assert "uninformative" in d00_section
    assert "uninformative" not in text.split("D10")[1].split("D20")[0]
