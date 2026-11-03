"""Tests for tools/prepare_stellenbosch.py: which photos are picked and how they are cropped."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from tools import prepare_stellenbosch as prep


def photo(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (80, 80, 80)).save(path)


def test_the_crop_keeps_the_top_rows_at_full_width() -> None:
    out = prep.crop_to_road(Image.new("RGB", (3680, 2760)))
    assert out.size == (3680, prep.CROP_HEIGHT)


def test_the_crop_height_matches_the_committed_fixtures() -> None:
    sample = next(Path("tests/fixtures/stellenbosch").glob("*.jpg"))
    with Image.open(sample) as fixture:
        assert round(prep.CROP_HEIGHT * fixture.width / 3680) == fixture.height


def test_a_photo_too_short_to_hold_the_road_is_refused() -> None:
    with pytest.raises(ValueError, match="rows tall"):
        prep.crop_to_road(Image.new("RGB", (3680, 1000)))


def test_the_same_photos_as_the_fixtures_are_picked_by_name(tmp_path: Path) -> None:
    fixtures, raw, out = tmp_path / "fx", tmp_path / "raw", tmp_path / "out"
    photo(fixtures / "G1.jpg", (10, 5))
    photo(raw / "Subset" / "Train" / "G1.JPG", (3680, 2760))
    photo(raw / "Subset" / "Train" / "G2.JPG", (3680, 2760))
    code = prep.main(["--raw", str(raw), "--fixtures", str(fixtures), "--out", str(out)])
    assert code == 0 and [p.name for p in out.iterdir()] == ["G1.jpg"]
    with Image.open(out / "G1.jpg") as made:
        assert made.size == (3680, prep.CROP_HEIGHT)


def test_a_fixture_missing_from_the_raw_download_stops_the_run(tmp_path: Path) -> None:
    fixtures = tmp_path / "fx"
    photo(fixtures / "G9.jpg", (10, 5))
    (tmp_path / "raw").mkdir()
    with pytest.raises(SystemExit, match="G9"):
        prep.main(["--raw", str(tmp_path / "raw"), "--fixtures", str(fixtures)])
